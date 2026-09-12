"""Infinite-ISP Front ISP wrapper（第三方接入层）。

对应：
    front_isp:
      enabled: true
      type: infinite_isp
      infinite_isp:
        repo_path:   <path to https://github.com/10x-Engineers/Infinite-ISP clone>
        config_path: configs/front_isp/infinite.yaml   # Infinite-ISP 自己的参数 YAML

设计约束（见 Pipeline 扩展方案 2.1）：
  - 第三方项目本身不修改，只通过本 wrapper 接入。
  - wrapper 模块 import 必须始终安全（注册表要能加载）：真正的第三方
    import 全部延迟到 `_load_backend()`，在仓库缺失时抛出带指引的错误。

Infinite-ISP 的官方入口 (`Infinite-ISP.py`) 是一个面向 PNG 文件的批处理
脚本，没有 in-memory Python API。因此本 wrapper 的接入契约是：

  1. `repo_path` 指向仓库 clone；
  2. `config_path` 是复制自该仓库 `config/` 并按需调整的参数 YAML
     （is_enable / 算法类型 / 参数 — Bayer denoise、WB、contrast、
     sharpen、2D denoise、scaling 等，全部由该 YAML 控制）；
  3. `process()` 内部完成 torch→numpy→(临时 RGGB RAW)→Infinite-ISP→
     sRGB→torch 的转换。具体到仓库版本 API 的粘合代码在
     `_run_pipeline()` 中补全（仓库不在本机上时会在构建期给出明确报错）。
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp

_REPO_URL = "https://github.com/10x-Engineers/Infinite-ISP"


@register_front_isp('infinite_isp')
class InfiniteISPFront(FrontISPBase):
    """Infinite-ISP (10x-Engineers) 的 Front ISP wrapper。

    config 项（全部可选，除 repo_path 外均有默认）：
        repo_path:   Infinite-ISP 仓库的本地 clone 路径
        config_path: Infinite-ISP 内部参数 YAML（保留其自身全部可配置能力）
        output_dir:  wrapper 工作目录（中间 RAW/RGGB 缓存），默认 `/tmp`
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.repo_path = os.path.expanduser(
            str(self.config.get('repo_path', 'third_party/Infinite-ISP')))
        self.config_path = self.config.get('config_path',
                                           'configs/front_isp/infinite.yaml')
        self.output_dir = self.config.get('output_dir', '/tmp')
        self._backend = None
        self._load_backend()

    # ---------------- third-party loading (deferred & guarded) ----------------

    def _load_backend(self):
        """把第三方仓库挂到 sys.path 并验证可导入。失败时给出可操作的报错。"""
        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(
                f"Infinite-ISP 仓库未找到: {self.repo_path}\n"
                f"  git clone {_REPO_URL} <repo_path>\n"
                f"  并在 front_isp.infinite_isp.repo_path 中指向它。"
            )
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f"Infinite-ISP 参数 YAML 未找到: {self.config_path}\n"
                f"  从仓库 config/ 目录复制一份到 configs/front_isp/ 并按需调整。"
            )
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        try:
            # Infinite-ISP 的 pipeline 主入口（按仓库结构导入）。
            self._backend = importlib.import_module('Infinite-ISP')
        except ImportError as exc:  # pragma: no cover - 依赖第三方仓库存在
            raise ImportError(
                f"Infinite-ISP 导入失败 ({exc!r})。请检查仓库版本与依赖。"
            ) from exc

    # ---------------- conversion helpers ----------------

    @staticmethod
    def _torch_to_rggb_uint16(image: torch.Tensor) -> np.ndarray:
        """(B, 3, H, W) [0,1] linear -> (B, 4, H, W) uint16 RGGB（近邻复制绿通道）。

        Infinite-ISP 消费 mosaic RAW；本框架的 simulated RAW 是 3 通道
        linear RGB。绿通道用双份复制，是 demosaic-averaging 的逆操作的
        最简近似。"""
        b, _, h, w = image.shape
        x = image.detach().cpu().numpy()
        g = 0.5 * (x[:, 1:2] + x[:, 1:2])  # single G plane duplicated
        rggb = np.concatenate([x[:, 0:1], g, g, x[:, 2:3]], axis=1)
        return (np.clip(rggb, 0.0, 1.0) * 65535.0).astype(np.uint16)

    @staticmethod
    def _rgb_to_torch(rgb: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 sRGB -> (1, 3, H, W) float [0,1]。"""
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    # ---------------- FrontISPBase API ----------------

    def _run_pipeline(self, rggb: np.ndarray) -> np.ndarray:
        """对单张 RGGB uint16 RAW 跑 Infinite-ISP，返回 (H, W, 3) uint8 sRGB。

        与具体仓库版本相关的粘合代码集中在这里。"""
        if self._backend is None:  # pragma: no cover
            raise RuntimeError("Infinite-ISP backend not loaded")
        raise NotImplementedError(
            "Infinite-ISP wrapper 的 _run_pipeline() 需要在接入真实仓库后补全：\n"
            f"  1. git clone {_REPO_URL} {self.repo_path}\n"
            "  2. 按 Infinite-ISP 的 in-memory 调用方式（或临时文件 + "
            "Infinite-ISP.py 入口）在这里实现 RAW→sRGB 的调用。"
        )

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        b, _, h, w = image.shape
        rggb = self._torch_to_rggb_uint16(image)
        outs = []
        for i in range(b):
            rgb = self._run_pipeline(rggb[i])
            outs.append(self._rgb_to_torch(rgb).to(image.device, image.dtype))
        return torch.cat(outs, dim=0)


__all__ = ["InfiniteISPFront"]
