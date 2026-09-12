"""Samsung Modular Neural ISP Front ISP wrapper（第三方接入层）。

对应：
    front_isp:
      enabled: true
      type: modular_neural_isp
      modular_neural_isp:
        repo_path:   <path to https://github.com/SamsungLabs/modular_neural_isp clone>
        config_path: configs/front_isp/modular_neural_isp.yaml
        weights_dir: <module checkpoints>

设计约束（见 Pipeline 扩展方案 2.1）：
  - 第三方项目本身不修改，只通过本 wrapper 接入。
  - import 始终安全：第三方依赖延迟到 `_load_backend()`。

Samsung Modular Neural ISP 官方实现已按阶段模块化：
    denoising / awb_ccm / photofinishing / upsampling / enhancement
每个阶段独立训练与配置（见其 README 的 modular training）。本 wrapper 的
接入契约：
  1. `repo_path` 指向仓库 clone；
  2. `config_path` 保留其自身的分阶段模块配置（哪些阶段启用、各阶段
     checkpoint 路径、内部超参）；
  3. `process()` 完成 torch→RAW 输入格式→逐阶段前向→sRGB→torch。
     与具体仓库版本相关的粘合代码集中在 `_run_pipeline()`。
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import torch

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp

_REPO_URL = "https://github.com/SamsungLabs/modular_neural_isp"

# 官方划分的五个模块化阶段。config 里可按阶段开关/指定 checkpoint。
_STAGES = ("denoising", "awb_ccm", "photofinishing", "upsampling", "enhancement")


@register_front_isp('modular_neural_isp')
class ModularNeuralISPFront(FrontISPBase):
    """Samsung Modular Neural ISP 的 Front ISP wrapper。

    config 项（全部可选，除 repo_path 外均有默认）：
        repo_path:   仓库的本地 clone 路径
        config_path: 分阶段模块配置 YAML（保留其自身可配置能力）
        weights_dir: 各阶段 checkpoint 根目录
        stages:      启用的阶段子集（默认全部五个）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        # 仓库内已有 vendored 副本（`isp/third_party/modular_neural_isp`，
        # `n_*` RL 算子所用）；也可指向外部 clone。
        vendored = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            'isp', 'third_party', 'modular_neural_isp')
        self.repo_path = os.path.expanduser(
            str(self.config.get('repo_path', vendored)))
        self.config_path = self.config.get(
            'config_path', 'configs/front_isp/modular_neural_isp.yaml')
        self.weights_dir = self.config.get('weights_dir', None)
        self.stages = tuple(self.config.get('stages', list(_STAGES)))
        self._backend = None
        self._load_backend()

    # ---------------- third-party loading (deferred & guarded) ----------------

    def _load_backend(self):
        """把第三方仓库挂到 sys.path 并验证可导入。失败时给出可操作的报错。"""
        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(
                f"modular_neural_isp 仓库未找到: {self.repo_path}\n"
                f"  git clone {_REPO_URL} <repo_path>\n"
                f"  并在 front_isp.modular_neural_isp.repo_path 中指向它。"
            )
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f"modular_neural_isp 配置 YAML 未找到: {self.config_path}\n"
                f"  按官方 modular 阶段 ({'/'.join(_STAGES)}) 建立配置后指向它。"
            )
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        # 只做结构校验（stage 目录存在）；真正的第三方 import 延迟到
        # `_run_pipeline()` — 仓库的 `utils/` 会与 yolov3 的同名包冲突，
        # 且其依赖（如 exiftool）可能缺席，接入时参考
        # `isp/learned/samsung_modular/backend.py::_samsung_import` 的
        # sys.modules 隔离做法。
        missing_stages = [s for s in self.stages
                          if not os.path.isdir(os.path.join(self.repo_path, s))]
        if missing_stages:
            raise FileNotFoundError(
                f"modular_neural_isp 仓库缺少阶段目录 {missing_stages}: "
                f"{self.repo_path}（可用阶段: {'/'.join(_STAGES)}）"
            )
        self._backend = True

    # ---------------- FrontISPBase API ----------------

    def _run_pipeline(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        """对 (1, C, H, W) RAW 跑启用的模块化阶段，返回 (1, 3, H', W') sRGB。

        与具体仓库版本相关的粘合代码集中在这里。"""
        if self._backend is None:  # pragma: no cover
            raise RuntimeError("modular_neural_isp backend not loaded")
        raise NotImplementedError(
            "modular_neural_isp wrapper 的 _run_pipeline() 需要在接入真实仓库后补全：\n"
            f"  1. git clone {_REPO_URL} {self.repo_path}\n"
            "  2. 训练/下载各模块 checkpoint，在 config 中配置 stages 与 weights_dir；\n"
            "  3. 按其分阶段 API 在这里实现 RAW→sRGB 的逐阶段前向。"
        )

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        return self._run_pipeline(image, metadata)


__all__ = ["ModularNeuralISPFront"]
