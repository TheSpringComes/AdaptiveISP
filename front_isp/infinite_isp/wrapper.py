"""Infinite-ISP Front ISP wrapper（第三方接入层，真实实现）。

对应：
    front_isp:
      enabled: true
      type: external
      external:
        backend: infinite_isp
        repo_path:   front_isp/third_party/Infinite-ISP
        config_path: front_isp/third_party/Infinite-ISP/config/configs.yml

设计约束（见 V3.1 规划）：
  - 第三方项目本身不修改，只通过本 wrapper 接入。
  - wrapper 模块 import 必须始终安全（注册表要能加载）：真正的第三方
    import 全部延迟到 `_load_backend()`，在仓库缺失时抛出带指引的错误。

Infinite-ISP 的官方入口 (`infinite_isp.py`) 是面向 RAW 文件的批处理
管线：`InfiniteISP(data_path, config_path)` 从磁盘读 `.raw`/DNG，跑完整
模块链（DPC → BLC → LSC → BNR → AWB → WB → Demosaic → CCM → Gamma →
CSC → LDCI → Sharpen → NR2D → RGB），把结果 png 写到 `out_frames/`。

本 wrapper 的接入契约：
  1. 每个输入样本：linear RGB (3,H,W) [0,1] → 重打包为 RGGB mosaic
     uint16 `.raw` 临时文件（绿通道复制到两个 G 位——已是 demosaic 后
     数据的最简逆操作）；
  2. 从仓库默认 `config/configs.yml` 生成逐样本参数 YAML：patch
     尺寸/位深/bayer pattern，关掉 BLC（输入已归一化）/AE（确定性基线）/
     crop/scale；
  3. patch `util.utils.OUTPUT_DIR` 到临时目录，`run_pipeline()` 后从
     那里读回输出 png（matplotlib imread → float [0,1] RGB）。
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
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
        bit_depth:   输出 mosaic 的量化位深（默认 16）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.repo_path = os.path.expanduser(
            str(self.config.get('repo_path',
                                'front_isp/third_party/Infinite-ISP')))
        self.config_path = self.config.get(
            'config_path',
            os.path.join(self.repo_path, 'config', 'configs.yml'))
        self.bit_depth = int(self.config.get('bit_depth', 12))  # 仓库 gamma 模块只提供 8/10/12/14-bit LUT
        self._mod = None          # InfiniteISP 类（延迟加载）
        self._util = None         # 仓库 util.utils 模块（patch OUTPUT_DIR）
        self._base_cfg = None     # 仓库默认参数 dict
        self._load_backend()

    # ---------------- third-party loading (deferred & guarded) ----------------

    def _load_backend(self):
        """把第三方仓库挂到 sys.path 并验证可导入。失败时给出可操作的报错。"""
        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(
                f"Infinite-ISP 仓库未找到: {self.repo_path}\n"
                f"  git clone {_REPO_URL} {self.repo_path}\n"
                f"  并在 front_isp.external.repo_path 中指向它。")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f"Infinite-ISP 参数 YAML 未找到: {self.config_path}\n"
                f"  从仓库 config/ 目录复制一份并按需调整。")
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        try:
            import util.utils as _su  # noqa: F401  (仓库顶层 util 包)
            spec = importlib.util.spec_from_file_location(
                "infinite_isp_backend",
                os.path.join(self.repo_path, "infinite_isp.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._mod = mod
            self._util = _su
            with open(self.config_path, "r", encoding="utf-8") as fh:
                import yaml
                self._base_cfg = yaml.safe_load(fh)
            if not isinstance(self._base_cfg, dict):
                raise ValueError(
                    f"Infinite-ISP 参数 YAML 为空或非映射: {self.config_path}\n"
                    f"  应为其 config/configs.yml 的完整拷贝（"
                    f"占位注释文件不行）。")
        except ImportError as exc:  # pragma: no cover - 依赖第三方仓库存在
            raise ImportError(
                f"Infinite-ISP 导入失败 ({exc!r})。请检查仓库版本与依赖。"
            ) from exc

    # ---------------- conversion helpers ----------------

    @staticmethod
    def _rgb_to_rggb_u16(image: np.ndarray, bit_depth: int) -> np.ndarray:
        """(H, W, 3) float [0,1] linear RGB → (H, W) uint16 RGGB mosaic。

        从全分辨率 RGB 直接重采样到 RGGB CFA 网格（R→(0::2,0::2)，
        G→两个 G 位，B→(1::2,1::2)），维度不变——Infinite-ISP 内部
        demosaic 后正好还原到输入分辨率。"""
        h, w, _ = image.shape
        if h % 2 or w % 2:
            h, w = h - (h % 2), w - (w % 2)
            image = image[:h, :w]
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        mosaic = np.zeros((h, w), dtype=np.float32)
        mosaic[0::2, 0::2] = r[0::2, 0::2]
        mosaic[0::2, 1::2] = g[0::2, 1::2]
        mosaic[1::2, 0::2] = g[1::2, 0::2]
        mosaic[1::2, 1::2] = b[1::2, 1::2]
        scale = float(2 ** bit_depth - 1)
        return (np.clip(mosaic, 0.0, 1.0) * scale).astype(np.uint16)

    @staticmethod
    def _grayworld_gains(image: np.ndarray) -> tuple[float, float]:
        """(H, W, 3) [0,1] RGB → (r_gain, b_gain)，锚定绿通道。

        与 fixed 链 `wb: auto: grayworld` / isp.operators 的 _grayworld_gain
        同一语义：gain_c = mean(G) / mean(c)。"""
        mu = image.reshape(-1, 3).mean(axis=0) + 1e-6
        return float(mu[1] / mu[0]), float(mu[1] / mu[2])

    def _make_cfg(self, h: int, w: int, tmpdir: str,
                   grayworld_gains: tuple[float, float] = (1.0, 1.0)) -> str:
        """从仓库默认参数生成逐样本 YAML（确定性 baseline：关 BLC/AE）。"""
        import copy
        import yaml
        cfg = copy.deepcopy(self._base_cfg)
        cfg.setdefault('platform', {})
        cfg['platform']['filename'] = 'input.raw'
        cfg['platform']['render_3a'] = False
        cfg['platform']['save_format'] = 'png'
        cfg.setdefault('sensor_info', {})
        cfg['sensor_info']['bayer_pattern'] = 'rggb'
        cfg['sensor_info']['bit_depth'] = self.bit_depth
        cfg['sensor_info']['range'] = 2 ** self.bit_depth - 1
        cfg['sensor_info']['width'] = w
        cfg['sensor_info']['height'] = h
        # 输入已做 black/white 归一化：关掉 BLC 防止二次减 black level
        cfg.setdefault('black_level_correction', {})['is_enable'] = False
        # 确定性基线：关自动曝光
        cfg.setdefault('auto_exposure', {})['is_enable'] = False
        # 固定数字增益 1（is_auto 走 ae_feedback，手动走 current_gain）
        cfg.setdefault('digital_gain', {})['is_auto'] = False
        cfg['digital_gain']['current_gain'] = 0
        # 输入是未白平衡的 linear camera RGB：由 wrapper 计算 gray-world
        # 增益写入 WB 模块（与 fixed 链的 `wb: auto: grayworld` 同一语义，
        # 确定性、不依赖仓库的 3A 反馈循环）。仓库默认增益 r=1.246/b=2.81
        # 是其测试传感器标定的，必须覆盖；AWB 模块本身关掉，其增益要经
        # render_3a + digital_gain.is_auto + AE 循环才能写回 WB，链条脆且
        # 不可控，等价结果由 wrapper 直接给出。
        cfg.setdefault('auto_white_balance', {})['is_enable'] = False
        cfg.setdefault('white_balance', {})['is_auto'] = False
        r_gain, b_gain = grayworld_gains
        cfg['white_balance']['r_gain'] = float(r_gain)
        cfg['white_balance']['b_gain'] = float(b_gain)
        # CCM 换成单位阵（仓库默认矩阵是其测试传感器标定的）
        cfg.setdefault('color_correction_matrix', {})['is_enable'] = True
        cfg['color_correction_matrix']['corrected_red'] = [1.0, 0.0, 0.0]
        cfg['color_correction_matrix']['corrected_green'] = [0.0, 1.0, 0.0]
        cfg['color_correction_matrix']['corrected_blue'] = [0.0, 0.0, 1.0]
        # 输出 RGB、不缩放不裁剪
        cfg.setdefault('rgb_conversion', {})['is_enable'] = True
        cfg.setdefault('scale', {})['is_enable'] = False
        cfg.setdefault('crop', {})['is_enable'] = False
        cfg_path = os.path.join(tmpdir, 'configs.yml')
        with open(cfg_path, 'w', encoding='utf-8') as fh:
            yaml.dump(cfg, fh, sort_keys=False)
        return cfg_path

    # ---------------- FrontISPBase API ----------------

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        b, _, height, width = image.shape
        outs = []
        with tempfile.TemporaryDirectory(prefix='infinite_isp_') as tmpdir:
            out_dir = os.path.join(tmpdir, 'out_frames')
            os.makedirs(out_dir, exist_ok=True)
            saved_output_dir = self._util.OUTPUT_DIR
            self._util.OUTPUT_DIR = out_dir + os.sep
            try:
                for i in range(b):
                    rgb = image[i].detach().cpu().numpy().transpose(1, 2, 0)
                    rgb_clipped = np.clip(rgb, 0.0, 1.0)
                    raw_u16 = self._rgb_to_rggb_u16(
                        rgb_clipped, self.bit_depth)
                    raw_u16.tofile(os.path.join(tmpdir, 'input.raw'))

                    cfg_path = self._make_cfg(
                        height, width, tmpdir,
                        grayworld_gains=self._grayworld_gains(rgb_clipped))
                    pipe = self._mod.InfiniteISP(tmpdir, cfg_path)
                    pipe.load_raw()
                    pipe.run_pipeline(visualize_output=True)

                    produced = sorted(
                        p for p in os.listdir(out_dir)
                        if p.startswith('Out_input') and p.endswith('.png'))
                    if not produced:
                        raise RuntimeError(
                            "Infinite-ISP 未产生输出 png（检查其参数配置）")
                    out_path = os.path.join(out_dir, produced[-1])
                    import matplotlib.image as mpimg
                    out = mpimg.imread(out_path).astype(np.float32)[..., :3]
                    outs.append(torch.from_numpy(
                        out.transpose(2, 0, 1)).to(image.device, image.dtype))
            finally:
                self._util.OUTPUT_DIR = saved_output_dir
        return torch.stack(outs, dim=0).clamp(0.0, 1.0)


__all__ = ["InfiniteISPFront"]
