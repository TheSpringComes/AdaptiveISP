"""Infinite-ISP Front ISP wrapper（第三方接入层，真实实现）。

后端使用 `front_isp/third_party/InfiniteISP_RAW`（上游 10x-Engineers/
Infinite-ISP 的整理版：算法文件与上游完全一致，附带已验证的基线参数
`config/hikrobot_baseline.yml`——WB/AWB/CCM/BLC/LSC/LDCI/AE/crop/scale
等传感器相关模块全部关闭，只保留 Bayer 降噪 → Malvar 去马赛克 →
gamma 2.2 → YUV 域锐化/NLM → RGB）。

本 wrapper 参照其便携入口 `process_raw.py` 的接入方式：
  1. 逐样本 patch sensor_info（尺寸/位深/pattern）并按位深重新生成
     gamma LUT（公式 `round(max*(i/max)^(1/2.2))`，同 process_raw.py）；
  2. platform 设 render_3a=False / disable_progress_bar，全部模块
     is_save=False，monkeypatch `util.save_pipeline_output` 直接捕获
     内存中的 uint8 RGB 输出（不走 PNG 落盘再读回）；
  3. 构建期校验基线配置：传感器标定模块必须保持关闭（防止误指向
     上游 configs.yml——其 WB/CCM 是其测试传感器标定值）；
  4. 可选 GPU：`device: gpu` 启用 optional_gpu 的联合双边滤波/NLM 加速。

输入域适配（本仓库数据流约定）：
  Dataset 输出的 demosaic linear camera RGB 带照明体色偏；InfiniteISP_RAW
  基线假定 RAW 已含相机固定白平衡状态（METHOD.md）。因此 wrapper 在
  重打包 Bayer 前先做 gray-world 白平衡（锚定绿通道，与 fixed 链/
  isp.operators 的 _grayworld_gain 同一语义）——等价于在 RAW 域做 WB，
  但保持第三方基线配置零改动。
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp

_REPO_URL = "https://github.com/10x-Engineers/Infinite-ISP"

# 这些模块承载传感器/相机标定值，对非标定输入必须保持关闭（基线已如此）。
_SENSOR_CALIBRATED_MODULES = (
    'auto_white_balance', 'white_balance', 'color_correction_matrix',
    'black_level_correction', 'lens_shading_correction', 'ldci', 'oecf',
    'auto_exposure', 'dead_pixel_correction', 'crop', 'scale',
)


@register_front_isp('infinite_isp')
class InfiniteISPFront(FrontISPBase):
    """InfiniteISP_RAW 基线的 Front ISP wrapper。

    config 项（除 repo_path 外均有默认）：
        repo_path:   InfiniteISP_RAW 仓库的本地 clone 路径
        config_path: 基线参数 YAML（默认其 config/hikrobot_baseline.yml）
        bit_depth:   输出 mosaic 的量化位深（默认 12；gamma LUT 按此重生成）
        device:      'cpu' | 'gpu'（gpu 启用 optional_gpu 加速，默认 cpu）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.repo_path = os.path.expanduser(str(self.config.get(
            'repo_path', 'front_isp/third_party/InfiniteISP_RAW')))
        self.config_path = self.config.get(
            'config_path',
            os.path.join(self.repo_path, 'config', 'hikrobot_baseline.yml'))
        self.bit_depth = int(self.config.get('bit_depth', 12))  # 仓库 gamma 模块只提供 8/10/12/14-bit LUT
        self.device = str(self.config.get('device', 'cpu')).lower()
        if self.device not in ('cpu', 'gpu'):
            raise ValueError(f"Infinite-ISP device 应为 'cpu'|'gpu'，得到 {self.device!r}")
        self._mod = None          # InfiniteISP 类（延迟加载）
        self._util = None         # 仓库 util.utils 模块（monkeypatch 保存）
        self._base_cfg = None     # 基线参数 dict
        self._load_backend()

    # ---------------- third-party loading (deferred & guarded) ----------------

    def _load_backend(self):
        """加载 InfiniteISP_RAW 仓库并验证可导入。失败时给出可操作的报错。"""
        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(
                f"InfiniteISP_RAW 仓库未找到: {self.repo_path}\n"
                f"  参照 METHOD.md 准备该目录（上游 {_REPO_URL} + 基线配置），\n"
                f"  并在 front_isp.external.repo_path 中指向它。")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f"InfiniteISP_RAW 基线参数 YAML 未找到: {self.config_path}\n"
                f"  应为其 config/hikrobot_baseline.yml。")
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
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as fh:
                self._base_cfg = yaml.safe_load(fh)
            if not isinstance(self._base_cfg, dict):
                raise ValueError("基线 YAML 为空或非映射")
        except ImportError as exc:  # pragma: no cover - 依赖第三方仓库存在
            raise ImportError(
                f"InfiniteISP_RAW 导入失败 ({exc!r})。请检查仓库版本与依赖。"
            ) from exc
        # 构建期校验：传感器标定模块必须关闭，输出必须为 RGB。
        for name in _SENSOR_CALIBRATED_MODULES:
            if self._base_cfg.get(name, {}).get('is_enable'):
                raise ValueError(
                    f"基线配置 {self.config_path} 中 {name}.is_enable=true：\n"
                    f"  该模块承载传感器标定值，对未标定输入会色彩爆炸。\n"
                    f"  请使用 InfiniteISP_RAW 的 hikrobot_baseline.yml。")
        if not self._base_cfg.get('rgb_conversion', {}).get('is_enable', False):
            raise ValueError("基线配置必须开启 rgb_conversion（本 wrapper 只接 RGB 输出）")
        if self._base_cfg.get('yuv_conversion_format', {}).get('is_enable', False):
            raise ValueError("基线配置必须关闭 yuv_conversion_format（YUV 打包输出不适用）")

    # ---------------- conversion helpers ----------------

    @staticmethod
    def _grayworld_gains(image: np.ndarray) -> np.ndarray:
        """(H, W, 3) [0,1] RGB → (3,) 通道增益，锚定绿通道（限幅防极端）。

        与 isp.operators 的 _grayworld_gain 同一语义：gain_c = mean(G)/mean(c)。"""
        mu = image.reshape(-1, 3).mean(axis=0) + 1e-6
        gain = mu[1] / mu
        return np.clip(gain, 0.25, 4.0).astype(np.float32)

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

    def _make_cfg(self, h: int, w: int, tmpdir: str) -> str:
        """从基线生成逐样本 YAML（参照 process_raw.py 的 patch 方式）。"""
        import copy
        import yaml
        cfg = copy.deepcopy(self._base_cfg)
        bits = self.bit_depth
        maximum = 2 ** bits - 1
        # sensor_info：本样本尺寸/位深；重打包为 RGGB
        cfg.setdefault('sensor_info', {}).update(
            width=w, height=h, bit_depth=bits, range=maximum,
            bayer_pattern='rggb')
        # gamma LUT 按位深重新生成（公式同 process_raw.py；基线自带的
        # gamma_lut_12 并非 1/2.2 曲线，是 8-bit 数据在 12-bit 容器下的标定）
        lut = np.rint(maximum * (np.arange(maximum + 1) / maximum) ** (1 / 2.2)).astype(int)
        cfg.setdefault('gamma_correction', {})[f'gamma_lut_{bits}'] = lut.tolist()
        # platform：确定性、无 3A、无进度条
        cfg.setdefault('platform', {}).update(
            filename='input.raw', render_3a=False,
            disable_progress_bar=True, save_format='png')
        # 全部模块关闭落盘
        for section in cfg.values():
            if isinstance(section, dict) and 'is_save' in section:
                section['is_save'] = False
        cfg_path = os.path.join(tmpdir, 'configs.yml')
        with open(cfg_path, 'w', encoding='utf-8') as fh:
            yaml.dump(cfg, fh, sort_keys=False)
        return cfg_path

    # ---------------- FrontISPBase API ----------------

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        b, _, height, width = image.shape
        outs = []
        accelerator = nullcontext()
        if self.device == 'gpu':
            from optional_gpu.denoise import accelerated_denoisers
            accelerator = accelerated_denoisers()
        saved_save = self._util.save_pipeline_output
        try:
            with accelerator, tempfile.TemporaryDirectory(prefix='infisp_') as tmpdir:
                for i in range(b):
                    rgb = image[i].detach().cpu().numpy().transpose(1, 2, 0)
                    rgb = np.clip(rgb, 0.0, 1.0)
                    # 输入域适配：gray-world WB（等价于 RAW 域 WB，见模块 docstring）
                    rgb = rgb * self._grayworld_gains(rgb)
                    rgb = np.clip(rgb, 0.0, 1.0)
                    self._rgb_to_rggb_u16(rgb, self.bit_depth).tofile(
                        os.path.join(tmpdir, 'input.raw'))

                    cfg_path = self._make_cfg(height, width, tmpdir)
                    captured: list[np.ndarray] = []

                    def _capture(_name, out_rgb, _config,
                                 _sink=captured):
                        # process_raw.py 的输出契约：uint8 (H, W, 3) RGB
                        if out_rgb.dtype != np.uint8 or out_rgb.ndim != 3 \
                                or out_rgb.shape[2] != 3:
                            raise RuntimeError(
                                f"Infinite-ISP 输出不是 uint8 RGB，得到 "
                                f"{out_rgb.dtype}/{out_rgb.shape}")
                        _sink.append(out_rgb)

                    self._util.save_pipeline_output = _capture
                    try:
                        self._mod.InfiniteISP(tmpdir, cfg_path).execute()
                    finally:
                        self._util.save_pipeline_output = saved_save
                    if not captured:
                        raise RuntimeError("Infinite-ISP 未产生输出（检查其参数配置）")
                    out = captured[0].astype(np.float32) / 255.0
                    outs.append(torch.from_numpy(
                        out.transpose(2, 0, 1)).to(image.device, image.dtype))
        finally:
            self._util.save_pipeline_output = saved_save
        return torch.stack(outs, dim=0).clamp(0.0, 1.0)


__all__ = ["InfiniteISPFront"]
