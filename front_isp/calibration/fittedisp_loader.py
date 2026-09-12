"""FittedISP 参数加载器（V3.1 §2 — FittedISP Parameters 初始化）。

对应 config：

    front_isp:
      type: calibrated
      calibration:
        init:
          type: fittedisp
          params: params.json

FittedISP 拟合输出的是 "RAW–Camera RGB" 对上的线性映射（CCM + Bias）
与 tone 近似，格式随导出脚本而异。本 loader 接受宽松的键名并输出
`CameraParamTable` 的参数化：

    返回 {log_wb: (3,), ccm: (3,3), bias: (3,), log_gamma: scalar-tensor}

识别的键名（任选其一，大小写不敏感）：
    wb        : "wb", "white_balance", "wb_gain", "gain"      → (3,)
    ccm       : "ccm", "color_matrix", "matrix", "cam2rgb"    → (3,3)
    bias      : "bias", "offset", "b"                          → (3,)
    gamma/tone: "gamma", "tone", "exponent"                    → scalar

支持的外层包装：dict 本体、或 {"params": {...}} / {"calibration": {...}}
嵌套；数值可给线性域（wb>0、gamma>0，取 log）。
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

import torch

_ALIASES = {
    "wb": ("wb", "white_balance", "wb_gain", "gain", "rgb_gain"),
    "ccm": ("ccm", "color_matrix", "matrix", "cam2rgb", "color_correction_matrix"),
    "bias": ("bias", "offset", "b", "rgb_bias"),
    "gamma": ("gamma", "tone", "exponent", "tone_gamma"),
}


def _find(d: Dict[str, Any], keys: tuple) -> Optional[Any]:
    for k, v in d.items():
        if str(k).lower() in keys:
            return v
    return None


def _as_tensor(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32).reshape(-1)


def load_fittedisp_params(cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """从 FittedISP 导出文件 / 内联 dict 读取，返回标定参数（log 域）。

    cfg 支持：
        params: JSON 文件路径 或 内联 dict 或 {"wb":..,"ccm":..}
        defaults 在键缺失时使用 identity。
    """
    src = cfg.get("params", None)
    data: Dict[str, Any] = {}
    if src:
        if isinstance(src, str):
            if not os.path.isfile(src):
                raise FileNotFoundError(
                    f"FittedISP 参数文件未找到: {src}\n"
                    "  请先将拟合结果导出为 JSON 并在 init.params 中指向它。"
                )
            with open(src) as fh:
                data = json.load(fh)
        elif isinstance(src, dict):
            data = src
        else:
            raise TypeError(f"init.params 应为路径或 dict，得到 {type(src)}")
        # 常见的外层包装。
        for wrap in ("params", "calibration", "fittedisp"):
            if isinstance(data.get(wrap), dict):
                data = {**data, **data[wrap]}

    out: Dict[str, torch.Tensor] = {
        "log_wb": torch.zeros(3),
        "ccm": torch.eye(3),
        "bias": torch.zeros(3),
        "log_gamma": torch.zeros(()),
    }

    wb = _find(data, _ALIASES["wb"])
    if wb is not None:
        t = _as_tensor(wb)
        if t.numel() == 3 and (t > 0).all():
            out["log_wb"] = t.clamp(min=1e-6).log()
        else:
            raise ValueError(f"FittedISP wb 应为 3 个正数，得到 {wb!r}")

    ccm = _find(data, _ALIASES["ccm"])
    if ccm is not None:
        t = torch.as_tensor(ccm, dtype=torch.float32)
        if t.shape == (3, 3):
            out["ccm"] = t
        else:
            raise ValueError(f"FittedISP ccm 应为 3×3，得到 shape={tuple(t.shape)}")

    bias = _find(data, _ALIASES["bias"])
    if bias is not None:
        t = _as_tensor(bias)
        if t.numel() == 3:
            out["bias"] = t
        else:
            raise ValueError(f"FittedISP bias 应为 3 个数，得到 {bias!r}")

    gamma = _find(data, _ALIASES["gamma"])
    if gamma is not None:
        g = float(_as_tensor(gamma).reshape(-1)[0])
        if g <= 0:
            raise ValueError(f"FittedISP gamma 应为正数，得到 {g}")
        out["log_gamma"] = torch.tensor(math.log(g))

    return out


__all__ = ["load_fittedisp_params"]
