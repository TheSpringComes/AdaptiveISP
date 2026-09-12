"""Fixed Front ISP — FittedISP 拟合方法（V3.1 模式之二）。

实现方式来自 `front_isp/FittedISP`（自动拟合 ISP）：对全部图像共用一套
全局参数——色调指数（exponent）→ 3×3 CCM ＋ RGB 偏置 → 亮度细节控制。
参数由 `tools/fit_front_isp.py` 用 IRLS 在训练集上离线拟合一次
（方法与 FittedISP/fit.py 一致），运行期完全固定、无任何可训练参数。

对应配置：

    front_isp:
      enabled: true
      type: fixed
      fixed:
        params: configs/front_isp/fitted_fivek.json   # FittedISP 格式参数

输入输出约定：
  - 输入：Dataset 已去马赛克的 linear RGB (B,3,H,W) [0,1]；
    FittedISP 原始流程中的 Bayer 去马赛克与暗角补偿发生在 RAW 域，
    对本输入不适用（去马赛克已由 Dataset 完成），此处只实现其
    去马赛克之后的 color/detail 两级；
  - 输出：clamp 到 [0,1]（Front ISP 的统一输出契约）。

参数文件格式（FittedISP params.json 兼容子集）：
  {"exponent": 1.0, "ccm": [[...3行...]], "offset": [r,g,b],
   "detail_strength": -0.5}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp

_DEFAULT_PARAMS = "configs/front_isp/fitted_fivek.json"

# FittedISP detail() 的 3×3 Gaussian 核（作用于亮度通道）。
_DETAIL_KERNEL = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 16.0
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


@register_front_isp('fixed')
class FixedFrontISP(FrontISPBase):
    """FittedISP 方法的固定前端：exponent → CCM＋偏置 → 细节控制。

    config 项：
        params: FittedISP 格式参数——JSON 文件路径或内联 dict。
                缺省 `configs/front_isp/fitted_fivek.json`。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        params = self.config.get('params', _DEFAULT_PARAMS)
        if isinstance(params, str):
            path = os.path.expanduser(params)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"fixed Front ISP 参数文件未找到: {path}\n"
                    f"  先运行 python tools/fit_front_isp.py 生成，或在\n"
                    f"  front_isp.fixed.params 中指定其他 FittedISP 格式参数。")
            with open(path, "r", encoding="utf-8") as fh:
                params = json.load(fh)
        if not isinstance(params, dict):
            raise ValueError("front_isp.fixed.params 应为 JSON 路径或 dict")

        try:
            exponent = float(params['exponent'])
            ccm = torch.as_tensor(params['ccm'], dtype=torch.float32)
            offset = torch.as_tensor(params['offset'], dtype=torch.float32)
            strength = float(params['detail_strength'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"fixed Front ISP 参数缺少必需字段 (exponent/ccm/offset/"
                f"detail_strength): {exc!r}") from exc
        if ccm.shape != (3, 3) or offset.shape != (3,):
            raise ValueError(f"ccm 应为 3×3、offset 应为 3，得到 {ccm.shape}/{offset.shape}")

        # 不可训练常量（buffers：parameters() 为空，RL 不会更新它们）。
        self.register_buffer('ccm', ccm)
        self.register_buffer('offset', offset)
        self.exponent = exponent
        self.detail_strength = strength
        self.params_source = params.get('fit_source', 'inline')

    # ---------------- FrontISPBase API ----------------

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        # color(): max(rgb,0)^exponent @ ccm.T + offset
        x = image.clamp_min(0) ** self.exponent
        rgb = torch.einsum('bchw,dc->bdhw', x, self.ccm) + self.offset.view(1, 3, 1, 1)
        # detail(): 亮度通道 3×3 Gaussian 细节控制（正值锐化，负值平滑）
        if self.detail_strength != 0.0:
            luma = (rgb * torch.tensor(_LUMA_WEIGHTS, device=rgb.device,
                                       dtype=rgb.dtype).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
            blur = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode='replicate'),
                            _DETAIL_KERNEL.to(luma.device, luma.dtype).expand(1, 1, 3, 3))
            rgb = rgb + self.detail_strength * (luma - blur)
        return rgb.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return (f"FixedFrontISP(exponent={self.exponent:.4f}, "
                f"detail={self.detail_strength}, src={self.params_source})")


__all__ = ["FixedFrontISP"]
