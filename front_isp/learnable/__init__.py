"""front_isp.learnable — V3.1 可学习相机参数（模式之三）。

    linear RGB → Learnable ISP（WB + CCM/Bias + Base Tone）→ Base RGB

组件：
    color_mapping    — 可微变换（apply_wb / apply_ccm_bias / apply_tone）
    camera_params    — CameraParamTable（camera-specific 参数表）
    fittedisp_loader — FittedISP params.json → 标定参数初始化
    module           — LearnableFrontISP（注册为 'learnable'，旧名 'calibrated'）
"""
from front_isp.learnable.camera_params import CameraParamTable
from front_isp.learnable.color_mapping import (
    CameraParams,
    apply_learnable,
    apply_ccm_bias,
    apply_tone,
    apply_wb,
)
from front_isp.learnable.module import LearnableFrontISP

# legacy 别名（V3.1 前的旧类名，保持旧 import 不破）
CalibratedFrontISP = LearnableFrontISP
apply_calibration = apply_learnable

__all__ = [
    "LearnableFrontISP",
    "CalibratedFrontISP",
    "CameraParamTable",
    "CameraParams",
    "apply_learnable",
    "apply_calibration",
    "apply_ccm_bias",
    "apply_tone",
    "apply_wb",
]
