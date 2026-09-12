"""front_isp.calibration — V3.1 learnable camera calibration。

    RAW → Calibrated ISP（Demosaic 固定 + WB + CCM/Bias + Base Tone）→ Base RGB

组件：
    color_mapping    — 可微变换（apply_wb / apply_ccm_bias / apply_tone）
    camera_params    — CameraParamTable（camera-specific 参数表）
    fittedisp_loader — FittedISP params.json → 标定参数初始化
    module           — CalibratedFrontISP（注册为 front_isp type 'calibrated'）
"""
from front_isp.calibration.camera_params import CameraParamTable
from front_isp.calibration.color_mapping import (
    CameraParams,
    apply_calibration,
    apply_ccm_bias,
    apply_tone,
    apply_wb,
)
from front_isp.calibration.module import CalibratedFrontISP

__all__ = [
    "CalibratedFrontISP",
    "CameraParamTable",
    "CameraParams",
    "apply_calibration",
    "apply_ccm_bias",
    "apply_tone",
    "apply_wb",
    "load_fittedisp_params",
]
