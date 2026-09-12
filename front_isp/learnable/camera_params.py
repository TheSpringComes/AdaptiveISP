"""Camera Parameter Table — camera-specific 标定参数存储（V3.1 §3）。

    Camera ID → Camera Parameter Table
        ├─ Camera A → Calibration A
        ├─ Camera B → Calibration B
        └─ Camera C → Calibration C

每台相机一行参数：{WB, CCM, Bias, Gamma}（V3.1 第一版不引入 neural
calibration predictor）。`camera_specific=False` 时退化为单行共享参数。

初始化方式（V3.1 §2）：
    identity       — WB=1, CCM=I, Bias=0, Gamma=1（中性标定，不改图像）
    fittedisp      — 从 FittedISP 导出的 params.json 初始化
    camera_specific — 每台相机各自初始化（第一版 = identity per camera，
                      为后续 per-camera 表预留入口）

参数化见 color_mapping.py：wb_log / ccm / bias / log_gamma 均为 nn.Parameter，
支持梯度训练，可按 config 的 learnable 开关冻结。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from front_isp.learnable.color_mapping import CameraParams

_PARAM_KEYS = ("wb", "ccm", "bias", "gamma")


class CameraParamTable(nn.Module):
    """`(n_cameras, ·)` 形状的标定参数表，按 camera id 索引取行。"""

    def __init__(
        self,
        n_cameras: int = 1,
        init: str = "identity",
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if n_cameras < 1:
            raise ValueError(f"n_cameras must be ≥ 1, got {n_cameras}")
        self.n_cameras = int(n_cameras)

        init = str(init)
        if init not in ("identity", "fittedisp", "camera_specific"):
            raise ValueError(
                f"未知 calibration init: '{init}'。"
                "可用: identity | fittedisp | camera_specific"
            )
        # camera_specific 第一版 = identity per camera（预留 per-camera 表）。
        if init in ("identity", "camera_specific"):
            wb = torch.zeros(n_cameras, 3)                    # gain = 1
            ccm = torch.eye(3).unsqueeze(0).repeat(n_cameras, 1, 1)
            bias = torch.zeros(n_cameras, 3)
            log_gamma = torch.zeros(n_cameras, 1)             # gamma = 1
        else:  # fittedisp
            from front_isp.learnable.fittedisp_loader import load_fittedisp_params
            p = load_fittedisp_params(init_params or {})
            n = n_cameras
            wb = p["log_wb"].view(1, 3).repeat(n, 1)
            ccm = p["ccm"].view(1, 3, 3).repeat(n, 1, 1)
            bias = p["bias"].view(1, 3).repeat(n, 1)
            log_gamma = p["log_gamma"].view(1, 1).repeat(n, 1)

        self.wb_log = nn.Parameter(wb)
        self.ccm = nn.Parameter(ccm)
        self.bias = nn.Parameter(bias)
        self.log_gamma = nn.Parameter(log_gamma)

    # ---------------- lookup ----------------

    def forward(self, camera_id: Optional[torch.Tensor] = None) -> CameraParams:
        """按 batch 的 camera id 取参数行。`camera_id` 缺席 → 第 0 行。
        越界 id（eval 时新相机）截断到 [0, n_cameras-1]。"""
        if camera_id is None:
            idx = torch.zeros(1, dtype=torch.long, device=self.wb_log.device)
        else:
            idx = camera_id.to(self.wb_log.device).long().flatten()
        if idx.numel() == 0:
            idx = torch.zeros(1, dtype=torch.long, device=self.wb_log.device)
        idx = idx.clamp(0, self.n_cameras - 1)
        return CameraParams(
            wb_log=self.wb_log[idx],
            ccm=self.ccm[idx],
            bias=self.bias[idx],
            log_gamma=self.log_gamma[idx],
        )

    # ---------------- learnability ----------------

    def set_learnable(self, wb: bool, ccm: bool, bias: bool, tone: bool) -> None:
        """按 config 的 learnable 开关冻结/解冻各参数组。"""
        self.wb_log.requires_grad_(bool(wb))
        self.ccm.requires_grad_(bool(ccm))
        self.bias.requires_grad_(bool(bias))
        self.log_gamma.requires_grad_(bool(tone))

    def learnable_parameter_groups(self) -> list[nn.Parameter]:
        """当前未被冻结的全部参数（供 optimizer 使用）。"""
        return [p for p in self.parameters() if p.requires_grad]

    # ---------------- interop ----------------

    def export_dict(self) -> Dict[str, Any]:
        """导出为可 JSON 化的参数 dict（与 fittedisp_loader 的读入格式同构）。"""
        return {
            "wb": torch.exp(self.wb_log).detach().cpu().tolist(),
            "ccm": self.ccm.detach().cpu().tolist(),
            "bias": self.bias.detach().cpu().tolist(),
            "gamma": torch.exp(self.log_gamma).detach().cpu().tolist(),
        }

    def __repr__(self) -> str:
        n_learn = sum(p.requires_grad for p in self.parameters())
        return (f"CameraParamTable(n_cameras={self.n_cameras}, "
                f"learnable_groups={n_learn}/4)")


__all__ = ["CameraParamTable", "_PARAM_KEYS"]
