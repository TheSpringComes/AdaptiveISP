"""Calibrated Front ISP — learnable camera calibration（V3.1 §1–§3）。

对应 config：

    front_isp:
      enabled: true
      type: calibrated
      calibration:
        camera_specific: true
        n_cameras: 24            # camera_specific 时需要；训练侧会自动注入
        demosaic:
          type: malvar_bilinear  # 暂固定（Input Adapter 层已完成），仅记录
          learnable: false
        white_balance: {learnable: true}
        ccm:          {learnable: true}
        bias:         {learnable: true}
        tone:         {learnable: true}
        init:
          type: fittedisp        # identity | fittedisp | camera_specific
          params: params.json

前向（V3.1 §1）：

    I_base = f_calib(RAW; θ_c),   θ_c = {WB, M, b, γ}

顺序：Demosaic(固定) → WB/Channel Gain → CCM+Bias → Base Tone。
全部参数支持梯度训练；camera_specific 时按 `metadata['camera_id']`
索引 Camera Parameter Table。与 AdaptiveISP 的耦合只经由
`process()` 的输入输出 — Controller/RL 侧不感知 calibration。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from front_isp.base import FrontISPBase
from front_isp.calibration.camera_params import CameraParamTable
from front_isp.calibration.color_mapping import apply_calibration
from front_isp.registry import register_front_isp


@register_front_isp('calibrated')
class CalibratedFrontISP(FrontISPBase):
    """可学习相机标定 Front ISP。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        # config 既可为 {'calibration': {...}}（YAML 主配置的子键），
        # 也可直接是 calibration 体（registry 以 'calibrated' 子键传入时）。
        calib = self.config.get('calibration', None)
        if calib is None:
            calib = dict(self.config)

        self.camera_specific = bool(calib.get('camera_specific', False))
        n_cameras = int(calib.get('n_cameras', 1))
        if not self.camera_specific:
            n_cameras = 1

        # Demosaic：V3.1 暂固定。demosaic 发生在 Input Adapter 层
        # （front_isp/raw_adapter.py：Bayer 重建 → 0.5*Malvar +
        # 0.5*Bilinear → canonical linear RGB），ISP 收到的已是 3 通道
        # RGB，因此这里只记录配置，不再做 mosaic 处理；learnable 必须
        # 为 false。
        dm = calib.get('demosaic', {}) or {}
        self.demosaic_type = str(dm.get('type', 'malvar_bilinear'))
        if bool(dm.get('learnable', False)):
            raise ValueError("V3.1: demosaic 暂为固定级，不支持 learnable: true")

        # 初始化（identity | fittedisp | camera_specific）。
        init = calib.get('init', {}) or {}
        self.table = CameraParamTable(
            n_cameras=n_cameras,
            init=str(init.get('type', 'identity')),
            init_params=init if 'params' in init else None,
        )

        # learnable 开关。
        def _lr(key: str) -> bool:
            return bool((calib.get(key, {}) or {}).get('learnable', True))
        self.table.set_learnable(wb=_lr('white_balance'), ccm=_lr('ccm'),
                                 bias=_lr('bias'), tone=_lr('tone'))

        # Stage 1 预训练结果加载（CalibrationTrainer 的 ckpt 含
        # 'front_isp' state_dict）。init.params (JSON) 与 init.ckpt (pth)
        # 二选一；ckpt 优先级更低，作为 V3.1-B 的标准入口。
        ckpt_path = calib.get('ckpt', None)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state = state.get('front_isp', state)
            try:
                self.load_state_dict(state)
            except RuntimeError:
                # 常见于：实验目录拷贝的 yaml 是训练侧注入 n_cameras 之前的
                # 版本，表尺寸与 ckpt 不一致。camera_specific 时按 ckpt 的
                # 实际表尺寸重建再加载（ckpt 是事实来源）。
                n_ckpt = state.get('table.wb_log')
                if (self.camera_specific and n_ckpt is not None
                        and n_ckpt.dim() == 2 and n_ckpt.shape[0] != n_cameras):
                    self.table = CameraParamTable(
                        n_cameras=int(n_ckpt.shape[0]),
                        init='identity',
                    )
                    self.table.set_learnable(wb=_lr('white_balance'), ccm=_lr('ccm'),
                                             bias=_lr('bias'), tone=_lr('tone'))
                    self.load_state_dict(state)
                else:
                    raise

    # ---------------- FrontISPBase API ----------------

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        """(B,3,H,W) RAW → Base RGB。

        metadata: {'camera_id': (B,) long}（camera_specific 时必传；
                  缺席/为 None 时所有样本用第 0 行参数）。
        """
        cam_id = None
        if isinstance(metadata, dict):
            cam_id = metadata.get('camera_id')
        params = self.table(cam_id)
        out = apply_calibration(image, params)
        # 推理语义下裁回 [0,1]；训练时梯度在端点外为 0，可接受。
        return out.clamp(0.0, 1.0)

    # ---------------- training stage helpers ----------------

    def freeze(self) -> None:
        """Stage 2（adaptive_train）：冻结全部标定参数。"""
        self.table.set_learnable(wb=False, ccm=False, bias=False, tone=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self) -> list:
        """Stage 1/3（calibration_pretrain / joint_finetune）：可训练参数。"""
        return [p for p in self.parameters() if p.requires_grad]

    def load_calibrated_state(self, state: Dict[str, Any]) -> None:
        """加载 ckpt 中保存的标定参数（按 state_dict 键名宽松匹配）。"""
        self.load_state_dict(state)

    def __repr__(self) -> str:
        return (f"CalibratedFrontISP(camera_specific={self.camera_specific}, "
                f"demosaic={self.demosaic_type}(fixed), table={self.table!r})")


__all__ = ["CalibratedFrontISP"]
