"""Samsung Modular Neural ISP Front ISP wrapper（第三方接入层，真实实现）。

对应：
    front_isp:
      enabled: true
      type: external
      external:
        backend: samsung_isp
        repo_path: front_isp/third_party/modular_neural_isp
        denoising_model_path:      <repo>/denoising/models/generic_base.pth
        photofinishing_model_path: <repo>/photofinishing/models/photofinishing_s24-style-0.pth
        # enhancement_model_path: <repo>/enhancement/models/enhancement_s24-style-0.pth  # 可选

设计约束（见 V3.1 规划）：
  - 第三方项目本身不修改，只通过本 wrapper 接入。
  - import 始终安全：第三方依赖延迟到 `_load_backend()`。

接入契约：Samsung 官方提供 in-memory Python API（`main/pipeline.py` 的
`PipeLine`，nn.Module）：

    net(raw=(H,W,3) float [0,1], illum=(3,), ccm=(3,3)) → {'srgb': (H,W,3)}

即输入是**demosaic 后的 linear camera RGB**（与本项目 Input Adapter 的
输出衔接自然），`raw_to_lsrgb` 内部做 `img @ (ccm @ diag(illum[1]/illum)).T`。
本 wrapper 逐样本：
  1. torch (3,H,W) linear [0,1] → numpy (H,W,3)；
  2. illum = 灰世界估计（逐通道均值，语义 = 场景光源色），
     ccm = 配置指定（默认 CycleISP 校准矩阵，或 3x3 字面量）；
  3. `PipeLine.forward` → `out['srgb']` → torch (1,3,H,W) [0,1]。

第三方 import 使用 sys.modules 隔离（Samsung 的顶层 `utils/` 包会与
yolov3 的同名包冲突，做法与 isp/learned/samsung_modular/backend.py
的 `_samsung_import` 一致）。
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp

_REPO_URL = "https://github.com/SamsungLabs/modular_neural_isp"


@contextlib.contextmanager
def _samsung_import(repo_root: str) -> Iterator[None]:
    """隔离 Samsung 的顶层 `utils*` 包与其它树（yolov3）的同名包。

    快照并移除 `sys.modules` 里的 `utils*` 条目，把 repo_root 前插到
    `sys.path`，yield，最后清掉 Samsung 装入的 utils 条目并恢复快照。
    """
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    saved: dict = {}
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            saved[key] = sys.modules.pop(key)
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "utils" or key.startswith("utils."):
                del sys.modules[key]
        sys.modules.update(saved)


@register_front_isp('modular_neural_isp')
class ModularNeuralISPFront(FrontISPBase):
    """Samsung Modular Neural ISP 的 Front ISP wrapper（真实实现）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.repo_path = os.path.expanduser(
            str(self.config.get('repo_path',
                                'front_isp/third_party/modular_neural_isp')))
        # 模型路径（缺省用仓库自带的预训练权重）
        self.denoising_model_path = self._resolve(
            self.config.get('denoising_model_path',
                            'denoising/models/generic_base.pth'))
        self.generic_denoising_model_path = self._resolve(
            self.config.get('generic_denoising_model_path',
                            'denoising/models/generic_base.pth'))
        self.photofinishing_model_path = self._resolve(
            self.config.get('photofinishing_model_path',
                            'photofinishing/models/photofinishing_s24-style-0.pth'))
        self.enhancement_model_path = self.config.get('enhancement_model_path')
        if self.enhancement_model_path:
            self.enhancement_model_path = self._resolve(self.enhancement_model_path)

        # 色彩参数：illum 语义 = 场景光源色（内部按 illum[1]/illum 归一）。
        #   illum: 'grayworld'（默认，逐样本估计）| 3 元列表
        #   ccm:   'cam2rgb'（默认，CycleISP 校准矩阵）| 3x3 列表
        self.illum_mode = self.config.get('illum', 'grayworld')
        self.ccm_spec = self.config.get('ccm', 'cam2rgb')
        device = str(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.device = torch.device(device)
        self._net = None      # PipeLine（首次 process 时构建，加载权重较重）
        self._pipeline_mod = None
        self._load_backend()

    # ---------------- helpers ----------------

    def _resolve(self, path: str) -> str:
        """相对路径按仓库根解析。"""
        p = str(path)
        if not os.path.isabs(p) and not os.path.exists(p):
            cand = os.path.join(self.repo_path, p)
            if os.path.exists(cand):
                return cand
        return p

    def _load_backend(self):
        """把第三方仓库挂到 sys.path 并验证可导入。失败时给出可操作的报错。"""
        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(
                f"modular_neural_isp 仓库未找到: {self.repo_path}\n"
                f"  git clone {_REPO_URL} {self.repo_path}\n"
                f"  并在 front_isp.external.repo_path 中指向它。")
        for p in (self.denoising_model_path, self.generic_denoising_model_path,
                  self.photofinishing_model_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"modular_neural_isp 权重未找到: {p}\n"
                    f"  从 { _REPO_URL } releases 下载预训练权重，或在 config 中指定路径。")

        pipeline_py = os.path.join(self.repo_path, 'main', 'pipeline.py')
        if not os.path.isfile(pipeline_py):
            raise FileNotFoundError(f"未找到 {pipeline_py}（检查仓库版本）")
        with _samsung_import(self.repo_path):
            spec = importlib.util.spec_from_file_location(
                "samsung_modular_pipeline", pipeline_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._pipeline_mod = mod

    def _build_net(self):
        """构建 PipeLine（重权重加载，首次 process 时调用一次）。"""
        with _samsung_import(self.repo_path):
            net = self._pipeline_mod.PipeLine(
                running_device=self.device,
                denoising_model_path=self.denoising_model_path,
                generic_denoising_model_path=self.generic_denoising_model_path,
                photofinishing_model_path=self.photofinishing_model_path,
                enhancement_model_path=self.enhancement_model_path,
            )
        net.eval()
        self._net = net

    def _get_illum(self, rgb: np.ndarray) -> np.ndarray:
        """场景光源色估计。'grayworld' = 逐通道均值（灰世界假设）。"""
        if isinstance(self.illum_mode, str) and self.illum_mode == 'grayworld':
            illum = rgb.reshape(-1, 3).mean(axis=0)
        else:
            illum = np.asarray(self.illum_mode, dtype=np.float32)
        illum = np.asarray(illum, dtype=np.float32)
        if illum.shape != (3,):
            raise ValueError(f"illum 必须是 (3,)，得到 {illum.shape}")
        return np.clip(illum, 1e-4, None)

    def _get_ccm(self) -> np.ndarray:
        """camera→sRGB 校准矩阵。默认 CycleISP 矩阵（与 canonical 链同源）。"""
        if isinstance(self.ccm_spec, str) and self.ccm_spec == 'cam2rgb':
            from isp.unprocess_np import get_calibrated_cam2rgb
            spec = get_calibrated_cam2rgb()
        else:
            spec = self.ccm_spec
        ccm = np.asarray(spec, dtype=np.float32)
        if ccm.shape != (3, 3):
            raise ValueError(f"ccm 必须是 3x3，得到 {ccm.shape}")
        return ccm

    # ---------------- FrontISPBase API ----------------

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        if self._net is None:
            self._build_net()
        ccm = self._get_ccm()
        outs = []
        with torch.no_grad():
            for i in range(image.shape[0]):
                rgb = image[i].detach().cpu().numpy().transpose(1, 2, 0)
                rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
                out = self._net(
                    raw=rgb,
                    illum=self._get_illum(rgb),
                    ccm=ccm,
                    img_metadata={},   # illum/ccm 均已提供；空 dict 过其内部断言
                    downscale_ps=bool(self.config.get('downscale_ps', True)),
                    log_messages=False,
                )
                srgb = out['srgb']
                if isinstance(srgb, torch.Tensor):
                    srgb = srgb.detach().cpu().numpy()
                srgb = np.clip(srgb.astype(np.float32), 0.0, 1.0)
                outs.append(torch.from_numpy(
                    srgb.transpose(2, 0, 1)).to(image.device, image.dtype))
        return torch.stack(outs, dim=0)


__all__ = ["ModularNeuralISPFront"]
