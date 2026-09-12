"""Backward-compat re-export: CanonicalBackbone 已迁移到 `front_isp.canonical`。

Front ISP 现在是独立于 pipeline 的可插拔模块（见 `front_isp/`），请改为：

    from front_isp import build_front_isp_from_cfg, CanonicalBackbone

本文件仅为旧的 `pipeline.CanonicalBackbone` 导入保留，避免外部实验
脚本失效。
"""
from front_isp.canonical import CanonicalBackbone

__all__ = ["CanonicalBackbone"]
