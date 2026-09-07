"""Infinite-ISP-derived operators — Torch-native reimplementations.

Reference (source of the algorithm specs, not the runtime code):
    https://github.com/10x-Engineers/Infinite-ISP

Side-effect imports register each operator into `isp.registry.OPERATORS`.
Names are prefixed `inf_` so they're stable columns in `op_usage` and
distinguishable from classical (no prefix) and neural (`n_`) siblings.
"""
from isp.operators.infinite_isp import awb            # noqa: F401
from isp.operators.infinite_isp import gain           # noqa: F401
from isp.operators.infinite_isp import contrast       # noqa: F401
from isp.operators.infinite_isp import sharpen        # noqa: F401
from isp.operators.infinite_isp import denoise        # noqa: F401
from isp.operators.infinite_isp import saturation     # noqa: F401
