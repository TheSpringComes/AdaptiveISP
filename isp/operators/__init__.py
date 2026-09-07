"""isp.operators: concrete ISP operator implementations.

Import-time side effects here register every built-in operator into the
`isp.registry.OPERATORS` dict. The `isp.base` module owns the abstract
base + ParameterSpec + math utilities; `isp.registry` owns the OPERATORS
dict + @register decorator + CANONICAL_ORDER.
"""
from isp.base import (
    ISPOperator,
    ParameterSpec,
    hsv2rgb,
    lerp,
    rgb2hsv,
    rgb2lum,
    tanh01,
    tanh_range,
)
from isp.registry import (
    CANONICAL_ORDER,
    OPERATORS,
    build_operator,
    register,
)

# side-effect imports: each module @register-s its class into OPERATORS
from isp.operators import exposure       # noqa: F401
from isp.operators import gamma          # noqa: F401
from isp.operators import ccm            # noqa: F401
from isp.operators import sharpen        # noqa: F401
from isp.operators import denoise        # noqa: F401
from isp.operators import tone           # noqa: F401
from isp.operators import contrast       # noqa: F401
from isp.operators import saturation     # noqa: F401
from isp.operators import wnb            # noqa: F401
from isp.operators import whitebalance   # noqa: F401

# Neural (learned) operators — Samsung Modular Neural ISP wrappers.
from isp import learned as _learned      # noqa: F401

# Infinite-ISP-derived classical operators (Torch-native reimplementations).
from isp.operators import infinite_isp   # noqa: F401


__all__ = [
    "ISPOperator", "OPERATORS", "ParameterSpec", "build_operator", "register",
    "CANONICAL_ORDER",
    "tanh01", "tanh_range", "rgb2lum", "rgb2hsv", "hsv2rgb", "lerp",
]
