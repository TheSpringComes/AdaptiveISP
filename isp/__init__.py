"""isp: image signal processing operators + registry.

Answers "what ISP operations are available?"

Importing this package populates `isp.registry.OPERATORS` with all built-in
operators. Consumers can then use `isp.registry.build_operator(name)` to
instantiate any of them.
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

# side-effect import: populate OPERATORS via @register on the concrete files
from isp import operators as _operators   # noqa: F401

__all__ = [
    "ISPOperator", "OPERATORS", "ParameterSpec", "build_operator", "register",
    "CANONICAL_ORDER",
    "tanh01", "tanh_range", "rgb2lum", "rgb2hsv", "hsv2rgb", "lerp",
]
