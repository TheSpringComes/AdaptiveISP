"""isp.learned: neural (learned) ISP operator wrappers.

Importing this package registers every wrapped neural operator into
`isp.registry.OPERATORS`. Each wrapper exposes the same `ISPOperator`
contract as classical operators: a `spec.dim=1` strength `alpha ∈ [0,1]`,
and `apply(img, params)` returning `img + alpha * (F(img) - img)`.

Backend weights are frozen and shared across wrappers.
"""
from isp.learned import samsung_modular as _samsung  # noqa: F401

__all__: list[str] = []
