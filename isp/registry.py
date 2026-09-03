"""ISP Operator Registry: name -> class map + @register decorator.

Kept intentionally minimal (~20 lines of substance) — zero external deps,
no config parsing. Operator files register themselves at import time via
side-effect imports in `isp/operators/__init__.py`.
"""
from __future__ import annotations

from typing import Callable

from isp.base import ISPOperator


OPERATORS: dict[str, type[ISPOperator]] = {}


def register(name: str) -> Callable[[type[ISPOperator]], type[ISPOperator]]:
    """Decorator: register an ISPOperator subclass under `name`."""
    def deco(cls: type[ISPOperator]) -> type[ISPOperator]:
        if name in OPERATORS:
            raise ValueError(
                f"ISP operator name collision: {name!r} already registered "
                f"as {OPERATORS[name].__name__}"
            )
        cls.name = name
        OPERATORS[name] = cls
        return cls
    return deco


def build_operator(name: str, **kwargs) -> ISPOperator:
    if name not in OPERATORS:
        raise KeyError(f"Unknown ISP operator: {name!r}. Registered: {sorted(OPERATORS)}")
    return OPERATORS[name](**kwargs)


# Canonical order pinned for observation-channel layout + PipelineExecutor dispatch
# stability. Matches config.py cfg.operators.
CANONICAL_ORDER: list[str] = [
    "exposure", "gamma", "ccm", "sharpen", "denoise",
    "tone", "contrast", "saturation", "wnb", "whitebalance",
]


__all__ = ["OPERATORS", "register", "build_operator", "CANONICAL_ORDER"]
