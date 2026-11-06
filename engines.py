"""Polars engines for tests: parametrize over ENGINES, then collect with `engine(name)`.

Test-only (it imports pytest, a dev dependency).
"""

import polars as pl
import pytest


def gpu_available() -> bool:
    try:
        pl.LazyFrame({"a": [1]}).collect(engine=pl.GPUEngine(raise_on_fail=True))
        return True
    except Exception:  # noqa: BLE001  # any failure means no usable GPU engine
        return False


GPU = pytest.param(
    "gpu",
    marks=pytest.mark.skipif(not gpu_available(), reason="GPU engine unavailable"),
)
ENGINES = ["streaming", "in-memory", GPU]


def engine(name: str):
    """The engine for `collect`; for gpu, fail rather than silently fall back to CPU."""
    return pl.GPUEngine(raise_on_fail=True) if name == "gpu" else name
