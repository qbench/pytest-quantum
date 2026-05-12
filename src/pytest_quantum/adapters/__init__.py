"""Adapter abstraction layer for quantum framework support.

Re-exports the public API so users can write::

    from pytest_quantum.adapters import get_adapter, register_adapter
"""

from __future__ import annotations

from pytest_quantum.adapters.protocol import FrameworkAdapter
from pytest_quantum.adapters.registry import (
    AdapterRegistry,
    detect_framework,
    get_adapter,
    register_adapter,
)

__all__ = [
    "AdapterRegistry",
    "FrameworkAdapter",
    "detect_framework",
    "get_adapter",
    "register_adapter",
]
