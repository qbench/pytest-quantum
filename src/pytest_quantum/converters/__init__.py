"""Circuit conversion utilities for pytest-quantum.

The main entry point is :func:`to_unitary`, which converts any supported
quantum circuit type into a numpy unitary matrix.
"""

from __future__ import annotations

from pytest_quantum.converters.to_unitary import to_unitary

__all__ = ["to_unitary"]
