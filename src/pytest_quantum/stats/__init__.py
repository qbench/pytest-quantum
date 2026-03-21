"""Statistical utilities for quantum test assertions.

These helpers are framework-independent pure Python/numpy/scipy functions.
They can be used directly in tests or are called internally by the assertion
helpers.
"""

from __future__ import annotations

from pytest_quantum.stats.shots import min_shots, recommended_shots
from pytest_quantum.stats.tests import chi_square_test, fidelity, tvd, tvd_from_counts

__all__ = [
    "chi_square_test",
    "fidelity",
    "min_shots",
    "recommended_shots",
    "tvd",
    "tvd_from_counts",
]
