"""pytest-quantum: A cross-framework pytest plugin for quantum program testing.

Public API — import anything you need directly from ``pytest_quantum``::

    from pytest_quantum import (
        assert_unitary,
        assert_circuits_equivalent,
        assert_state_fidelity_above,
        assert_states_close,
        assert_measurement_distribution,
        assert_counts_close,
        assert_circuit_depth,
        assert_circuit_width,
        assert_gate_count,
        min_shots,
        recommended_shots,
        fidelity,
        tvd,
        tvd_from_counts,
        chi_square_test,
    )

Fixtures (``aer_simulator``, ``cirq_simulator``, etc.) are injected automatically
by pytest — no import needed, just declare them as test parameters.
"""

from __future__ import annotations

# Register assertion rewriting BEFORE the modules are first imported.
# This lets pytest show detailed expression diffs inside our helper functions.
import pytest

pytest.register_assert_rewrite("pytest_quantum.assertions.unitary")
pytest.register_assert_rewrite("pytest_quantum.assertions.states")
pytest.register_assert_rewrite("pytest_quantum.assertions.distributions")
pytest.register_assert_rewrite("pytest_quantum.assertions.structure")

from pytest_quantum.assertions.distributions import (
    assert_counts_close,
    assert_measurement_distribution,
)
from pytest_quantum.assertions.states import (
    assert_state_fidelity_above,
    assert_states_close,
)
from pytest_quantum.assertions.structure import (
    assert_circuit_depth,
    assert_circuit_width,
    assert_gate_count,
)
from pytest_quantum.assertions.unitary import (
    assert_circuits_equivalent,
    assert_unitary,
)
from pytest_quantum.stats.shots import min_shots, recommended_shots
from pytest_quantum.stats.tests import (
    chi_square_test,
    fidelity,
    tvd,
    tvd_from_counts,
)

__version__ = "0.1.0"

__all__ = [
    "assert_circuit_depth",
    "assert_circuit_width",
    "assert_circuits_equivalent",
    "assert_counts_close",
    "assert_gate_count",
    "assert_measurement_distribution",
    "assert_state_fidelity_above",
    "assert_states_close",
    "assert_unitary",
    "chi_square_test",
    "fidelity",
    "min_shots",
    "recommended_shots",
    "tvd",
    "tvd_from_counts",
]
