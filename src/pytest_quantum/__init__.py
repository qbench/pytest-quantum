"""pytest-quantum: A cross-framework pytest plugin for quantum program testing.

Public API — import anything you need directly from ``pytest_quantum``::

    from pytest_quantum import (
        # Unitary
        assert_unitary,
        assert_circuits_equivalent,
        # States
        assert_state_fidelity_above,
        assert_states_close,
        # Density matrices (NEW in v0.2.0)
        assert_density_matrix_close,
        assert_trace_distance_below,
        assert_purity_above,
        assert_partial_trace_close,
        # Observables / expectation values (NEW in v0.2.0)
        assert_expectation_value_close,
        assert_ground_state_energy_close,
        # Distributions
        assert_measurement_distribution,
        assert_counts_close,
        # Primitives (NEW in v0.2.0)
        assert_sampler_distribution,
        assert_estimator_close,
        # Structure
        assert_circuit_depth,
        assert_circuit_width,
        assert_gate_count,
        assert_circuit_is_clifford,  # NEW in v0.2.0
        # Snapshots (NEW in v0.2.0)
        assert_unitary_snapshot,
        assert_distribution_snapshot,
        # Stats
        min_shots,
        recommended_shots,
        fidelity,
        tvd,
        tvd_from_counts,
        chi_square_test,
    )

Fixtures (``aer_simulator``, ``cirq_simulator``, ``cirq_sampler``,
``qiskit_sampler``, ``qiskit_estimator``, etc.) are injected automatically
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
pytest.register_assert_rewrite("pytest_quantum.assertions.density")
pytest.register_assert_rewrite("pytest_quantum.assertions.observables")

from pytest_quantum.assertions.density import (
    assert_density_matrix_close,
    assert_partial_trace_close,
    assert_purity_above,
    assert_trace_distance_below,
)
from pytest_quantum.assertions.distributions import (
    assert_counts_close,
    assert_measurement_distribution,
)
from pytest_quantum.assertions.observables import (
    assert_expectation_value_close,
    assert_ground_state_energy_close,
)
from pytest_quantum.assertions.primitives import (
    assert_estimator_close,
    assert_sampler_distribution,
)
from pytest_quantum.assertions.snapshot import (
    assert_distribution_snapshot,
    assert_unitary_snapshot,
)
from pytest_quantum.assertions.states import (
    assert_state_fidelity_above,
    assert_states_close,
)
from pytest_quantum.assertions.structure import (
    assert_circuit_depth,
    assert_circuit_is_clifford,
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

__version__ = "0.2.0"

__all__ = [
    "assert_circuit_depth",
    "assert_circuit_is_clifford",
    "assert_circuit_width",
    "assert_circuits_equivalent",
    "assert_counts_close",
    "assert_density_matrix_close",
    "assert_distribution_snapshot",
    "assert_estimator_close",
    "assert_expectation_value_close",
    "assert_gate_count",
    "assert_ground_state_energy_close",
    "assert_measurement_distribution",
    "assert_partial_trace_close",
    "assert_purity_above",
    "assert_sampler_distribution",
    "assert_state_fidelity_above",
    "assert_states_close",
    "assert_trace_distance_below",
    "assert_unitary",
    "assert_unitary_snapshot",
    "chi_square_test",
    "fidelity",
    "min_shots",
    "recommended_shots",
    "tvd",
    "tvd_from_counts",
]
