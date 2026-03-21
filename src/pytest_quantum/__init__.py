"""pytest-quantum: A cross-framework pytest plugin for quantum program testing.

Public API — import anything you need directly from ``pytest_quantum``::

    from pytest_quantum import (
        # Unitary
        assert_unitary,
        assert_circuits_equivalent,
        # States
        assert_state_fidelity_above,
        assert_states_close,
        # Density matrices (v0.2.0)
        assert_density_matrix_close,
        assert_trace_distance_below,
        assert_purity_above,
        assert_partial_trace_close,
        # Observables / expectation values (v0.2.0)
        assert_expectation_value_close,
        assert_ground_state_energy_close,
        # Distributions
        assert_measurement_distribution,
        assert_counts_close,
        # Primitives (v0.2.0)
        assert_sampler_distribution,
        assert_estimator_close,
        # Structure
        assert_circuit_depth,
        assert_circuit_width,
        assert_gate_count,
        assert_circuit_is_clifford,
        # Snapshots (v0.2.0)
        assert_unitary_snapshot,
        assert_distribution_snapshot,
        # Channels / operators (NEW in v0.3.0)
        assert_hermitian,
        assert_positive_semidefinite,
        assert_commutes_with,
        assert_channel_is_cptp,
        assert_process_fidelity_above,
        assert_noise_fidelity_above,
        # Entanglement (NEW in v0.3.0)
        assert_entanglement_entropy_below,
        assert_bloch_sphere_close,
        assert_schmidt_rank_at_most,
        # Information theory (NEW in v0.3.0)
        assert_hellinger_close,
        assert_kl_divergence_below,
        assert_cross_entropy_below,
        # QASM round-trip (NEW in v0.3.0)
        assert_qasm_roundtrip,
        # Stats
        min_shots,
        recommended_shots,
        fidelity,
        tvd,
        tvd_from_counts,
        chi_square_test,
    )

Fixtures (``aer_simulator``, ``cirq_simulator``, ``cirq_sampler``,
``qiskit_sampler``, ``qiskit_estimator``, ``pytket_circuit_factory``,
``stim_sampler``, ``quantum_benchmark``, ``shot_budget``, etc.) are injected
automatically by pytest — no import needed, just declare them as test
parameters.
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
pytest.register_assert_rewrite("pytest_quantum.assertions.channels")
pytest.register_assert_rewrite("pytest_quantum.assertions.entanglement")
pytest.register_assert_rewrite("pytest_quantum.assertions.information")
pytest.register_assert_rewrite("pytest_quantum.assertions.qasm")

from pytest_quantum.assertions.channels import (
    assert_channel_is_cptp,
    assert_commutes_with,
    assert_hermitian,
    assert_noise_fidelity_above,
    assert_positive_semidefinite,
    assert_process_fidelity_above,
)
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
from pytest_quantum.assertions.entanglement import (
    assert_bloch_sphere_close,
    assert_entanglement_entropy_below,
    assert_schmidt_rank_at_most,
)
from pytest_quantum.assertions.information import (
    assert_cross_entropy_below,
    assert_hellinger_close,
    assert_kl_divergence_below,
)
from pytest_quantum.assertions.observables import (
    assert_expectation_value_close,
    assert_ground_state_energy_close,
)
from pytest_quantum.assertions.primitives import (
    assert_estimator_close,
    assert_sampler_distribution,
)
from pytest_quantum.assertions.qasm import assert_qasm_roundtrip
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

__version__ = "0.3.0"

__all__ = [
    "assert_bloch_sphere_close",
    "assert_channel_is_cptp",
    "assert_circuit_depth",
    "assert_circuit_is_clifford",
    "assert_circuit_width",
    "assert_circuits_equivalent",
    "assert_commutes_with",
    "assert_counts_close",
    "assert_cross_entropy_below",
    "assert_density_matrix_close",
    "assert_distribution_snapshot",
    "assert_entanglement_entropy_below",
    "assert_estimator_close",
    "assert_expectation_value_close",
    "assert_gate_count",
    "assert_ground_state_energy_close",
    "assert_hellinger_close",
    "assert_hermitian",
    "assert_kl_divergence_below",
    "assert_measurement_distribution",
    "assert_noise_fidelity_above",
    "assert_partial_trace_close",
    "assert_positive_semidefinite",
    "assert_process_fidelity_above",
    "assert_purity_above",
    "assert_qasm_roundtrip",
    "assert_sampler_distribution",
    "assert_schmidt_rank_at_most",
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
