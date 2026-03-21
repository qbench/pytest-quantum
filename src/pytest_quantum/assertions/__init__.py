"""Quantum-aware assertion helpers for pytest.

All functions raise ``AssertionError`` with detailed, human-readable messages
on failure — designed to integrate naturally with pytest's output.
"""

from __future__ import annotations

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
    assert_normalized,
    assert_state_fidelity_above,
    assert_states_close,
)
from pytest_quantum.assertions.stim_assertions import (
    assert_stabilizer_state,
    assert_stim_detector_error_rate_below,
    assert_stim_logical_error_rate_below,
)
from pytest_quantum.assertions.structure import (
    assert_circuit_depth,
    assert_circuit_is_clifford,
    assert_circuit_width,
    assert_gate_count,
    assert_has_diagram,
)
from pytest_quantum.assertions.unitary import (
    assert_circuits_equivalent,
    assert_transpilation_preserves_semantics,
    assert_unitary,
)

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
    "assert_has_diagram",
    "assert_hellinger_close",
    "assert_hermitian",
    "assert_kl_divergence_below",
    "assert_measurement_distribution",
    "assert_noise_fidelity_above",
    "assert_normalized",
    "assert_partial_trace_close",
    "assert_positive_semidefinite",
    "assert_process_fidelity_above",
    "assert_purity_above",
    "assert_qasm_roundtrip",
    "assert_sampler_distribution",
    "assert_schmidt_rank_at_most",
    "assert_stabilizer_state",
    "assert_state_fidelity_above",
    "assert_states_close",
    "assert_stim_detector_error_rate_below",
    "assert_stim_logical_error_rate_below",
    "assert_trace_distance_below",
    "assert_transpilation_preserves_semantics",
    "assert_unitary",
    "assert_unitary_snapshot",
]
