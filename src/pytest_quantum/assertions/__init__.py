"""Quantum-aware assertion helpers for pytest.

All functions raise ``AssertionError`` with detailed, human-readable messages
on failure — designed to integrate naturally with pytest's output.
"""

from __future__ import annotations

from pytest_quantum.assertions.benchmarking import (
    assert_gate_fidelity_above,
    assert_interleaved_rb,
    assert_quantum_volume,
    assert_randomized_benchmarking,
    assert_t1_above,
    assert_t2_above,
    assert_t2star_above,
)
from pytest_quantum.assertions.channels import (
    assert_channel_is_cptp,
    assert_commutes_with,
    assert_hermitian,
    assert_noise_fidelity_above,
    assert_positive_semidefinite,
    assert_process_fidelity_above,
)
from pytest_quantum.assertions.compilation import (
    assert_gate_count_after_transpilation,
    assert_transpilation_depth_below,
    assert_transpilation_equivalent,
)
from pytest_quantum.assertions.cross_platform import (
    assert_cross_platform_equivalent,
    assert_qiskit_cirq_equivalent,
    assert_qiskit_pytket_equivalent,
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
from pytest_quantum.assertions.hardware import (
    assert_backend_calibration,
    assert_backend_executes,
    assert_circuit_fits_backend,
    assert_mirror_fidelity,
    assert_real_counts_close,
)
from pytest_quantum.assertions.information import (
    assert_cross_entropy_below,
    assert_hellinger_close,
    assert_kl_divergence_below,
)
from pytest_quantum.assertions.mitiq_assertions import (
    assert_cdr_reduces_error,
    assert_error_mitigation_benchmark,
    assert_mitigation_improves_fidelity,
    assert_pec_expectation_close,
    assert_pec_reduces_error,
    assert_zne_expectation_close,
    assert_zne_reduces_error,
)
from pytest_quantum.assertions.noise_models import (
    assert_amplitude_damping_channel,
    assert_channel_diamond_norm_below,
    assert_channel_preserves_trace,
    assert_dephasing_channel,
    assert_depolarizing_channel,
    assert_no_leakage,
)
from pytest_quantum.assertions.observables import (
    assert_cost_decreases,
    assert_expectation_value_close,
    assert_ground_state_energy_close,
    assert_vqe_converges,
)
from pytest_quantum.assertions.primitives import (
    assert_estimator_close,
    assert_sampler_distribution,
)
from pytest_quantum.assertions.qasm import assert_qasm2_roundtrip, assert_qasm_roundtrip
from pytest_quantum.assertions.quantum_ml import (
    assert_entanglement_capability_above,
    assert_expressibility_above,
    assert_no_barren_plateau,
    assert_xeb_fidelity_above,
)
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
    assert_gates_in_basis_set,
    assert_has_diagram,
    assert_no_mid_circuit_measurement,
)
from pytest_quantum.assertions.sweeps import (
    assert_circuit_sweep,
    assert_circuit_sweep_states,
    assert_parametrized_unitary_continuous,
)
from pytest_quantum.assertions.unitary import (
    assert_circuits_equivalent,
    assert_transpilation_preserves_semantics,
    assert_unitary,
)
from pytest_quantum.assertions.resource_estimation import (
    assert_ancilla_count_below,
    assert_clifford_t_depth_below,
    assert_t_count_below,
)
from pytest_quantum.assertions.topology import (
    assert_circuit_respects_topology,
    assert_routing_overhead_below,
)
from pytest_quantum.assertions.tomography import (
    assert_process_tomography_close,
    assert_state_tomography_close,
)
from pytest_quantum.assertions.qec import (
    assert_code_distance,
    assert_syndrome_decoding_correct,
)

__all__ = [
    "assert_amplitude_damping_channel",
    "assert_ancilla_count_below",
    "assert_backend_calibration",
    "assert_backend_executes",
    "assert_bloch_sphere_close",
    "assert_cdr_reduces_error",
    "assert_channel_diamond_norm_below",
    "assert_channel_is_cptp",
    "assert_channel_preserves_trace",
    "assert_circuit_depth",
    "assert_circuit_fits_backend",
    "assert_circuit_is_clifford",
    "assert_circuit_respects_topology",
    "assert_cost_decreases",
    "assert_circuit_sweep",
    "assert_circuit_sweep_states",
    "assert_circuit_width",
    "assert_circuits_equivalent",
    "assert_clifford_t_depth_below",
    "assert_code_distance",
    "assert_commutes_with",
    "assert_counts_close",
    "assert_cross_entropy_below",
    "assert_cross_platform_equivalent",
    "assert_density_matrix_close",
    "assert_dephasing_channel",
    "assert_depolarizing_channel",
    "assert_distribution_snapshot",
    "assert_entanglement_capability_above",
    "assert_entanglement_entropy_below",
    "assert_error_mitigation_benchmark",
    "assert_estimator_close",
    "assert_expectation_value_close",
    "assert_expressibility_above",
    "assert_gate_count",
    "assert_gate_count_after_transpilation",
    "assert_gate_fidelity_above",
    "assert_gates_in_basis_set",
    "assert_ground_state_energy_close",
    "assert_has_diagram",
    "assert_hellinger_close",
    "assert_hermitian",
    "assert_interleaved_rb",
    "assert_kl_divergence_below",
    "assert_measurement_distribution",
    "assert_mirror_fidelity",
    "assert_mitigation_improves_fidelity",
    "assert_no_barren_plateau",
    "assert_no_leakage",
    "assert_no_mid_circuit_measurement",
    "assert_noise_fidelity_above",
    "assert_normalized",
    "assert_parametrized_unitary_continuous",
    "assert_partial_trace_close",
    "assert_pec_expectation_close",
    "assert_pec_reduces_error",
    "assert_positive_semidefinite",
    "assert_process_fidelity_above",
    "assert_process_tomography_close",
    "assert_purity_above",
    "assert_qasm2_roundtrip",
    "assert_qasm_roundtrip",
    "assert_qiskit_cirq_equivalent",
    "assert_qiskit_pytket_equivalent",
    "assert_quantum_volume",
    "assert_randomized_benchmarking",
    "assert_real_counts_close",
    "assert_routing_overhead_below",
    "assert_sampler_distribution",
    "assert_schmidt_rank_at_most",
    "assert_stabilizer_state",
    "assert_state_fidelity_above",
    "assert_state_tomography_close",
    "assert_states_close",
    "assert_stim_detector_error_rate_below",
    "assert_stim_logical_error_rate_below",
    "assert_syndrome_decoding_correct",
    "assert_t1_above",
    "assert_t2_above",
    "assert_t2star_above",
    "assert_t_count_below",
    "assert_trace_distance_below",
    "assert_transpilation_depth_below",
    "assert_transpilation_equivalent",
    "assert_transpilation_preserves_semantics",
    "assert_unitary",
    "assert_unitary_snapshot",
    "assert_vqe_converges",
    "assert_xeb_fidelity_above",
    "assert_zne_expectation_close",
    "assert_zne_reduces_error",
]
