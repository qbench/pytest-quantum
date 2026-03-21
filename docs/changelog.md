# Changelog

All notable changes are documented here following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [0.3.0] - 2026-03-21

### Added

- **Pytket support**: `to_unitary`, depth/width/gate-count, `assert_circuit_is_clifford` via `UnitaryTableau`
- **Stim support**: `stim_sampler` fixture + 3 QEC assertions: `assert_stim_logical_error_rate_below`, `assert_stim_detector_error_rate_below`, `assert_stabilizer_state`
- **Channel assertions**: `assert_hermitian`, `assert_positive_semidefinite`, `assert_commutes_with`, `assert_channel_is_cptp`, `assert_process_fidelity_above`, `assert_noise_fidelity_above`
- **Entanglement assertions**: `assert_entanglement_entropy_below`, `assert_bloch_sphere_close`, `assert_schmidt_rank_at_most`
- **Information theory**: `assert_hellinger_close`, `assert_kl_divergence_below`, `assert_cross_entropy_below`
- **`assert_normalized`**: validate that a statevector has unit norm (catches forgotten normalization bugs)
- **Random generators** (`pytest_quantum.random`): `random_statevector`, `random_density_matrix`, `random_unitary`, `random_kraus_channel`, `depolarizing_kraus`
- **`assert_has_diagram`**: text diagram comparison for Qiskit/Cirq/Pytket circuits
- **`assert_transpilation_preserves_semantics`**: Qiskit transpile + equivalence check
- **`assert_qasm_roundtrip`**: OpenQASM export/import round-trip (Qiskit + Cirq native JSON)
- `quantum_benchmark` fixture for circuit timing (integrates with pytest-benchmark)
- `shot_budget` fixture for manual shot counting
- `pytket_circuit_factory` and `stim_sampler` fixtures
- Extended `assert_circuit_is_clifford` to Braket, PennyLane, and Pytket (was Qiskit + Cirq only)
- Improved `pytest_assertrepr_compare`: Qiskit circuit diagrams and count dict tables in failure messages

### Fixed

- `assert_qasm_roundtrip` Cirq path: switched from broken `cirq.contrib.qasm_import` to `cirq.to_json`/`cirq.read_json`
- `_reverse_qubit_order`: added power-of-2 validation (silent corruption bug)
- `_partial_trace`: replaced iterative `np.trace` with provably correct `np.einsum` (axis index bug for non-contiguous qubits)
- `assertions/__init__.py`: was only exporting 8 original functions; now exports all 38 (including `assert_normalized`)

---

## [0.2.0] - 2026-03-21

### Added

- **Density matrix assertions**: `assert_density_matrix_close`, `assert_trace_distance_below`, `assert_purity_above`, `assert_partial_trace_close`
- **Observable/expectation value assertions**: `assert_expectation_value_close`, `assert_ground_state_energy_close`
- **Qiskit Primitives support**: `assert_sampler_distribution`, `assert_estimator_close`, `qiskit_sampler`, `qiskit_estimator` fixtures
- **Snapshot / golden-file testing**: `assert_unitary_snapshot`, `assert_distribution_snapshot`, `--quantum-update-snapshots` CLI flag
- **`assert_circuit_is_clifford`**: Qiskit and Cirq support
- **`cirq_sampler` fixture**: shot-based Cirq simulation returning count dicts
- **`--quantum-shots` and `--quantum-significance`** CLI options now correctly wired to `quantum_shots` / `quantum_significance` fixtures
- `py.typed` marker (PEP 561)

### Fixed

- Cirq shot simulation (previously raised `TypeError` for measurement assertions)
- Braket `assert_gate_count` (previously raised `NotImplementedError`)
- PennyLane `assert_circuit_depth` (now uses `qml.specs`)
- PennyLane gate count auto-dry-run (no longer requires prior execution)
- Cross-framework multi-qubit comparison (Qiskit little-endian ↔ Cirq big-endian)

---

## [0.1.1] - 2026-03-01

### Added

- Published to PyPI
- GitHub Actions CI with post-publish smoke test
- ReadTheDocs documentation

---

## [0.1.0] - 2026-02-15

### Added

- Initial release with core assertions: `assert_unitary`, `assert_circuits_equivalent`, `assert_states_close`, `assert_state_fidelity_above`, `assert_measurement_distribution`, `assert_counts_close`, `assert_circuit_depth`, `assert_circuit_width`, `assert_gate_count`
- Fixtures: `aer_simulator`, `aer_statevector_simulator`, `aer_noise_simulator`, `cirq_simulator`, `braket_simulator`, `graphix_backend`, `pennylane_device`
- Statistical utilities: `min_shots`, `recommended_shots`, `fidelity`, `tvd`, `chi_square_test`
- Markers: `quantum`, `quantum_slow`, `shots(n)`, `significance(p)`
- CLI: `--quantum-slow`, `--quantum-shots`, `--quantum-significance`
- Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane, Graphix
