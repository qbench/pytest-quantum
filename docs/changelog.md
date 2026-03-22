# Changelog

All notable changes are documented here.

---

## [1.0.0] — 2026-03-22

### First stable release

### Added

**Benchmarking (new module)**
- `assert_t2_above` — Hahn echo T2 coherence time
- `assert_t2star_above` — Ramsey (free induction decay) T2* coherence time
- `assert_interleaved_rb` — Interleaved RB for individual gate fidelity

**Quantum ML (new module)**
- `assert_xeb_fidelity_above` — Cross-entropy benchmarking (XEB) fidelity
- `assert_expressibility_above` — Haar-random expressibility score
- `assert_entanglement_capability_above` — Meyer-Wallach entanglement capability
- `assert_no_barren_plateau` — Gradient variance / barren plateau detection

**Circuit structure**
- `assert_no_mid_circuit_measurement` — detect mid-circuit measurements before hardware submission

**QASM**
- `assert_qasm2_roundtrip` — OpenQASM 2.0 export/import round-trip

**Fixtures**
- `multi_backend_runner` — parallel multi-backend circuit runner
- `benchmark_suite` — per-assertion timing collection
- `ionq_backend` — real IonQ hardware fixture
- `quantinuum_backend` — real Quantinuum hardware fixture

**Project**
- `CITATION.cff` — academic citation metadata
- `cliff.toml` — git-cliff changelog configuration

### Changed
- `qiskit-ibm-runtime` moved to optional dependency (`pip install pytest-quantum[ibm]`);
  users without IBM credentials no longer need it installed

### Fixed
- Divide-by-zero in Clifford RB matrix composition (replaced element-wise
  division with `np.vdot` phase comparison)
- ruff format on 7 files after automated CI check

---

## [0.5.0] — 2026-03-21

### Added

**Benchmarking module** (`pytest_quantum.assertions.benchmarking`)
- `assert_quantum_volume` — IBM QV protocol with binomial confidence test
- `assert_randomized_benchmarking` — 1-qubit Clifford RB with exponential decay fit
- `assert_t1_above` — T1 relaxation via delay circuits
- `assert_gate_fidelity_above` — reads backend calibration properties

**Cross-platform module** (`pytest_quantum.assertions.cross_platform`)
- `assert_cross_platform_equivalent` — qubit-endianness-aware cross-framework equivalence
- `assert_qiskit_cirq_equivalent`
- `assert_qiskit_pytket_equivalent`

**Noise models module** (`pytest_quantum.assertions.noise_models`)
- `assert_depolarizing_channel`, `assert_amplitude_damping_channel`, `assert_dephasing_channel`
- `assert_no_leakage`, `assert_channel_preserves_trace`, `assert_channel_diamond_norm_below`

**Mitiq PEC**
- `assert_pec_reduces_error`, `assert_pec_expectation_close`, `assert_error_mitigation_benchmark`

**Hardware fixtures**
- `ibm_backend` — real IBM Quantum backend (requires `IBM_QUANTUM_TOKEN`)
- `ionq_backend`, `quantinuum_backend`, `braket_cloud_device`, `quantum_hardware_info`

**Optional dependencies**
- `cvxpy` extra for diamond norm SDP computation

### Fixed
- MyPy strict mode: all new modules pass `mypy --strict`
- ruff lint: no warnings across all 65 source files

---

## [0.4.0] — 2026-03

### Added
- Mitiq error mitigation: `assert_zne_expectation_close`, `assert_zne_reduces_error`,
  `assert_cdr_reduces_error`, `assert_mitigation_improves_fidelity`
- Sweep assertions: `assert_circuit_sweep`, `assert_circuit_sweep_states`,
  `assert_parametrized_unitary_continuous`
- Compilation assertions: `assert_transpilation_equivalent`, `assert_transpilation_depth_below`,
  `assert_gate_count_after_transpilation`

---

## [0.3.0] — 2026-02

### Added
- Pytket and Stim framework support
- Channel assertions: `assert_channel_is_cptp`, `assert_process_fidelity_above`, etc.
- Entanglement assertions: `assert_entanglement_entropy_below`, `assert_bloch_sphere_close`
- Information theory: `assert_hellinger_close`, `assert_kl_divergence_below`
- Random generators: `random_statevector`, `random_unitary`, `random_kraus_channel`
- QASM round-trip: `assert_qasm_roundtrip`
- QEC assertions for Stim: `assert_stim_logical_error_rate_below`

---

## [0.2.0] — 2026-01

### Added
- Density matrix assertions: `assert_density_matrix_close`, `assert_purity_above`, etc.
- Qiskit Primitives: `assert_sampler_distribution`, `assert_estimator_close`
- Snapshot testing: `assert_unitary_snapshot`, `assert_distribution_snapshot`
- VQE/observable assertions

---

## [0.1.0] — 2025-12

Initial release with core assertions:
- `assert_unitary`, `assert_circuits_equivalent`
- `assert_state_fidelity_above`, `assert_states_close`, `assert_normalized`
- `assert_measurement_distribution`, `assert_counts_close`
- `assert_circuit_depth`, `assert_circuit_width`, `assert_gate_count`
- `aer_simulator`, `cirq_simulator`, `braket_simulator`, `pennylane_device` fixtures
