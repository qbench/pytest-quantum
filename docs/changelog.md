# Changelog

## 0.1.0 (2026-03-21)

Initial release.

### Added

- `assert_unitary` — verify a circuit implements a specific unitary matrix (global-phase safe)
- `assert_circuits_equivalent` — compare circuits across frameworks; uses mqt.qcec fast path for Qiskit pairs
- `assert_state_fidelity_above` — fidelity-based state comparison, primary assertion for Graphix MBQC
- `assert_states_close` — strict elementwise state comparison up to global phase
- `assert_measurement_distribution` — chi-square goodness-of-fit for shot-based tests
- `assert_counts_close` — Total Variation Distance comparison between count dicts
- `assert_circuit_depth` — depth bounds for Qiskit, Cirq, and Braket circuits
- `assert_circuit_width` — qubit count assertion for Qiskit, Cirq, Braket, PennyLane
- `assert_gate_count` — gate count assertion for Qiskit, Cirq, and PennyLane
- `aer_simulator`, `aer_statevector_simulator` — session-scoped Qiskit/Aer fixtures
- `aer_noise_simulator` — configurable depolarizing noise fixture
- `cirq_simulator` — session-scoped Cirq fixture
- `braket_simulator` — session-scoped Amazon Braket LocalSimulator
- `graphix_backend` — session-scoped Graphix pattern runner
- `pennylane_device` — session-scoped PennyLane device factory
- `min_shots(epsilon)` — shot count for chi-square power analysis
- `recommended_shots(probs)` — shot count for chi-square validity
- `fidelity`, `tvd`, `tvd_from_counts`, `chi_square_test` — statistical primitives
- `--quantum-slow`, `--quantum-shots`, `--quantum-significance` CLI options
- `quantum`, `quantum_slow`, `shots(n)`, `significance(p)` markers
