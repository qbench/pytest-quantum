# Changelog

All notable changes to pytest-quantum will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

### Added
- **Adapter abstraction layer** (`pytest_quantum.adapters`): formal `FrameworkAdapter` Protocol
  with per-framework adapter classes and a registry for user-extensible framework support.
- **Configuration support**: `[tool.pytest.ini_options]` / `pytest.ini` settings for
  `quantum_shots`, `quantum_significance`, `quantum_slow`, `quantum_real`.
- **`@pytest.mark.quantum_retry(n=N)`**: retry flaky quantum tests up to N times.
- **`--quantum-only`**: run only quantum-marked tests.
- **`--quantum-report=json|html`**: quantum-specific test reports with shot counts,
  framework coverage, and benchmark data.
- **Terminal summary**: quantum test statistics printed at end of session.
- **New assertion modules**:
  - `resource_estimation`: `assert_t_count_below`, `assert_ancilla_count_below`,
    `assert_clifford_t_depth_below`
  - `topology`: `assert_circuit_respects_topology`, `assert_routing_overhead_below`
  - `tomography`: `assert_state_tomography_close`, `assert_process_tomography_close`
  - `qec`: `assert_code_distance`, `assert_syndrome_decoding_correct`
- **Random circuit generators**: `random_qiskit_circuit`, `random_cirq_circuit`,
  `random_braket_circuit`, `random_pennylane_circuit`
- **Hypothesis circuit strategies**: `qiskit_circuits`, `cirq_circuits`,
  `braket_circuits`, `pennylane_circuits`
- **CUDA Quantum support**: adapter, fixture (`cuda_quantum_simulator`), integration tests
- **Qibo support**: adapter, fixture (`qibo_backend`), integration tests
- CI jobs for Pytket, Stim, Braket, Mitiq, and Hypothesis

### Changed
- **BREAKING: Plugin architecture**: `plugin.py` split into focused fixture
  submodules (`fixtures/qiskit.py`, `fixtures/cirq.py`, `fixtures/hardware.py`,
  `fixtures/benchmarks.py`, `fixtures/other.py`). Code that imported directly
  from ``pytest_quantum.plugin`` (e.g. fixture helpers) must update imports to
  the new submodule paths.
- **`@pytest.mark.shots(n)` and `@pytest.mark.significance(p)` now work**: per-test
  markers override CLI and ini values.
- `converters/to_unitary.py` now delegates to the adapter registry internally.
- `cirq_sampler` fixture reuses simulator instance instead of creating one per call.

### Fixed
- Deduplicated internal helpers: global-phase comparison (6 copies -> 1),
  `_extract_sampler_counts` (4 copies -> 1), `_kraus_to_choi` (2 copies -> 1),
  `_is_ibm_backend` (2 copies -> 1), `_backend_name` (3 copies -> 1).
- `hypothesis_strategies` module now exported from top-level `__init__.py`.
- `quantum_ml` assertions now documented in `__init__.py` docstring.

## [1.0.0] - 2025-01-15

### Added
- Initial stable release with support for Qiskit, Cirq, Braket, PennyLane,
  Graphix, Pytket, Stim, QuTiP, and Tequila.
- Assertion modules: states, distributions, unitary, structure, cross-platform,
  snapshot, sweeps, compilation, channels, noise models, benchmarking,
  quantum ML, hardware, observables, QASM, primitives, error mitigation.
- Random state generators and Hypothesis strategies.
- Pytest fixtures for all supported frameworks.
- CI pipeline with multi-platform testing.
