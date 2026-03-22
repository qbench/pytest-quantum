# pytest-quantum

A cross-framework pytest plugin for quantum program testing.

[![PyPI](https://img.shields.io/pypi/v/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![Python](https://img.shields.io/pypi/pyversions/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/qbench/pytest-quantum/actions/workflows/ci.yml/badge.svg)](https://github.com/qbench/pytest-quantum/actions)
[![Docs](https://readthedocs.org/projects/pytest-quantum/badge/?version=latest)](https://pytest-quantum.readthedocs.io)

Test quantum programs the same way you test classical code, using `pytest`.
Works with **Qiskit**, **Cirq**, **Amazon Braket**, **PennyLane**, **Graphix**,
**Pytket**, **Stim**, **QuTiP**, **Tequila**, and **Mitiq**.

**[Full documentation](https://pytest-quantum.readthedocs.io)**

---

## Who should use pytest-quantum?

- **Quantum software engineers** writing gate-model circuits in Qiskit, Cirq, PennyLane, or Pytket who need robust, reproducible unit tests.
- **Researchers running VQE or QAOA** who need statistically sound assertions on variational algorithm convergence and expressibility.
- **Teams CI-testing quantum algorithms** across multiple frameworks and backends, including noisy simulation and hardware targets.
- **Hardware validation engineers** characterising real devices with quantum volume, randomised benchmarking, T1/T2/T2* coherence, and XEB fidelity.

---

## What's new in v1.0.0

- **Benchmarking assertions**: `assert_quantum_volume`, `assert_randomized_benchmarking`, `assert_t1_above`, `assert_t2_above`, `assert_t2star_above`, `assert_interleaved_rb`, `assert_gate_fidelity_above`
- **Quantum ML assertions**: `assert_xeb_fidelity_above`, `assert_expressibility_above`, `assert_entanglement_capability_above`, `assert_no_barren_plateau`
- **Cross-platform equivalence**: `assert_cross_platform_equivalent`, `assert_qiskit_cirq_equivalent`, `assert_qiskit_pytket_equivalent`
- **Noise channel assertions**: `assert_depolarizing_channel`, `assert_amplitude_damping_channel`, `assert_dephasing_channel`, `assert_no_leakage`, `assert_channel_preserves_trace`, `assert_channel_diamond_norm_below`
- **Hardware assertions**: `assert_backend_calibration`, `assert_backend_executes`, `assert_circuit_fits_backend`, `assert_mirror_fidelity`, `assert_real_counts_close`
- **Mitiq error mitigation**: `assert_zne_expectation_close`, `assert_zne_reduces_error`, `assert_cdr_reduces_error`, `assert_mitigation_improves_fidelity`, `assert_pec_reduces_error`, `assert_pec_expectation_close`, `assert_error_mitigation_benchmark`
- **Sweep / variational**: `assert_circuit_sweep`, `assert_circuit_sweep_states`, `assert_parametrized_unitary_continuous`
- **Hardware fixtures**: `ibm_backend`, `ionq_backend`, `quantinuum_backend`, `quantum_hardware_info`
- **New fixtures**: `multi_backend_runner` (parallel multi-backend comparison), `benchmark_suite` (assertion timing)
- **QASM 2.0 round-trip**: `assert_qasm2_roundtrip`
- **Mid-circuit measurement detection**: `assert_no_mid_circuit_measurement`
- **qiskit-ibm-runtime is now optional** (`pip install pytest-quantum[ibm]`)

---

## Why pytest-quantum?

Quantum programs fail in ways classical tests don't handle:

| Problem | Without pytest-quantum | With pytest-quantum |
|---|---|---|
| Shot noise flakiness | `assert counts["00"] == 512` fails ~5% of runs | `assert_measurement_distribution` uses chi-square; fails only when distribution is genuinely wrong |
| Global phase | `np.allclose(U1, U2)` fails for physically identical states | `assert_unitary` handles global phase automatically |
| Framework boilerplate | Copy-paste `AerSimulator()` setup in every project | `aer_simulator` fixture injected automatically |
| Shot count guessing | Pick 1024 shots and hope | `min_shots(epsilon=0.02)` gives the statistically correct answer |
| No structure testing | No standard way to assert depth or gate counts | `assert_circuit_depth`, `assert_circuit_width`, `assert_gate_count` |
| No mixed-state testing | No standard way to test noisy density matrices | `assert_density_matrix_close`, `assert_purity_above`, `assert_trace_distance_below` |
| No hardware benchmarking | No built-in quantum volume or RB | `assert_quantum_volume`, `assert_randomized_benchmarking`, `assert_t1_above` |
| No cross-platform validation | Can't compare Qiskit vs Cirq circuit results | `assert_cross_platform_equivalent`, `assert_qiskit_cirq_equivalent` |

---

## Installation

```bash
pip install pytest-quantum                  # core (no quantum SDK required)
pip install "pytest-quantum[qiskit]"        # + Qiskit + Aer
pip install "pytest-quantum[cirq]"          # + Cirq
pip install "pytest-quantum[braket]"        # + Amazon Braket
pip install "pytest-quantum[pennylane]"     # + PennyLane
pip install "pytest-quantum[graphix]"       # + Graphix (MBQC)
pip install "pytest-quantum[ibm]"           # + qiskit-ibm-runtime (real IBM hardware)
pip install "pytest-quantum[mitiq]"         # + Mitiq error mitigation
pip install "pytest-quantum[cvxpy]"         # + CVXPY (diamond norm / SDP noise assertions)
pip install "pytest-quantum[all]"           # everything
pip install stim                            # + Stim (QEC)
pip install pytket                          # + Pytket
```

---

## Quick start (Qiskit)

```python
# test_bell.py — no conftest.py needed, fixtures are injected automatically
from pytest_quantum import assert_measurement_distribution, assert_unitary


def test_bell_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=2000).result().get_counts()

    # Chi-square test: won't flake on statistical noise
    assert_measurement_distribution(counts, expected_probs={"00": 0.5, "11": 0.5})


def test_hadamard_unitary():
    import numpy as np
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    # Global-phase-safe — e^(i*theta)*H passes too
    assert_unitary(qc, H)
```

## Quick start (PennyLane)

```python
# test_pennylane.py
import numpy as np
from pytest_quantum import assert_state_fidelity_above


def test_rx_gate(pennylane_device):
    import pennylane as qml

    dev = pennylane_device(wires=1)

    @qml.qnode(dev)
    def rx_circuit(theta):
        qml.RX(theta, wires=0)
        return qml.state()

    state = np.array(rx_circuit(np.pi))  # RX(π) = -iX|0⟩ = -i|1⟩
    assert_state_fidelity_above(state, np.array([0, -1j]), threshold=0.99)
```

## Quick start (Cirq)

```python
# test_cirq.py
import math
import numpy as np
from pytest_quantum import assert_unitary


def test_hadamard_cirq():
    import cirq

    q = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.H(q[0]))
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    assert_unitary(circuit, H)
```

```bash
pytest                        # normal suite
pytest --quantum-slow         # include shot-heavy tests
pytest --quantum-shots=4000   # override shot count globally
pytest --quantum-real         # enable real hardware tests (requires credentials)
```

---

## Decision guide: which assertion to use?

| I want to test... | Best assertion | Alternative |
|---|---|---|
| A gate implements a specific unitary | `assert_unitary` | `assert_circuits_equivalent` |
| Two circuits are equivalent | `assert_circuits_equivalent` | `assert_unitary` |
| A noisy circuit's output state | `assert_state_fidelity_above` | `assert_trace_distance_below` |
| Measurement distribution matches expected | `assert_measurement_distribution` | `assert_counts_close` |
| Two measurement distributions are close | `assert_counts_close` | `assert_hellinger_close` |
| A density matrix from noisy simulation | `assert_density_matrix_close` | `assert_trace_distance_below` |
| How mixed/noisy a state is | `assert_purity_above` | `assert_trace_distance_below` |
| Entanglement in a pure state | `assert_entanglement_entropy_below` | `assert_schmidt_rank_at_most` |
| Single-qubit state on Bloch sphere | `assert_bloch_sphere_close` | `assert_states_close` |
| Quantum channel is valid | `assert_channel_is_cptp` | `assert_process_fidelity_above` |
| VQE / QAOA energy result | `assert_ground_state_energy_close` | `assert_expectation_value_close` |
| Circuit doesn't change after refactor | `assert_unitary_snapshot` | `assert_distribution_snapshot` |
| Circuit uses only Clifford gates | `assert_circuit_is_clifford` |  |
| QASM export/import preserves semantics | `assert_qasm_roundtrip` (v3) or `assert_qasm2_roundtrip` (v2) |  |
| Logical error rate of QEC code | `assert_stim_logical_error_rate_below` |  |
| Backend meets quantum volume spec | `assert_quantum_volume` |  |
| Gate fidelity from RB experiment | `assert_randomized_benchmarking` | `assert_interleaved_rb` |
| T1 / T2 / T2* coherence times | `assert_t1_above` / `assert_t2_above` / `assert_t2star_above` |  |
| Cross-entropy benchmarking (XEB) | `assert_xeb_fidelity_above` |  |
| VQA circuit is expressive | `assert_expressibility_above` | `assert_entanglement_capability_above` |
| No barren plateau in VQA training | `assert_no_barren_plateau` |  |
| Circuit compatible with hardware | `assert_circuit_fits_backend` | `assert_no_mid_circuit_measurement` |
| Error mitigation improves results | `assert_zne_reduces_error` | `assert_pec_reduces_error` |

---

## All 80+ assertions

### Unitary / circuit equivalence
```python
assert_unitary(circuit, expected_matrix)
assert_circuits_equivalent(circuit_a, circuit_b)
assert_transpilation_preserves_semantics(orig, compiled)
assert_cross_platform_equivalent(circuit_a, circuit_b)  # v1.0.0
assert_qiskit_cirq_equivalent(qiskit_qc, cirq_circuit)  # v1.0.0
assert_qiskit_pytket_equivalent(qiskit_qc, pytket_circ)  # v1.0.0
```

### State assertions
```python
assert_normalized(statevector)
assert_state_fidelity_above(actual, target, threshold=0.99)
assert_states_close(actual, target, atol=1e-6)
```

### Measurement distributions
```python
assert_measurement_distribution(counts, expected_probs)
assert_counts_close(counts_a, counts_b, max_tvd=0.05)
```

### Density matrix assertions
```python
assert_density_matrix_close(rho, sigma, atol=1e-6)
assert_trace_distance_below(rho, sigma, max_distance=0.01)
assert_purity_above(rho, min_purity=0.95)
assert_partial_trace_close(rho, keep_qubits, expected)
```

### Quantum channel assertions
```python
assert_hermitian(matrix)
assert_positive_semidefinite(matrix)
assert_commutes_with(op_a, op_b)
assert_channel_is_cptp(kraus_ops)
assert_process_fidelity_above(channel_a, channel_b, threshold=0.99)
assert_noise_fidelity_above(noisy_dm, ideal_state, threshold=0.99)
```

### Noise channel assertions *(v1.0.0)*
```python
assert_depolarizing_channel(kraus_ops, error_rate, atol=1e-6)
assert_amplitude_damping_channel(kraus_ops, gamma, atol=1e-6)
assert_dephasing_channel(kraus_ops, p_dephase, atol=1e-6)
assert_no_leakage(kraus_ops, computational_dim=2)
assert_channel_preserves_trace(kraus_ops)
assert_channel_diamond_norm_below(kraus_ops_a, kraus_ops_b, max_norm=0.01)
```

### Entanglement assertions
```python
assert_entanglement_entropy_below(sv, partition, max_entropy)
assert_bloch_sphere_close(sv, expected_theta, expected_phi, atol=0.1)
assert_schmidt_rank_at_most(sv, partition, max_rank)
```

### Information theory
```python
assert_hellinger_close(counts_a, counts_b, max_distance=0.1)
assert_kl_divergence_below(counts, expected_probs, max_kl=0.1)
assert_cross_entropy_below(counts, expected_probs, max_ce=1.0)
```

### Observable / expectation value
```python
assert_expectation_value_close(actual, expected, atol=0.01)
assert_ground_state_energy_close(actual_energy, expected_energy, atol=0.01)
assert_vqe_converges(vqe_fn, hamiltonian, target_energy, atol=0.1)
assert_cost_decreases(cost_fn, initial_params)
```

### Qiskit Primitives
```python
assert_sampler_distribution(sampler_result, expected_probs)
assert_estimator_close(estimator_result, expected, atol=0.01)
```

### Circuit structure
```python
assert_circuit_depth(circuit, max_depth=10)
assert_circuit_width(circuit, expected_qubits=3)
assert_gate_count(circuit, "cx", expected=2)
assert_gates_in_basis_set(circuit, basis_gates={"cx", "u3"})
assert_circuit_is_clifford(circuit)
assert_has_diagram(circuit, expected_diagram)
assert_no_mid_circuit_measurement(circuit)  # v1.0.0
```

### Transpilation / compilation
```python
assert_transpilation_equivalent(circuit, backend, atol=1e-6)
assert_transpilation_depth_below(circuit, backend, max_depth=20)
assert_gate_count_after_transpilation(circuit, backend, gate, expected)
```

### Sweeps / parametrised circuits
```python
assert_circuit_sweep(circuit, param_values, expected_probs_list)
assert_circuit_sweep_states(circuit, param_values, expected_states)
assert_parametrized_unitary_continuous(circuit, param_range)
```

### Snapshots / golden-file testing
```python
assert_unitary_snapshot(circuit, name)
assert_distribution_snapshot(counts, name, max_tvd=0.05)
```

### QASM round-trips
```python
assert_qasm_roundtrip(circuit)    # OpenQASM 3 (Qiskit) or Cirq JSON
assert_qasm2_roundtrip(circuit)   # OpenQASM 2.0 (Qiskit only) — v1.0.0
```

### QEC / Stim
```python
assert_stim_logical_error_rate_below(circuit, max_error_rate, shots=10000)
assert_stim_detector_error_rate_below(circuit, max_error_rate, shots=10000)
assert_stabilizer_state(statevector, stabilizers)
```

### Benchmarking *(v1.0.0)*
```python
assert_quantum_volume(backend, target_qv=16, num_trials=100)
assert_randomized_benchmarking(backend, qubit=0, min_fidelity_per_clifford=0.999)
assert_t1_above(backend, qubit=0, target_t1_us=50.0)
assert_t2_above(backend, qubit=0, target_t2_us=30.0)      # Hahn echo
assert_t2star_above(backend, qubit=0, target_t2star_us=20.0)  # Ramsey
assert_interleaved_rb(backend, qubit=0, gate_name="X", gate_circuit=x_circ)
assert_gate_fidelity_above(backend, "cx", [0, 1], target_fidelity=0.99)
```

### Quantum ML *(v1.0.0)*
```python
assert_xeb_fidelity_above(backend, num_qubits=2, target_fidelity=0.9)
assert_expressibility_above(ansatz_fn, num_qubits=2, num_params=4, target=0.5)
assert_entanglement_capability_above(ansatz_fn, num_qubits=2, num_params=4, target=0.3)
assert_no_barren_plateau(ansatz_fn, num_qubits=4, num_params=16)
```

### Hardware assertions *(v1.0.0)*
```python
assert_backend_calibration(backend, min_t1_us=30.0, min_cx_fidelity=0.99)
assert_backend_executes(circuit, backend, shots=1024)
assert_circuit_fits_backend(circuit, backend)
assert_mirror_fidelity(backend, qubit, target_fidelity=0.95)
assert_real_counts_close(job, expected_probs, max_tvd=0.1)
```

### Mitiq error mitigation *(v0.4.0+)*
```python
assert_zne_expectation_close(circuit, observable, expected, atol=0.1)
assert_zne_reduces_error(circuit, observable, noisy_val, ideal_val)
assert_cdr_reduces_error(circuit, observable, noisy_val, ideal_val)
assert_mitigation_improves_fidelity(circuit, noisy_state, ideal_state)
assert_pec_reduces_error(circuit, observable, noisy_val, ideal_val)
assert_pec_expectation_close(circuit, observable, expected, atol=0.1)
assert_error_mitigation_benchmark(circuit, observable, methods=["zne", "cdr"])
```

---

## Framework support

| Framework | Version | Fixtures | Unitary | Clifford | Gate count |
|---|---|---|---|---|---|
| Qiskit + Aer | ≥ 1.0 | `aer_simulator`, `aer_statevector_simulator`, `aer_noise_simulator`, `qiskit_sampler`, `qiskit_estimator` | ✓ | ✓ | ✓ |
| Cirq | ≥ 1.0 | `cirq_simulator`, `cirq_sampler` | ✓ | ✓ | ✓ |
| Amazon Braket | ≥ 1.0 | `braket_simulator`, `braket_cloud_device` | ✓ | ✓ | ✓ |
| PennyLane | ≥ 0.36 | `pennylane_device` | ✓ | ✓ | ✓ |
| Graphix | ≥ 0.3 | `graphix_backend` | ✗ | ✗ | ✗ |
| Pytket | ≥ 1.0 | `pytket_circuit_factory` | ✓ | ✓ | ✓ |
| Stim | ≥ 1.13 | `stim_sampler` | ✗ | ✗ | ✗ |
| QuTiP | ≥ 4.7 | `qutip_solver` | ✗ | ✗ | ✗ |
| Tequila | ≥ 1.9 | `tequila_backend` | ✗ | ✗ | ✗ |
| IBM Quantum | runtime ≥ 0.45 | `ibm_backend` | ✓ | ✗ | ✗ |
| IonQ | via qiskit-ionq | `ionq_backend` | ✗ | ✗ | ✗ |
| Quantinuum | via pytket-quantinuum | `quantinuum_backend` | ✗ | ✗ | ✗ |

---

## Fixtures

All fixtures are **auto-discovered** (no imports needed) and **skip automatically**
if the required SDK is not installed.

| Fixture | Framework | Returns |
|---|---|---|
| `aer_simulator` | Qiskit / Aer | `AerSimulator()` |
| `aer_statevector_simulator` | Qiskit / Aer | `AerSimulator(method="statevector")` |
| `aer_noise_simulator` | Qiskit / Aer | `make_simulator(error_rate)` factory |
| `qiskit_sampler` | Qiskit 1.0+ | `StatevectorSampler()` |
| `qiskit_estimator` | Qiskit 1.0+ | `StatevectorEstimator()` |
| `cirq_simulator` | Cirq | `cirq.Simulator()` |
| `cirq_sampler` | Cirq | `run_fn(circuit, shots)` callable |
| `braket_simulator` | Amazon Braket | `LocalSimulator()` |
| `braket_cloud_device` | Amazon Braket cloud | `AwsDevice(arn)` (requires `--quantum-real`) |
| `graphix_backend` | Graphix | backend with `.run_pattern(pattern)` |
| `pennylane_device` | PennyLane | `make_device(wires, shots=None)` factory |
| `pytket_circuit_factory` | Pytket | `pytket.Circuit` class |
| `stim_sampler` | Stim | `sample_fn(circuit, shots)` callable |
| `qutip_solver` | QuTiP | `solve(H, psi0, tlist, c_ops)` callable |
| `tequila_backend` | Tequila | `tequila` module |
| `ibm_backend` | IBM Quantum | `IBMBackend` (requires `--quantum-real`) |
| `ionq_backend` | IonQ | IonQ backend (requires `--quantum-real`) |
| `quantinuum_backend` | Quantinuum | Quantinuum backend (requires `--quantum-real`) |
| `quantum_hardware_info` | All | Dict of credential availability |
| `quantum_benchmark` | All | benchmark timing wrapper |
| `benchmark_suite` | All | assertion timing suite (v1.0.0) |
| `shot_budget` | All | shot counter |
| `multi_backend_runner` | All | parallel multi-backend runner (v1.0.0) |
| `quantum_shots` | All | `int \| None` from `--quantum-shots` |
| `quantum_significance` | All | `float \| None` from `--quantum-significance` |

---

## Markers

```python
@pytest.mark.quantum                # tag as a quantum test
@pytest.mark.quantum_slow           # skip unless --quantum-slow is passed
@pytest.mark.quantum_real           # skip unless --quantum-real is passed (hardware tests)
@pytest.mark.shots(n=4000)          # shot count hint for this test
@pytest.mark.significance(p=0.01)   # p-value threshold for this test
@pytest.mark.quantum_backends("qiskit", "cirq", "pennylane")  # parametrize over backends
```

---

## Shot budget utilities

```python
from pytest_quantum import min_shots, recommended_shots

n = min_shots(epsilon=0.05)                            # 293 shots to detect 5% TVD
n = recommended_shots({"00": 0.499, "01": 0.001, "11": 0.5})  # 5000 (driven by 0.1% outcome)
```

---

## Statistical primitives

```python
from pytest_quantum import fidelity, tvd, tvd_from_counts, chi_square_test

fidelity(psi, phi)                       # |<psi|phi>|^2, global-phase invariant
tvd(p, q)                                # Total Variation Distance (0=identical, 1=disjoint)
tvd_from_counts(counts_a, counts_b)      # TVD from count dicts
chi_square_test(counts, expected_probs)  # returns (statistic, p_value)
```

---

## Random generators

```python
from pytest_quantum.random import (
    random_statevector,      # Haar-random pure state
    random_density_matrix,   # random mixed state
    random_unitary,          # Haar-random unitary
    random_kraus_channel,    # random CPTP channel
    depolarizing_kraus,      # depolarizing channel Kraus operators
)
```

---

## CLI options

| Option | Description |
|---|---|
| `--quantum-slow` | Run `quantum_slow`-marked tests (skipped by default) |
| `--quantum-real` | Run `quantum_real`-marked tests against real hardware |
| `--quantum-shots N` | Override shot count for all tests |
| `--quantum-significance P` | Override p-value threshold globally |
| `--quantum-update-snapshots` | Regenerate all snapshot files |

---

## Academic citation

If you use pytest-quantum in research, please cite it:

```bibtex
@software{ghatule2026pytest_quantum,
  title  = {pytest-quantum: A cross-framework pytest plugin for quantum program testing},
  author = {Ghatule, Tejas},
  year   = {2026},
  url    = {https://github.com/qbench/pytest-quantum},
  version = {1.0.0}
}
```

See [CITATION.cff](CITATION.cff) for full metadata.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, test commands, code style, and PR checklist.

```bash
git clone https://github.com/qbench/pytest-quantum
cd pytest-quantum
uv sync --all-extras --group dev
uv run pytest          # 646+ tests
uv run ruff check src/ tests/
uv run mypy src/
```

---

## License

MIT. See [LICENSE](LICENSE).
