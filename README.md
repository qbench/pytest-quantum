# pytest-quantum

A cross-framework pytest plugin for quantum program testing.

[![PyPI](https://img.shields.io/pypi/v/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![Python](https://img.shields.io/pypi/pyversions/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/qbench/pytest-quantum/actions/workflows/ci.yml/badge.svg)](https://github.com/qbench/pytest-quantum/actions)
[![Docs](https://readthedocs.org/projects/pytest-quantum/badge/?version=latest)](https://pytest-quantum.readthedocs.io)

Test quantum programs the same way you test classical code — with `pytest`.
Works with **Qiskit**, **Cirq**, **Amazon Braket**, **PennyLane**, **Graphix**,
**Pytket**, and **Stim**.

**[Full documentation](https://pytest-quantum.readthedocs.io)**

---

## What's new in v0.3.0

- **Pytket and Stim support**: `pytket_circuit_factory`, `stim_sampler` fixtures + 3 QEC assertions
- **Quantum channel assertions**: `assert_channel_is_cptp`, `assert_process_fidelity_above`, `assert_noise_fidelity_above`, `assert_hermitian`, `assert_positive_semidefinite`, `assert_commutes_with`
- **Entanglement assertions**: `assert_entanglement_entropy_below`, `assert_bloch_sphere_close`, `assert_schmidt_rank_at_most`
- **Information-theoretic distribution tests**: `assert_hellinger_close`, `assert_kl_divergence_below`, `assert_cross_entropy_below`
- **Random state/circuit generators** (`pytest_quantum.random`): `random_statevector`, `random_density_matrix`, `random_unitary`, `random_kraus_channel`, `depolarizing_kraus`
- **OpenQASM round-trip testing**: `assert_qasm_roundtrip`
- **QEC assertions for Stim circuits**: `assert_stim_logical_error_rate_below`, `assert_stim_detector_error_rate_below`, `assert_stabilizer_state`
- **`assert_normalized`**: validate statevector has unit norm
- **`assert_has_diagram`**: text diagram comparison for Qiskit/Cirq/Pytket circuits
- **`assert_transpilation_preserves_semantics`**: Qiskit transpile + equivalence check

---

## Why pytest-quantum?

Quantum programs fail in ways classical tests don't handle:

| Problem | Without pytest-quantum | With pytest-quantum |
|---|---|---|
| Shot noise flakiness | `assert counts["00"] == 512` fails ~5% of runs | `assert_measurement_distribution` uses chi-square — only fails when distribution is genuinely wrong |
| Global phase | `np.allclose(U1, U2)` fails for physically identical states | `assert_unitary` handles global phase automatically |
| Framework boilerplate | Copy-paste `AerSimulator()` setup in every project | `aer_simulator` fixture injected automatically |
| Shot count guessing | Pick 1024 shots and hope | `min_shots(epsilon=0.02)` gives the statistically correct answer |
| No structure testing | No standard way to assert depth or gate counts | `assert_circuit_depth`, `assert_circuit_width`, `assert_gate_count` |
| No mixed-state testing | No standard way to test noisy density matrices | `assert_density_matrix_close`, `assert_purity_above`, `assert_trace_distance_below` |

---

## Installation

```bash
pip install pytest-quantum                  # core (no quantum SDK required)
pip install "pytest-quantum[qiskit]"        # + Qiskit + Aer
pip install "pytest-quantum[cirq]"          # + Cirq
pip install "pytest-quantum[braket]"        # + Amazon Braket
pip install "pytest-quantum[pennylane]"     # + PennyLane
pip install "pytest-quantum[graphix]"       # + Graphix (MBQC)
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
```

---

## Decision guide — which assertion to use?

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
| Circuit uses only Clifford gates | `assert_circuit_is_clifford` | — |
| QASM export/import preserves semantics | `assert_qasm_roundtrip` | — |
| Logical error rate of QEC code | `assert_stim_logical_error_rate_below` | — |
| Matrix is Hermitian | `assert_hermitian` | — |
| Two operators commute | `assert_commutes_with` | — |
| Statevector is normalized | `assert_normalized` | — |

---

## All 38 assertions

### Unitary / circuit equivalence
```python
assert_unitary(circuit, expected_matrix)          # verifies circuit implements this unitary
assert_circuits_equivalent(circuit_a, circuit_b)  # two circuits are equivalent (cross-framework)
assert_transpilation_preserves_semantics(orig, compiled)  # transpilation is semantics-preserving
```

### State assertions
```python
assert_normalized(statevector)                    # ||ψ||₂ = 1  (v0.3.0)
assert_state_fidelity_above(actual, target)       # |⟨actual|target⟩|² ≥ threshold
assert_states_close(actual, target, atol=1e-6)    # elementwise, up to global phase
```

### Measurement distributions
```python
assert_measurement_distribution(counts, expected_probs)  # chi-square goodness-of-fit
assert_counts_close(counts_a, counts_b, max_tvd=0.05)   # Total Variation Distance
```

### Density matrix assertions *(v0.2.0)*
```python
assert_density_matrix_close(rho, sigma, atol=1e-6)
assert_trace_distance_below(rho, sigma, max_distance=0.01)
assert_purity_above(rho, min_purity=0.95)
assert_partial_trace_close(rho, keep_qubits, expected)
```

### Quantum channel assertions *(v0.3.0)*
```python
assert_hermitian(matrix)
assert_positive_semidefinite(matrix)
assert_commutes_with(op_a, op_b)
assert_channel_is_cptp(kraus_ops)
assert_process_fidelity_above(channel_a, channel_b, threshold=0.99)
assert_noise_fidelity_above(noisy_dm, ideal_state, threshold=0.99)
```

### Entanglement assertions *(v0.3.0)*
```python
assert_entanglement_entropy_below(sv, partition, max_entropy)
assert_bloch_sphere_close(sv, theta, phi, atol=0.1)
assert_schmidt_rank_at_most(sv, partition, max_rank)
```

### Information theory *(v0.3.0)*
```python
assert_hellinger_close(counts_a, counts_b, max_distance=0.1)
assert_kl_divergence_below(counts, expected_probs, max_kl=0.1)
assert_cross_entropy_below(counts, expected_probs, max_ce=1.0)
```

### Observable / expectation value *(v0.2.0)*
```python
assert_expectation_value_close(actual, expected, atol=0.01)
assert_ground_state_energy_close(actual_energy, expected_energy, atol=0.01)
```

### Qiskit Primitives *(v0.2.0)*
```python
assert_sampler_distribution(sampler_result, expected_probs)
assert_estimator_close(estimator_result, expected, atol=0.01)
```

### Circuit structure
```python
assert_circuit_depth(circuit, max_depth=10)
assert_circuit_width(circuit, expected_qubits=3)
assert_gate_count(circuit, "cx", expected=2)
assert_circuit_is_clifford(circuit)
assert_has_diagram(circuit, expected_diagram)         # (v0.3.0)
```

### Snapshots / golden-file testing *(v0.2.0)*
```python
assert_unitary_snapshot(circuit, name)
assert_distribution_snapshot(counts, name, max_tvd=0.05)
```

### OpenQASM round-trip *(v0.3.0)*
```python
assert_qasm_roundtrip(circuit)
```

### QEC / Stim *(v0.3.0)*
```python
assert_stim_logical_error_rate_below(circuit, max_error_rate, shots=10000)
assert_stim_detector_error_rate_below(circuit, max_error_rate, shots=10000)
assert_stabilizer_state(statevector, stabilizers)
```

---

## Framework support

| Framework | Version | Fixtures | assert_unitary | assert_circuit_is_clifford | assert_gate_count |
|---|---|---|---|---|---|
| Qiskit + Aer | ≥ 1.0 | `aer_simulator`, `aer_statevector_simulator`, `aer_noise_simulator`, `qiskit_sampler`, `qiskit_estimator` | yes | yes | yes |
| Cirq | ≥ 1.0 | `cirq_simulator`, `cirq_sampler` | yes | yes | yes |
| Amazon Braket | ≥ 1.0 | `braket_simulator` | yes | yes | yes |
| PennyLane | ≥ 0.36 | `pennylane_device` | yes | yes | yes |
| Graphix | ≥ 0.3 | `graphix_backend` | — | — | — |
| Pytket | ≥ 1.0 | `pytket_circuit_factory` | yes | yes | yes |
| Stim | ≥ 1.13 | `stim_sampler` | — | — | — |

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
| `graphix_backend` | Graphix | backend with `.run_pattern(pattern)` |
| `pennylane_device` | PennyLane | `make_device(wires, shots=None)` factory |
| `pytket_circuit_factory` | Pytket | `pytket.Circuit` class |
| `stim_sampler` | Stim | `sample_fn(circuit, shots)` callable |
| `quantum_benchmark` | All | benchmark wrapper |
| `shot_budget` | All | shot counter |
| `quantum_shots` | All | `int \| None` from `--quantum-shots` |
| `quantum_significance` | All | `float \| None` from `--quantum-significance` |

---

## Markers

```python
@pytest.mark.quantum                # tag as a quantum test
@pytest.mark.quantum_slow           # skip unless --quantum-slow is passed
@pytest.mark.shots(n=4000)          # shot count hint for this test
@pytest.mark.significance(p=0.01)   # p-value threshold for this test
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

## Random generators *(v0.3.0)*

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
| `--quantum-shots N` | Override shot count for all tests |
| `--quantum-significance P` | Override p-value threshold globally |
| `--quantum-update-snapshots` | Regenerate all snapshot files |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, test commands, code style, and PR checklist.

```bash
git clone https://github.com/qbench/pytest-quantum
cd pytest-quantum
uv sync --all-extras --group dev
uv run pytest          # 401+ tests
uv run ruff check src/ tests/
uv run mypy src/
```

---

## License

MIT — see [LICENSE](LICENSE).
