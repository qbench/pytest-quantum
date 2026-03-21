# pytest-quantum

A cross-framework pytest plugin for quantum program testing.

[![PyPI](https://img.shields.io/pypi/v/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![Python](https://img.shields.io/pypi/pyversions/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/qbench/pytest-quantum/actions/workflows/ci.yml/badge.svg)](https://github.com/qbench/pytest-quantum/actions)
[![Docs](https://readthedocs.org/projects/pytest-quantum/badge/?version=latest)](https://pytest-quantum.readthedocs.io)

Test quantum programs the same way you test classical code — with `pytest`.
Works with **Qiskit**, **Cirq**, **Amazon Braket**, **PennyLane**, and **Graphix**.

**[Full documentation](https://pytest-quantum.readthedocs.io)**

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
```

---

## Quick start

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


def test_compiler_didnt_add_gates():
    from qiskit import QuantumCircuit

    from pytest_quantum import assert_circuit_depth, assert_gate_count

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    assert_circuit_depth(qc, max_depth=3)
    assert_gate_count(qc, "cx", expected=1)
```

```bash
pytest                        # normal suite
pytest --quantum-slow         # include shot-heavy tests
pytest --quantum-shots=4000   # override shot count globally
```

---

## Assertions

### Unitary / circuit equivalence

```python
from pytest_quantum import assert_unitary, assert_circuits_equivalent

assert_unitary(circuit, expected_matrix)          # verifies circuit implements this unitary
assert_circuits_equivalent(circuit_a, circuit_b)  # two circuits are equivalent
```

`assert_circuits_equivalent` uses **mqt.qcec** for fast exact verification on Qiskit pairs,
falling back to numpy matrix comparison for cross-framework cases.

> **Note on cross-framework multi-qubit circuits:** Qiskit uses little-endian qubit ordering
> and Cirq uses big-endian, so their CNOT matrices differ by a qubit permutation.
> `assert_circuits_equivalent` works correctly for single-qubit cross-framework comparisons.
> For multi-qubit cross-framework comparisons, compare within a single framework.

### State fidelity

```python
from pytest_quantum import assert_state_fidelity_above, assert_states_close

assert_state_fidelity_above(actual, target, threshold=0.99)  # |<actual|target>|^2 >= threshold
assert_states_close(actual, target, atol=1e-6)               # elementwise, up to global phase
```

Primary assertion for **Graphix** MBQC patterns where state vectors are the output.

### Measurement distributions

```python
from pytest_quantum import assert_measurement_distribution, assert_counts_close

# Chi-square goodness-of-fit — statistically sound, won't flake on shot noise
assert_measurement_distribution(counts, expected_probs={"00": 0.5, "11": 0.5})

# Total Variation Distance — compare two backends or pre/post-transpilation
assert_counts_close(counts_a, counts_b, max_tvd=0.05)
```

### Circuit structure

```python
from pytest_quantum import assert_circuit_depth, assert_circuit_width, assert_gate_count

assert_circuit_depth(circuit, max_depth=10)         # Qiskit, Cirq, Braket
assert_circuit_width(circuit, expected_qubits=3)    # Qiskit, Cirq, Braket, PennyLane
assert_gate_count(circuit, "cx", expected=2)        # Qiskit, Cirq, PennyLane
```

---

## Fixtures

All fixtures are **session-scoped** (initialised once per run) and **skip automatically**
if the required SDK is not installed.

| Fixture | Framework | Returns |
|---|---|---|
| `aer_simulator` | Qiskit / Aer | `AerSimulator()` |
| `aer_statevector_simulator` | Qiskit / Aer | `AerSimulator(method="statevector")` |
| `aer_noise_simulator` | Qiskit / Aer | `make_simulator(error_rate)` factory |
| `cirq_simulator` | Cirq | `cirq.Simulator()` |
| `braket_simulator` | Amazon Braket | `LocalSimulator()` |
| `graphix_backend` | Graphix | backend with `.run_pattern(pattern)` |
| `pennylane_device` | PennyLane | `make_device(wires, shots=None)` factory |

```python
# Noise simulator example
def test_noisy_circuit(aer_noise_simulator):
    sim = aer_noise_simulator(error_rate=0.01)   # 1% depolarizing noise
    # ... run your circuit on sim
```

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

n = min_shots(epsilon=0.05)                            # 1570 shots to detect 5% TVD
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

## CLI options

| Option | Description |
|---|---|
| `--quantum-slow` | Run `quantum_slow`-marked tests (skipped by default) |
| `--quantum-shots N` | Override shot count for all tests |
| `--quantum-significance P` | Override p-value threshold globally |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, test commands, code style, and PR checklist.

```bash
git clone https://github.com/qbench/pytest-quantum
cd pytest-quantum
uv sync --all-extras --group dev
uv run pytest          # 136+ tests
uv run ruff check src/ tests/
uv run mypy src/
```

---

## License

MIT — see [LICENSE](LICENSE).
