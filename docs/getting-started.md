# Getting Started

This guide walks you from a fresh install to your first passing quantum tests
in under five minutes.

---

## 5-Minute Quick Start

Install with Qiskit and Cirq support:

```bash
pip install "pytest-quantum[qiskit,cirq]"
```

Create `test_quick.py`:

```python
# test_quick.py — copy this and run immediately
import math
import numpy as np
from pytest_quantum import (
    assert_circuit_depth,
    assert_circuits_equivalent,
    assert_measurement_distribution,
    assert_unitary,
)

HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def test_h_gate_unitary():
    """H gate should implement the Hadamard unitary."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    assert_unitary(qc, HADAMARD)


def test_bell_measurement_distribution(aer_simulator):
    """Bell state should produce 50/50 counts."""
    from qiskit import QuantumCircuit, transpile
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=2000).result().get_counts()
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})


def test_bell_circuit_structure():
    """Bell circuit should have 2 qubits and depth ≤ 3."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    assert_circuit_depth(qc, max_depth=3)
```

Run it:

```bash
pytest test_quick.py -v
```

Expected output:

```
test_quick.py::test_h_gate_unitary PASSED
test_quick.py::test_bell_measurement_distribution PASSED
test_quick.py::test_bell_circuit_structure PASSED
```

---

## Installation

### Core package

```bash
pip install pytest-quantum
```

The core package depends only on `pytest`, `numpy`, and `scipy`. All assertion
and statistics utilities work without any quantum SDK installed.

### Framework extras

Install the extras for whichever frameworks you use:

```bash
pip install "pytest-quantum[qiskit]"    # Qiskit + Aer
pip install "pytest-quantum[cirq]"      # Cirq
pip install "pytest-quantum[braket]"    # Amazon Braket
pip install "pytest-quantum[pennylane]" # PennyLane
pip install "pytest-quantum[graphix]"   # Graphix (MBQC)
pip install "pytest-quantum[all]"       # all of the above
```

### Development install

```bash
git clone https://github.com/qbench/pytest-quantum
cd pytest-quantum
pip install -e ".[all]"
```

---

## First Qiskit test — Bell state distribution

The Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
should produce outcomes `"00"` and `"11"` with equal probability.

```python
# test_bell.py
import pytest
from pytest_quantum import assert_measurement_distribution, min_shots

@pytest.mark.quantum
def test_bell_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    shots = min_shots(epsilon=0.05)   # 293 shots — enough to catch 5% TVD
    qc_t = transpile(qc, aer_simulator)
    counts = aer_simulator.run(qc_t, shots=shots).result().get_counts()

    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
    )
```

Run it:

```bash
pytest test_bell.py -v
```

The `aer_simulator` fixture is injected automatically — no import needed. If
`qiskit-aer` is not installed, the test is skipped with a helpful message.

---

## First unitary test — Hadamard gate

For deterministic circuits, compare the full unitary matrix. This is stronger
than comparing shot distributions because it catches phase errors too.

```python
# test_hadamard.py
import numpy as np
import pytest
from pytest_quantum import assert_unitary

HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

@pytest.mark.quantum
def test_h_gate_qiskit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    assert_unitary(qc, HADAMARD)

@pytest.mark.quantum
def test_h_gate_cirq():
    import cirq

    q = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.H(q[0]))
    assert_unitary(circuit, HADAMARD)
```

`assert_unitary` is global-phase safe: a circuit that produces
$e^{i\theta} H$ still passes because global phase is not physically
observable.

---

## First PennyLane test

```python
# test_pennylane_basic.py
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
    expected = np.array([0, -1j])
    assert_state_fidelity_above(state, expected, threshold=0.99)
```

Install PennyLane support:

```bash
pip install "pytest-quantum[pennylane]"
```

---

## First Cirq test

```python
# test_cirq_basic.py
import math
import numpy as np
from pytest_quantum import assert_unitary


def test_hadamard_cirq(cirq_simulator):
    import cirq

    q = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.H(q[0]))
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    assert_unitary(circuit, H)
```

Install Cirq support:

```bash
pip install "pytest-quantum[cirq]"
```

---

## Using markers

pytest-quantum ships four markers. Declare them in your test functions or
at module level.

### `@pytest.mark.quantum`

Marks a test as quantum-related. Has no mechanical effect — just a
convention for selecting or excluding quantum tests:

```bash
pytest -m quantum       # run only quantum tests
pytest -m "not quantum" # skip all quantum tests
```

### `@pytest.mark.quantum_slow`

Tests that require a large shot count and take more than a few seconds.
**Skipped by default** — run explicitly with `--quantum-slow`:

```python
@pytest.mark.quantum_slow
def test_precision_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_measurement_distribution, min_shots

    shots = min_shots(epsilon=0.01)   # 7 299 shots — very precise
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=shots
    ).result().get_counts()
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
```

```bash
pytest --quantum-slow   # now quantum_slow tests are included
```

### `@pytest.mark.shots(n)`

Override the default shot count for a single test without modifying the
function body — useful in CI to run a cheaper variant:

```python
@pytest.mark.shots(500)
def test_bell_fast(aer_simulator):
    ...
```

### `@pytest.mark.significance(p)`

Override the statistical significance threshold for one test:

```python
@pytest.mark.significance(0.001)   # very lenient — expects noisy results
def test_noisy_circuit(aer_noise_simulator):
    ...
```

---

## Registering markers in pyproject.toml

To suppress `PytestUnknownMarkWarning`, register markers in your `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "quantum: mark test as quantum",
    "quantum_slow: mark slow quantum tests (run with --quantum-slow)",
    "quantum_snapshot: mark snapshot tests (update with --quantum-update-snapshots)",
]
```

---

## How many shots?

Use the `min_shots` and `recommended_shots` utilities to calculate the right shot count:

```python
from pytest_quantum import min_shots, recommended_shots

# Minimum shots to detect a 5% TVD error with 80% statistical power
n = min_shots(epsilon=0.05)      # → 293 shots

# Minimum shots to detect a 1% TVD error
n = min_shots(epsilon=0.01)      # → 7299 shots

# Shots driven by your specific expected distribution
# (ensures at least 5 expected counts per bucket for chi-square validity)
n = recommended_shots({"00": 0.499, "01": 0.001, "10": 0.001, "11": 0.499})
# → 5000 (driven by the 0.1% outcomes)
```

Rule of thumb:

| Sensitivity needed | Shots |
|---|---|
| Rough check (10% error detectable) | ~74 |
| Standard CI (5% error detectable) | ~293 |
| Thorough (2% error detectable) | ~1832 |
| High precision (1% error detectable) | ~7299 |

---

## CLI options

| Option | Default | Description |
|---|---|---|
| `--quantum-slow` | off | Include `quantum_slow`-marked tests |
| `--quantum-shots N` | `None` | Override shot count globally |
| `--quantum-significance P` | `None` | Override p-value threshold globally |
| `--quantum-update-snapshots` | off | Regenerate snapshot files |

```bash
# CI: run everything including slow tests, with 1 000 shots
pytest --quantum-slow --quantum-shots 1000

# Local: quick smoke test with lenient statistics
pytest --quantum-shots 100 --quantum-significance 0.001

# Update all snapshots after intentional circuit change
pytest --quantum-update-snapshots
```

---

## Cross-framework equivalence

Use `assert_circuits_equivalent` to verify that the same algorithm
implemented in two different frameworks produces the same unitary. This is
invaluable when porting code between SDKs.

```python
# test_cross_framework.py
import pytest
from pytest_quantum import assert_circuits_equivalent

@pytest.mark.quantum
def test_cnot_qiskit_vs_cirq():
    """Verify Qiskit and Cirq CNOT circuits are unitarily equivalent."""
    import cirq
    from qiskit import QuantumCircuit

    # Qiskit CNOT
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    # Cirq CNOT
    q0, q1 = cirq.LineQubit.range(2)
    cc = cirq.Circuit(cirq.CNOT(q0, q1))

    assert_circuits_equivalent(qc, cc)

@pytest.mark.quantum
def test_bell_qiskit_vs_braket():
    """Verify Bell-state preparation is equivalent in Qiskit and Braket."""
    from braket.circuits import Circuit as BraketCircuit
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    bc = BraketCircuit().h(0).cnot(0, 1)

    assert_circuits_equivalent(qc, bc)
```

When both circuits are Qiskit and `mqt.qcec` is installed, the comparison
uses exact formal verification via decision diagrams — faster and more
scalable than matrix multiplication for large circuits:

```bash
pip install "pytest-quantum[qcec]"
```

---

## Next steps

- Read [Concepts](concepts.md) to understand shot noise, global phase, and qubit ordering.
- Read the [Assertions Reference](assertions.md) for all 38 assertions with examples.
- See [Fixtures](fixtures.md) for all simulator fixtures and their usage.
- Browse the [Cookbook](cookbook.md) for complete working recipes.
- See [Statistics Utilities](stats.md) for shot-count calculators and statistical primitives.
