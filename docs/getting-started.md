# Getting Started

> From zero to your first passing quantum test in **under 5 minutes**.

---

## Step 1: Install

```bash
# Just Qiskit? Most common setup:
pip install "pytest-quantum[qiskit]"

# Cirq user?
pip install "pytest-quantum[cirq]"

# Want everything?
pip install "pytest-quantum[all]"
```

That's it. No `conftest.py`. No imports. The plugin registers itself automatically.

---

## Step 2: Write your first test

Create `test_quantum.py`:

```python
from pytest_quantum import assert_measurement_distribution, assert_unitary
import numpy as np


def test_bell_state_distribution(aer_simulator):
    """Fixture injected automatically — no import or setup needed."""
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=2000
    ).result().get_counts()

    # Won't flake on shot noise — uses chi-square statistical test
    assert_measurement_distribution(counts, expected_probs={"00": 0.5, "11": 0.5})


def test_hadamard_gate():
    """No fixture needed for pure unitary checks."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    assert_unitary(qc, H)          # global-phase safe ✓
```

```bash
pytest test_quantum.py -v
```

```
PASSED test_quantum.py::test_bell_state_distribution
PASSED test_quantum.py::test_hadamard_gate
```

---

## Step 3: Understand the core ideas

### Why not just `assert counts["00"] == 512`?

Quantum measurements are **random**. Even a perfect Bell circuit produces
`{"00": 487, "11": 537}` sometimes. Hard-coded counts fail ~5% of runs for
no reason other than statistics.

`assert_measurement_distribution` uses a chi-square goodness-of-fit test.
It only fails when the distribution is **genuinely wrong**, not just noisy.

### Why not just `np.allclose(U1, U2)`?

Two physically identical quantum gates can have different matrices if they
differ by a **global phase**: `U2 = e^(iθ) U1`. Classical array comparison
flags this as a bug. `assert_unitary` handles global phase automatically.

### Why fixtures instead of setup code?

Simulators are expensive to construct. pytest-quantum's session-scoped fixtures
create one simulator per test session, shared across all tests. Your CI runs
faster, your tests are simpler.

---

## Step 4: Try more assertions

```python
from pytest_quantum import (
    assert_state_fidelity_above,
    assert_circuit_depth,
    assert_gate_count,
    assert_circuit_is_clifford,
    assert_unitary_snapshot,
)


def test_rx_gate_state():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(1)
    qc.rx(3.14159, 0)  # RX(π) ≈ -iX

    sv = Statevector(qc).data
    assert_state_fidelity_above(sv, np.array([0, -1j]), threshold=0.999)


def test_circuit_structure():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    assert_circuit_depth(qc, max_depth=3)
    assert_gate_count(qc, "cx", expected=1)
    assert_circuit_is_clifford(qc)           # H + CNOT = Clifford ✓


def test_snapshot_regression(aer_simulator):
    """Run once to create golden file; subsequent runs compare against it."""
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=2000) \
                          .result().get_counts()

    # First run: creates .snapshots/bell_snapshot.json
    # All future runs: compares against it
    assert_unitary_snapshot(qc, "bell_snapshot")
```

---

## Step 5: Real hardware (optional)

```bash
export IBM_QUANTUM_TOKEN=your_token_here
pytest --quantum-real test_hardware.py
```

```python
@pytest.mark.quantum_real
def test_on_real_backend(ibm_backend):
    from pytest_quantum import assert_backend_calibration, assert_circuit_fits_backend
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)

    assert_circuit_fits_backend(qc, ibm_backend)
    assert_backend_calibration(ibm_backend, min_t1_us=20.0)
```

---

## CLI cheatsheet

```bash
pytest                              # normal run (hardware tests skipped)
pytest --quantum-slow               # include slow shot-heavy tests
pytest --quantum-real               # enable real hardware tests
pytest --quantum-shots=4000         # override shot count globally
pytest --quantum-update-snapshots   # regenerate snapshot golden files
pytest --quantum-significance=0.01  # tighten statistical thresholds
```

---

## What's next?

- 📖 [Assertions Reference](assertions.md): every assertion with signature and examples
- 🔧 [Fixtures Reference](fixtures.md): all 25 fixtures documented
- 🧪 [VQE Tutorial](tutorials/01_vqe_end_to_end.md): full VQE testing workflow
- 🔬 [Noise-Aware Testing](tutorials/02_noise_aware_testing.md): noisy simulators + error mitigation
- 💡 [Concepts](concepts.md): the statistics behind shot noise, fidelity, and more
