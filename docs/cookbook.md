# Cookbook

Self-contained, runnable recipes for common quantum testing patterns. Each recipe is
a complete test file you can copy directly into your project.

---

## Recipe 1: Bell State Circuit

**Context:** The Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 is the simplest entangled state
and a good first test for any quantum framework. This recipe tests measurement
distribution, circuit structure, and the exact unitary.

```python
# test_bell.py
import math
import numpy as np
import pytest
from pytest_quantum import (
    assert_circuit_depth,
    assert_circuit_width,
    assert_measurement_distribution,
    assert_unitary,
)

BELL_UNITARY = np.array([
    [1,  0,  0,  1],
    [0,  1,  1,  0],
    [0,  1, -1,  0],
    [1,  0,  0, -1],
], dtype=complex) / math.sqrt(2)


def test_bell_measurement_distribution(aer_simulator):
    """Bell state should measure 50/50."""
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()

    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=2000
    ).result().get_counts()

    assert_measurement_distribution(counts, expected_probs={"00": 0.5, "11": 0.5})


def test_bell_circuit_structure():
    """Bell circuit should have 2 qubits and depth ≤ 3."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)

    assert_circuit_width(qc, expected_qubits=2)
    assert_circuit_depth(qc, max_depth=3)


def test_bell_unitary():
    """Bell circuit should implement the correct unitary."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)

    assert_unitary(qc, BELL_UNITARY)
```

**Expected output:**

```
test_bell.py::test_bell_measurement_distribution PASSED
test_bell.py::test_bell_circuit_structure PASSED
test_bell.py::test_bell_unitary PASSED
```

**Variations:** Use `assert_entanglement_entropy_below` to verify the Bell state has
exactly 1 ebit of entanglement; use `assert_partial_trace_close` to check the reduced
density matrix is `I/2`.

---

## Recipe 2: Testing a VQE

**Context:** Variational Quantum Eigensolver (VQE) is an optimization algorithm that
finds the ground state energy of a Hamiltonian. The result is stochastic, so use
`assert_ground_state_energy_close` with an appropriate tolerance.

```python
# test_vqe.py
import pytest
from pytest_quantum import assert_ground_state_energy_close


@pytest.mark.quantum_slow
def test_vqe_ground_state(pennylane_device):
    import pennylane as qml
    import numpy as np
    from scipy.optimize import minimize

    dev = pennylane_device(wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    result = minimize(lambda t: circuit(t[0]), x0=[0.5], method="COBYLA")
    # Ground state of Z is |1>, energy = -1
    assert_ground_state_energy_close(result.fun, expected_energy=-1.0, atol=0.05)
```

**Run with:**

```bash
pip install "pytest-quantum[pennylane]" scipy
pytest test_vqe.py --quantum-slow -v
```

---

## Recipe 3: QFT Round-Trip

**Context:** The Quantum Fourier Transform (QFT) followed by its inverse should return
to the original state. This is a stringent test of the QFT implementation.

```python
# test_qft.py
import numpy as np
from pytest_quantum import assert_states_close


def test_qft_roundtrip(aer_statevector_simulator):
    """QFT then inverse QFT = identity."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT

    n = 3
    qc = QuantumCircuit(n)
    qc.x(0)  # Start in |001⟩
    qc.append(QFT(n), range(n))
    qc.append(QFT(n).inverse(), range(n))
    qc.save_statevector()

    sv = aer_statevector_simulator.run(
        transpile(qc, aer_statevector_simulator)
    ).result().get_statevector().data

    # Qiskit LSB ordering: qubit 0 = 1, others = 0 → index 1
    expected = np.zeros(2**n, dtype=complex)
    expected[1] = 1.0
    assert_states_close(sv, expected)
```

---

## Recipe 4: Testing Noise Tolerance

**Context:** Real quantum hardware has noise. This recipe tests that a circuit behaves
reasonably even with depolarizing noise.

```python
# test_noise_tolerance.py
import pytest
from pytest_quantum import assert_measurement_distribution


def test_circuit_noise_tolerance(aer_noise_simulator):
    """H gate should produce ~50/50 even with 5% depolarizing noise."""
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(1)
    qc.h(0); qc.measure_all()

    sim = aer_noise_simulator(error_rate=0.05)
    counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()

    # With 5% noise, use lenient significance to avoid false failures
    assert_measurement_distribution(
        counts, {"0": 0.5, "1": 0.5}, significance=0.001
    )
```

---

## Recipe 5: Snapshot Regression Testing

**Context:** After implementing a circuit, save its unitary as a snapshot. Future runs
compare against the snapshot — catch accidental changes without needing to re-derive
the expected matrix.

```python
# test_snapshot.py
from pytest_quantum import assert_unitary_snapshot


def test_grover_oracle_unchanged():
    """Catch any accidental changes to the Grover oracle unitary."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h([0, 1]); qc.cz(0, 1); qc.h([0, 1])

    assert_unitary_snapshot(qc, name="grover_2qubit_oracle")
    # First run: creates .pytest-quantum-snapshots/grover_2qubit_oracle.npy
    # Subsequent runs: compares against saved snapshot
    # Regenerate: pytest --quantum-update-snapshots
```

**Workflow:**

```bash
# 1. First run: creates snapshot
pytest test_snapshot.py

# 2. Later runs: verify unchanged
pytest test_snapshot.py  # passes if circuit is unchanged

# 3. After intentional change: update snapshot
pytest --quantum-update-snapshots test_snapshot.py
```

---

## Recipe 6: Cross-Framework Equivalence

**Context:** When porting a circuit from one framework to another, verify they produce
the same unitary.

```python
# test_cross_framework.py
import numpy as np
import pytest
from pytest_quantum import assert_circuits_equivalent, assert_unitary

H_MATRIX = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

HAS_QISKIT = HAS_CIRQ = HAS_BRAKET = HAS_PENNYLANE = False
try:
    import qiskit; HAS_QISKIT = True
except ImportError: pass
try:
    import cirq; HAS_CIRQ = True
except ImportError: pass
try:
    import braket; HAS_BRAKET = True
except ImportError: pass
try:
    import pennylane; HAS_PENNYLANE = True
except ImportError: pass


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
def test_hadamard_qiskit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1); qc.h(0)
    assert_unitary(qc, H_MATRIX)


@pytest.mark.skipif(not HAS_CIRQ, reason="cirq not installed")
def test_hadamard_cirq():
    import cirq
    q = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.H(q[0]))
    assert_unitary(circuit, H_MATRIX)


@pytest.mark.skipif(not HAS_BRAKET, reason="braket not installed")
def test_hadamard_braket():
    from braket.circuits import Circuit
    circuit = Circuit().h(0)
    assert_unitary(circuit, H_MATRIX)


@pytest.mark.skipif(not (HAS_QISKIT and HAS_CIRQ), reason="qiskit and cirq required")
def test_hadamard_cross_framework():
    from qiskit import QuantumCircuit
    import cirq

    qk = QuantumCircuit(1); qk.h(0)
    q = cirq.LineQubit.range(1)
    cc = cirq.Circuit(cirq.H(q[0]))

    assert_circuits_equivalent(qk, cc)
```

---

## Recipe 7: Noisy Density Matrix Testing

**Context:** For circuits that interact with a noisy environment, test the full density
matrix to verify that the state remains close to ideal.

```python
# test_density_matrix.py
import numpy as np
from pytest_quantum import assert_density_matrix_close, assert_purity_above


def test_bell_density_matrix_purity(aer_noise_simulator):
    """Bell state should remain nearly pure under low noise."""
    from qiskit import QuantumCircuit, transpile

    sim = aer_noise_simulator(error_rate=0.01)  # 1% depolarizing noise

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    qc.save_density_matrix()

    rho = sim.run(transpile(qc, sim)).result().data()["density_matrix"].data

    # With low noise, Bell state should remain nearly pure
    assert_purity_above(rho, min_purity=0.95)

    # Compare to ideal Bell state density matrix
    ideal = np.array([
        [0.5, 0, 0, 0.5],
        [0,   0, 0, 0  ],
        [0,   0, 0, 0  ],
        [0.5, 0, 0, 0.5],
    ], dtype=complex)
    assert_density_matrix_close(rho, ideal, atol=0.05)
```

---

## Recipe 8: Property-Based Testing with Random Generators

**Context:** Instead of testing specific inputs, test mathematical properties that must
hold for *all* valid quantum objects. This catches bugs that deterministic tests miss.

```python
# test_property_based.py
import numpy as np
import pytest
from pytest_quantum import (
    assert_channel_is_cptp,
    assert_hermitian,
    assert_normalized,
    assert_positive_semidefinite,
)
from pytest_quantum.random import (
    depolarizing_kraus,
    random_density_matrix,
    random_kraus_channel,
    random_statevector,
    random_unitary,
)


def test_unitary_roundtrip_property():
    """Property: applying U then U† returns to original state."""
    for seed in range(5):
        U = random_unitary(2, seed=seed)
        sv = random_statevector(2, seed=seed + 100)

        recovered = U.conj().T @ (U @ sv)
        assert np.allclose(recovered, sv, atol=1e-10), \
            f"U†U ≠ I for seed={seed}"


@pytest.mark.parametrize("seed", range(5))
def test_random_statevectors_normalized(seed):
    """Haar-random statevectors must have unit norm."""
    sv = random_statevector(2, seed=seed)
    assert_normalized(sv)


@pytest.mark.parametrize("n_qubits", [1, 2])
def test_random_density_matrices_are_valid(n_qubits):
    """Random density matrices must be Hermitian, PSD, trace=1."""
    rho = random_density_matrix(n_qubits, seed=42)
    assert_hermitian(rho)
    assert_positive_semidefinite(rho)
    assert abs(np.trace(rho) - 1.0) < 1e-10


def test_depolarizing_kraus_is_cptp():
    """Depolarizing channel must be CPTP for any valid error rate."""
    for p in [0.0, 0.01, 0.1, 0.5, 1.0]:
        kraus = depolarizing_kraus(n_qubits=1, error_rate=p)
        assert_channel_is_cptp(kraus)


@pytest.mark.parametrize("seed", range(3))
def test_random_kraus_channel_is_cptp(seed):
    """Random Kraus channels must satisfy CPTP completeness relation."""
    kraus = random_kraus_channel(n_qubits=1, n_kraus=4, seed=seed)
    assert_channel_is_cptp(kraus)
```

---

## Recipe 9: QEC with Stim

**Context:** Stim is a fast stabilizer circuit simulator optimized for QEC. Use
`assert_stim_logical_error_rate_below` to verify your QEC code meets its
error-correcting threshold.

```python
# test_qec.py
import importlib.util
import pytest


@pytest.mark.skipif(
    not importlib.util.find_spec("stim"), reason="stim not installed"
)
def test_repetition_code_logical_error_rate(stim_sampler):
    """3-qubit repetition code logical error rate should be below 5%."""
    import stim
    from pytest_quantum import assert_stim_logical_error_rate_below

    # 3-qubit repetition code with 1% physical error rate
    circuit = stim.Circuit("""
        X_ERROR(0.01) 0 1 2
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-3]
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-2] rec[-1]
    """)
    assert_stim_logical_error_rate_below(circuit, max_error_rate=0.05, shots=5000)
```

**Install Stim:**

```bash
pip install stim
```

---

## Recipe 10: Testing a Clifford Circuit

**Context:** Clifford circuits (using only H, S, CNOT, CZ, and Pauli gates) can be
efficiently simulated classically via the Gottesman-Knill theorem. Use
`assert_circuit_is_clifford` to verify your circuit stays within the Clifford set.

```python
# test_clifford.py
import pytest
from pytest_quantum import assert_circuit_is_clifford


def test_bell_circuit_is_clifford():
    """Bell circuit uses only Clifford gates."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    assert_circuit_is_clifford(qc)  # H and CX are Clifford gates


def test_t_gate_not_clifford():
    """T gate is not Clifford — this should raise AssertionError."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.t(0)
    with pytest.raises(AssertionError, match="non-Clifford"):
        assert_circuit_is_clifford(qc)


def test_syndrome_extraction_is_clifford():
    """QEC syndrome extraction should use only Clifford operations."""
    from qiskit import QuantumCircuit

    # 3-qubit repetition code syndrome extraction
    qc = QuantumCircuit(5)  # 3 data qubits + 2 ancilla
    # Ancilla Hadamards (for X-type stabilizers if needed)
    # Z-type stabilizers: CNOT from data to ancilla
    qc.cx(0, 3); qc.cx(1, 3)
    qc.cx(1, 4); qc.cx(2, 4)
    assert_circuit_is_clifford(qc)
```
