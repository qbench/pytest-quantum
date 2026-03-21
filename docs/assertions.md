# Assertions Reference

pytest-quantum provides nine assertion functions covering three levels of
quantum correctness verification: unitary equivalence, statevector fidelity,
and shot-based distribution testing.

Import any assertion directly from the top-level package:

```python
from pytest_quantum import (
    assert_unitary,
    assert_circuits_equivalent,
    assert_state_fidelity_above,
    assert_states_close,
    assert_measurement_distribution,
    assert_counts_close,
    assert_circuit_depth,
    assert_circuit_width,
    assert_gate_count,
)
```

---

## Unitary assertions

These functions compare circuits at the matrix level — the most rigorous
check for deterministic quantum operations.

### `assert_unitary`

```python
assert_unitary(
    circuit,
    expected,
    *,
    atol=1e-6,
    allow_global_phase=True,
)
```

Asserts that `circuit` implements the `expected` unitary matrix.

The circuit is simulated to extract its full $2^n \times 2^n$ unitary matrix,
which is then compared element-wise to `expected`. Works with Qiskit, Cirq,
Amazon Braket, and PennyLane.

**Parameters**

: `circuit` — Any supported quantum circuit.
: `expected` — Target unitary as a `numpy` array of shape `(2**n, 2**n)`.
: `atol` — Absolute tolerance per element (default `1e-6`).
: `allow_global_phase` — If `True` (default), circuits that differ only by
  $e^{i\theta}$ are considered equivalent. This is physically correct because
  global phase is not observable.

**Raises**

: `AssertionError` — Unitary does not match.
: `TypeError` — Circuit type not recognised.

**Example — Hadamard gate**

```python
import numpy as np
from pytest_quantum import assert_unitary

HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def test_h_gate():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    assert_unitary(qc, HADAMARD)
```

**Example — CNOT gate**

```python
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

def test_cnot():
    import cirq
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))
    assert_unitary(circuit, CNOT)
```

**Example — strict phase check**

```python
def test_z_gate_exact_phase():
    from qiskit import QuantumCircuit

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    qc = QuantumCircuit(1)
    qc.z(0)
    # Require exact match — no global-phase tolerance
    assert_unitary(qc, Z, allow_global_phase=False)
```

---

### `assert_circuits_equivalent`

```python
assert_circuits_equivalent(
    circuit_a,
    circuit_b,
    *,
    atol=1e-6,
)
```

Asserts that two circuits implement the same unitary, up to global phase.
Works across frameworks — you can compare Qiskit against Cirq, Braket, or
PennyLane.

**Fast path for Qiskit pairs:** When both circuits are Qiskit and
`mqt.qcec` is installed (`pip install "pytest-quantum[qcec]"`), formal
equivalence checking via ZX-calculus and decision diagrams is used. This
is significantly faster than matrix multiplication for large circuits.

**Parameters**

: `circuit_a` — First circuit (any supported framework).
: `circuit_b` — Second circuit (any supported framework).
: `atol` — Absolute tolerance for the numpy matrix comparison (default
  `1e-6`). Not used when mqt.qcec handles the comparison.

**Raises**

: `AssertionError` — Circuits are not equivalent.
: `TypeError` — One or both circuit types not recognised.

**Example — cross-framework CNOT**

```python
from pytest_quantum import assert_circuits_equivalent

def test_cnot_cross_framework():
    import cirq
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    q0, q1 = cirq.LineQubit.range(2)
    cc = cirq.Circuit(cirq.CNOT(q0, q1))

    assert_circuits_equivalent(qc, cc)
```

**Example — verifying compiler optimisation**

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from pytest_quantum import assert_circuits_equivalent

def test_transpilation_preserves_unitary():
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.cx(1, 2)

    sim = AerSimulator()
    qc_opt = transpile(qc, sim, optimization_level=3)

    # The optimised circuit must be unitarily equivalent to the original
    assert_circuits_equivalent(qc, qc_opt)
```

---

## Statevector assertions

Use these when you have access to the full statevector from a simulator —
more informative than shot-based testing because every amplitude is checked.

### `assert_state_fidelity_above`

```python
assert_state_fidelity_above(
    actual,
    target,
    threshold=0.99,
)
```

Asserts that the fidelity $F = |\langle\text{actual}|\text{target}\rangle|^2$
is at or above `threshold`.

Fidelity is `1.0` for identical states (up to global phase) and `0.0` for
orthogonal states. This is the primary assertion for **Graphix MBQC tests**
where the circuit does not have a fixed unitary representation, and for
**noisy simulation** where you expect slight deviation.

**Parameters**

: `actual` — Simulated output statevector (any shape; flattened internally).
: `target` — Ideal target statevector (same number of elements).
: `threshold` — Minimum acceptable fidelity (default `0.99`).

**Raises**

: `AssertionError` — Fidelity is below `threshold`, with the actual value
  and shortfall reported.
: `ValueError` — Arrays have incompatible sizes.

**Example — Graphix Bell state**

```python
import numpy as np
from pytest_quantum import assert_state_fidelity_above

BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

def test_bell_graphix(graphix_backend):
    from graphix.transpiler import Circuit

    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    pattern = circuit.transpile().pattern
    output = graphix_backend.run_pattern(pattern)

    assert_state_fidelity_above(output, BELL, threshold=0.999)
```

**Example — noisy simulation with relaxed threshold**

```python
def test_h_noisy(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector

    sim = aer_noise_simulator(error_rate=0.02)

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.save_statevector()

    result = sim.run(transpile(qc, sim)).result()
    sv = result.get_statevector().data

    PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
    # Allow for 2% noise degradation
    assert_state_fidelity_above(sv, PLUS, threshold=0.95)
```

---

### `assert_states_close`

```python
assert_states_close(
    actual,
    target,
    *,
    atol=1e-6,
)
```

Asserts that two statevectors are element-wise close, up to global phase.
Stricter than `assert_state_fidelity_above` — use for **exact
simulator-to-simulator comparisons** where you want bit-for-bit agreement.

Both arrays are flattened, normalised, and global-phase aligned before
comparison.

**Parameters**

: `actual` — Simulated statevector (any shape; flattened and normalised).
: `target` — Ideal statevector (same number of elements).
: `atol` — Absolute tolerance per element (default `1e-6`).

**Raises**

: `AssertionError` — Any element differs by more than `atol` after
  global-phase alignment.

**Example — plus state from Aer**

```python
import numpy as np
from pytest_quantum import assert_states_close

PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)

def test_plus_state(aer_statevector_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.save_statevector()

    qc_t = transpile(qc, aer_statevector_simulator)
    sv = aer_statevector_simulator.run(qc_t).result().get_statevector()

    assert_states_close(sv.data, PLUS)
```

---

## Distribution assertions

Use these for shot-based tests where you run a circuit many times and
check the resulting probability distribution.

### `assert_measurement_distribution`

```python
assert_measurement_distribution(
    counts,
    expected_probs,
    *,
    significance=0.05,
    min_expected_per_bucket=5,
)
```

Asserts that measured counts match the expected probability distribution
using a **chi-square goodness-of-fit test**. The test fails only when the
deviation is statistically significant (`p < significance`), so random
fluctuations from finite sampling do not cause spurious test failures.

**Parameters**

: `counts` — Measured counts dict, e.g. `{"00": 489, "11": 511}`.
: `expected_probs` — Expected probability dict, e.g. `{"00": 0.5, "11":
  0.5}`. Must sum to 1.0. Outcomes absent from this dict are treated as
  having zero expected probability.
: `significance` — P-value threshold for rejection (default `0.05`).
: `min_expected_per_bucket` — Chi-square requires expected count ≥ 5 per
  non-zero cell for valid results. A `UserWarning` is raised (but the test
  does not fail) if any bucket has fewer expected counts; use more shots or
  merge low-probability outcomes.

**Raises**

: `AssertionError` — `p_value < significance`, with a per-state table of
  observed vs expected probabilities.
: `ValueError` — `expected_probs` does not sum to 1.0, or `counts` is empty.

**Example — Bell state**

```python
from pytest_quantum import assert_measurement_distribution, min_shots

def test_bell_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    shots = min_shots(epsilon=0.05)   # 293 shots for 5% sensitivity
    qc_t = transpile(qc, aer_simulator)
    counts = aer_simulator.run(qc_t, shots=shots).result().get_counts()

    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
    )
```

**Example — uniform distribution over 3 qubits**

```python
def test_uniform_3qubit(aer_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_measurement_distribution

    qc = QuantumCircuit(3)
    qc.h([0, 1, 2])
    qc.measure_all()

    uniform = {f"{i:03b}": 1/8 for i in range(8)}
    shots = 2000  # 250 expected per bucket — well above min
    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=shots
    ).result().get_counts()

    assert_measurement_distribution(counts, uniform, significance=0.05)
```

**Example — noisy circuit with lenient threshold**

```python
def test_noisy_bell(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_measurement_distribution

    sim = aer_noise_simulator(error_rate=0.01)
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()

    # With 1% noise, use p=0.001 so we don't fail on small realistic deviations
    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
        significance=0.001,
    )
```

---

### `assert_counts_close`

```python
assert_counts_close(
    counts_a,
    counts_b,
    *,
    max_tvd=0.05,
)
```

Asserts that two count dictionaries produce statistically close distributions,
measured by Total Variation Distance (TVD).

TVD of `0` means identical distributions; `1` means disjoint support.
Use this to compare two backends or to verify that transpilation has not
changed a circuit's output distribution.

**Parameters**

: `counts_a` — First counts dict.
: `counts_b` — Second counts dict.
: `max_tvd` — Maximum acceptable TVD (default `0.05`).

**Raises**

: `AssertionError` — TVD exceeds `max_tvd`, with the actual TVD and excess
  reported.

**Example — comparing two backends**

```python
from pytest_quantum import assert_counts_close

def test_aer_vs_statevector_sampling(aer_simulator, aer_statevector_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()

    counts_shot = aer_simulator.run(
        transpile(qc, aer_simulator), shots=2000
    ).result().get_counts()

    counts_sv = aer_statevector_simulator.run(
        transpile(qc, aer_statevector_simulator), shots=2000
    ).result().get_counts()

    assert_counts_close(counts_shot, counts_sv, max_tvd=0.05)
```

**Example — verifying transpilation**

```python
def test_transpile_preserves_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_counts_close

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()

    qc_opt = transpile(qc, aer_simulator, optimization_level=3)

    counts_orig = aer_simulator.run(qc, shots=2000).result().get_counts()
    counts_opt  = aer_simulator.run(qc_opt, shots=2000).result().get_counts()

    assert_counts_close(counts_orig, counts_opt, max_tvd=0.05)
```

---

## Circuit structure assertions

These assertions check static properties of a circuit without executing it.
Useful for catching regressions in compiler output or ensuring circuits
meet hardware constraints.

### `assert_circuit_depth`

```python
assert_circuit_depth(
    circuit,
    *,
    max_depth=None,
    min_depth=None,
)
```

Asserts that a circuit's depth is within the specified bounds. At least one
of `max_depth` or `min_depth` must be provided. Supported frameworks: Qiskit,
Cirq, Amazon Braket.

**Parameters**

: `circuit` — A quantum circuit from a supported framework.
: `max_depth` — If given, the circuit depth must be ≤ this value.
: `min_depth` — If given, the circuit depth must be ≥ this value.

**Raises**

: `AssertionError` — Depth is outside the specified bounds.
: `TypeError` — Circuit type not supported.
: `ValueError` — Neither `max_depth` nor `min_depth` was provided.

**Example — Qiskit Bell circuit**

```python
from pytest_quantum import assert_circuit_depth

def test_bell_depth():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    assert_circuit_depth(qc, max_depth=3)
```

**Example — depth bounds after transpilation**

```python
def test_compiled_depth_regression(aer_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_circuit_depth

    qc = QuantumCircuit(5)
    for i in range(4):
        qc.cx(i, i + 1)

    qc_t = transpile(qc, aer_simulator, optimization_level=3)

    # After optimisation, depth should not blow up
    assert_circuit_depth(qc_t, max_depth=10)
    # But it should still be a non-trivial circuit
    assert_circuit_depth(qc_t, min_depth=2)
```

---

### `assert_circuit_width`

```python
assert_circuit_width(
    circuit,
    expected_qubits,
)
```

Asserts that a circuit acts on exactly `expected_qubits` qubits. Supported
frameworks: Qiskit, Cirq, Amazon Braket, PennyLane.

**Parameters**

: `circuit` — A quantum circuit from a supported framework.
: `expected_qubits` — Expected number of qubits.

**Raises**

: `AssertionError` — Qubit count does not match.
: `TypeError` — Circuit type not supported.

**Example**

```python
from pytest_quantum import assert_circuit_width

def test_ghz_width():
    from qiskit import QuantumCircuit

    n = 4
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    assert_circuit_width(qc, expected_qubits=4)
```

---

### `assert_gate_count`

```python
assert_gate_count(
    circuit,
    gate_name,
    expected,
)
```

Asserts that a circuit contains exactly `expected` occurrences of `gate_name`.
Supported frameworks: Qiskit, Cirq, PennyLane.

Gate name matching:

- **Qiskit** — case-insensitive string, matches `circuit.count_ops()` keys
  (e.g. `"cx"`, `"h"`, `"t"`).
- **Cirq** — case-insensitive match against `str(op.gate)` (e.g. `"CNOT"`,
  `"H"`, `"CZ"`).
- **PennyLane** — case-insensitive match against the operation class name
  (e.g. `"CNOT"`, `"Hadamard"`, `"T"`). The QNode must have been called at
  least once before `assert_gate_count` can inspect its tape.

**Parameters**

: `circuit` — A quantum circuit from a supported framework.
: `gate_name` — Gate name string (see matching rules above).
: `expected` — Expected integer count.

**Raises**

: `AssertionError` — Actual count differs from `expected`.
: `NotImplementedError` — Framework not yet supported.

**Example — T-count after synthesis**

```python
from pytest_quantum import assert_gate_count

def test_t_count():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.t(0); qc.t(1); qc.cx(0, 1)

    assert_gate_count(qc, "t", 2)
    assert_gate_count(qc, "cx", 1)
```

**Example — Cirq gate counts**

```python
def test_cirq_gate_counts():
    import cirq
    from pytest_quantum import assert_gate_count

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.H(q1)])

    assert_gate_count(circuit, "H", 2)
    assert_gate_count(circuit, "CNOT", 1)
```

**Example — PennyLane QNode (must call first)**

```python
def test_pl_gate_count(pennylane_device):
    import pennylane as qml
    from pytest_quantum import assert_gate_count

    dev = pennylane_device(wires=2)

    @qml.qnode(dev)
    def bell_circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.state()

    bell_circuit()   # must call before inspecting tape

    assert_gate_count(bell_circuit, "Hadamard", 1)
    assert_gate_count(bell_circuit, "CNOT", 1)
```

---

## Decision guide — which assertion to use?

| Scenario | Recommended assertion |
|---|---|
| Deterministic gate / unitary circuit | `assert_unitary` |
| Same algorithm in two frameworks | `assert_circuits_equivalent` |
| MBQC / Graphix output state | `assert_state_fidelity_above` |
| Exact simulator-to-simulator state check | `assert_states_close` |
| Shot-based test vs known distribution | `assert_measurement_distribution` |
| Compare two backends' shot outputs | `assert_counts_close` |
| Compiler depth regression test | `assert_circuit_depth` |
| Qubit count constraint | `assert_circuit_width` |
| Gate type count (e.g. T-count) | `assert_gate_count` |

**Unitary vs statevector vs distribution**

Use `assert_unitary` when the circuit is fully deterministic and you can
compute the exact matrix. It is the strongest check.

Use `assert_state_fidelity_above` or `assert_states_close` when you have
access to a statevector but not the full unitary (e.g., MBQC, mid-circuit
measurements, parametric circuits at a fixed point).

Use `assert_measurement_distribution` or `assert_counts_close` for
shot-based tests on real hardware or noisy simulators where statevector
access is unavailable.
