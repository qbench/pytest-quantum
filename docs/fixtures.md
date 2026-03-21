# Fixtures Reference

pytest-quantum provides session-scoped simulator fixtures for every supported
framework. Fixtures are discovered automatically when the plugin is installed —
no import needed.

Declare any fixture as a test parameter and pytest injects it:

```python
def test_something(aer_simulator, graphix_backend):
    ...
```

If the required SDK is not installed, the test is **automatically skipped**
with a helpful install message rather than erroring out.

---

## Quick reference

| Fixture | Framework | Return type | Scope | Extra required |
|---|---|---|---|---|
| `aer_simulator` | Qiskit / Aer | `AerSimulator` (shot mode) | session | `[qiskit]` |
| `aer_statevector_simulator` | Qiskit / Aer | `AerSimulator` (statevector mode) | session | `[qiskit]` |
| `aer_noise_simulator` | Qiskit / Aer | `Callable[[float], AerSimulator]` | function | `[qiskit]` |
| `cirq_simulator` | Cirq | `cirq.Simulator` | session | `[cirq]` |
| `braket_simulator` | Amazon Braket | `LocalSimulator` | session | `[braket]` |
| `graphix_backend` | Graphix | `_GraphixBackend` | session | `[graphix]` |
| `pennylane_device` | PennyLane | `Callable[[int, int|None], Device]` | session | `[pennylane]` |

**Session scope** means the backend is initialised **once per test run** and
reused across all tests that request it. This avoids repeated startup costs
(JIT compilation, process spawning) and can cut suite runtime significantly
for large test suites.

`aer_noise_simulator` uses function scope because the noise model is
parameterised per-test.

---

## Qiskit / Aer fixtures

### `aer_simulator`

A session-scoped `AerSimulator` configured for shot-based sampling. Use this
for tests that measure a circuit and compare the resulting count distribution.

```python
def test_bell_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_measurement_distribution, min_shots

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    shots = min_shots(epsilon=0.05)
    qc_t = transpile(qc, aer_simulator)
    counts = aer_simulator.run(qc_t, shots=shots).result().get_counts()

    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
    )
```

---

### `aer_statevector_simulator`

A session-scoped `AerSimulator` running in statevector mode (`method="statevector"`).
Use this when you need the full quantum state vector rather than sampled counts.

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
    result = aer_statevector_simulator.run(qc_t).result()
    sv = result.get_statevector().data

    assert_states_close(sv, PLUS)
```

---

### `aer_noise_simulator`

A function-scoped fixture that returns a **factory function**
`make_simulator(error_rate)`. Call the factory with the single-qubit
depolarizing error probability you want to model.

The noise model applies:

- Single-qubit depolarizing error at `error_rate` on all single-qubit gates
  (`h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `u`).
- Two-qubit depolarizing error at `min(error_rate * 10, 1.0)` on all
  two-qubit gates (`cx`, `cz`, `cy`, `swap`), reflecting that two-qubit
  gates are typically ~10× noisier on real hardware.

```python
def test_noisy_bell(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_measurement_distribution

    # Create a simulator with 1% single-qubit error rate
    sim = aer_noise_simulator(error_rate=0.01)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()

    # With 1% noise the Bell distribution is slightly perturbed —
    # use a lenient significance threshold
    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
        significance=0.001,
    )
```

**Different error rates in one test:**

```python
def test_error_rate_sweep(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import tvd_from_counts

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()

    for rate in [0.0, 0.01, 0.05, 0.10]:
        sim = aer_noise_simulator(error_rate=rate)
        counts = sim.run(transpile(qc, sim), shots=1000).result().get_counts()
        # Noisier simulators will deviate more from ideal {0: 0.5, 1: 0.5}
        distance = tvd_from_counts(counts, {"0": 500, "1": 500})
        assert distance < rate + 0.15   # rough bound
```

---

## Cirq

### `cirq_simulator`

A session-scoped `cirq.Simulator`. Supports exact statevector simulation and
density matrix simulation.

```python
def test_cirq_hadamard(cirq_simulator):
    import cirq
    import numpy as np

    q = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.H(q[0]))

    result = cirq_simulator.simulate(circuit)
    sv = result.final_state_vector

    PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(np.abs(sv), np.abs(PLUS), atol=1e-6)
```

**Sampling with Cirq:**

```python
def test_cirq_bell_sampling(cirq_simulator):
    import cirq
    from pytest_quantum import assert_counts_close

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="result"),
    ])

    result = cirq_simulator.run(circuit, repetitions=1000)
    # cirq returns a dict of {key: np.ndarray of shape (reps, n_qubits)}
    raw = result.measurements["result"]
    # Convert to bitstring counts for assert_counts_close
    counts = {}
    for row in raw:
        key = "".join(str(b) for b in row)
        counts[key] = counts.get(key, 0) + 1

    assert_counts_close(counts, {"00": 500, "11": 500}, max_tvd=0.1)
```

---

## Amazon Braket

### `braket_simulator`

A session-scoped `LocalSimulator` from the Amazon Braket SDK. Runs entirely
locally — no AWS account needed.

```python
def test_braket_bell(braket_simulator):
    from braket.circuits import Circuit
    from pytest_quantum import assert_measurement_distribution

    circ = Circuit().h(0).cnot(0, 1)
    circ.measure_all()

    task = braket_simulator.run(circ, shots=1000)
    counts_raw = task.result().measurement_counts
    # Braket uses integer-key tuples — convert to string bitstrings
    counts = {"".join(str(b) for b in k): v for k, v in counts_raw.items()}

    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
    )
```

**Circuit structure check with Braket:**

```python
from pytest_quantum import assert_circuit_depth, assert_circuit_width

def test_braket_circuit_structure():
    from braket.circuits import Circuit

    circ = Circuit().h(0).cnot(0, 1).h(1)

    assert_circuit_width(circ, expected_qubits=2)
    assert_circuit_depth(circ, max_depth=3)
```

---

## Graphix

### `graphix_backend`

A session-scoped backend for running Graphix measurement-based quantum
computation (MBQC) patterns. Returns a `_GraphixBackend` instance with a
single method:

```
graphix_backend.run_pattern(pattern, input_state=None)
```

This compiles and runs the `Pattern` using `graphix.simulator.PatternSimulator`
in statevector mode, then returns the output state as a 1-D complex numpy array
of shape `(2**n_output_qubits,)`.

**Parameters of `run_pattern`:**

: `pattern` — A `graphix.pattern.Pattern` instance.
: `input_state` — Optional `graphix.states.BasicStates` value. If `None`,
  the default input state is used.

```python
import numpy as np
from pytest_quantum import assert_state_fidelity_above

BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

def test_graphix_bell(graphix_backend):
    from graphix.transpiler import Circuit

    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)

    pattern = circuit.transpile().pattern
    output_state = graphix_backend.run_pattern(pattern)

    assert_state_fidelity_above(output_state, BELL, threshold=0.999)
```

**With a custom input state:**

```python
def test_graphix_custom_input(graphix_backend):
    from graphix.states import BasicStates
    from graphix.transpiler import Circuit
    import numpy as np

    # Run a single-qubit H on a |1⟩ input
    circuit = Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern

    MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
    output = graphix_backend.run_pattern(pattern, input_state=BasicStates.ONE)
    assert_state_fidelity_above(output, MINUS, threshold=0.999)
```

:::{note}
`graphix_backend` is the primary fixture for MBQC tests. Use
`assert_state_fidelity_above` rather than `assert_unitary` because MBQC
patterns produce a specific output state for a given input — they do not
have a universal unitary matrix.
:::

---

## PennyLane

### `pennylane_device`

A session-scoped fixture that returns a **factory function**
`make_device(wires, shots=None)`. Call it to create a `pennylane.device`
with the right number of wires for your circuit.

Using a factory instead of a single device allows different tests to use
different wire counts while still benefiting from a single fixture scope.

```python
def test_pennylane_bell(pennylane_device):
    import pennylane as qml
    import numpy as np

    dev = pennylane_device(wires=2)

    @qml.qnode(dev)
    def bell_circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.state()

    state = bell_circuit()

    BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    from pytest_quantum import assert_state_fidelity_above
    assert_state_fidelity_above(state, BELL)
```

**Shot-based sampling with PennyLane:**

```python
def test_pennylane_sampling(pennylane_device):
    import pennylane as qml
    import numpy as np

    dev = pennylane_device(wires=2, shots=1000)

    @qml.qnode(dev)
    def bell_circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.counts()

    raw_counts = bell_circuit()
    # PennyLane returns tensor keys — convert to string counts
    counts = {str(k): int(v) for k, v in raw_counts.items()}

    from pytest_quantum import assert_measurement_distribution
    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
    )
```

**Gate count check with PennyLane:**

```python
def test_pennylane_gate_count(pennylane_device):
    import pennylane as qml
    from pytest_quantum import assert_gate_count

    dev = pennylane_device(wires=3)

    @qml.qnode(dev)
    def ghz_circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.CNOT([1, 2])
        return qml.state()

    ghz_circuit()   # must be called first to populate the tape

    assert_gate_count(ghz_circuit, "Hadamard", 1)
    assert_gate_count(ghz_circuit, "CNOT", 2)
```

---

## Session scope and performance

All fixtures except `aer_noise_simulator` use `scope="session"`. This means:

- Each simulator is constructed **once** per `pytest` invocation.
- The same instance is shared across every test that requests it.
- Startup costs (Aer JIT, Braket process init, Graphix import) are paid once.

For a suite of 100 tests each using `aer_simulator`, this typically saves
several seconds of redundant initialisation.

If you need an isolated simulator (e.g., to change configuration mid-suite),
create it directly inside your test rather than via the fixture:

```python
def test_custom_backend():
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="matrix_product_state")
    ...
```

---

## Missing SDK behaviour

If the required package is not installed, the test is skipped with a message:

```
SKIPPED [1] conftest.py: 'qiskit_aer' is not installed.
Install it with: pip install pytest-quantum[qiskit]
```

This means you can have a single test suite that covers multiple frameworks
and run it in environments where only some are installed. CI jobs can be
defined per-framework by controlling which extras are installed.
