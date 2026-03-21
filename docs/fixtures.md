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

### Skip message example

```
SKIPPED [1] conftest.py:22: 'qiskit_aer' is not installed.
Install it with: pip install "pytest-quantum[qiskit]"
```

---

## Quick reference

| Fixture | Framework | Return type | Scope | Extra required |
|---|---|---|---|---|
| `aer_simulator` | Qiskit / Aer | `AerSimulator` (shot mode) | session | `[qiskit]` |
| `aer_statevector_simulator` | Qiskit / Aer | `AerSimulator` (statevector mode) | session | `[qiskit]` |
| `aer_noise_simulator` | Qiskit / Aer | `Callable[[float], AerSimulator]` | function | `[qiskit]` |
| `qiskit_sampler` | Qiskit 1.0+ | `StatevectorSampler` | session | `[qiskit]` |
| `qiskit_estimator` | Qiskit 1.0+ | `StatevectorEstimator` | session | `[qiskit]` |
| `cirq_simulator` | Cirq | `cirq.Simulator` | session | `[cirq]` |
| `cirq_sampler` | Cirq | `Callable[[Circuit, int], dict]` | session | `[cirq]` |
| `braket_simulator` | Amazon Braket | `LocalSimulator` | session | `[braket]` |
| `graphix_backend` | Graphix | `_GraphixBackend` | session | `[graphix]` |
| `pennylane_device` | PennyLane | `Callable[[int, int\|None], Device]` | session | `[pennylane]` |
| `pytket_circuit_factory` | Pytket | `pytket.Circuit` class | session | `[pytket]` |
| `stim_sampler` | Stim | `Callable[[stim.Circuit, int], dict]` | session | `stim` |
| `quantum_benchmark` | All | benchmark wrapper | function | `pytest-benchmark` |
| `shot_budget` | All | shot counter object | function | (none) |
| `quantum_shots` | All | `int \| None` | session | (none) |
| `quantum_significance` | All | `float \| None` | session | (none) |

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
    qc.h(0); qc.cx(0, 1); qc.measure_all()

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
    qc.h(0); qc.cx(0, 1); qc.measure_all()

    counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()

    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
        significance=0.001,
    )
```

---

### `qiskit_sampler` *(NEW in v0.2.0)*

A session-scoped `StatevectorSampler` from the Qiskit 1.0+ Primitives API.
Use this when your circuit uses the new Primitives interface instead of
the legacy `backend.run()` pattern.

**Returns:** `qiskit.primitives.StatevectorSampler`

**Auto-skips when:** `qiskit` is not installed.

```python
def test_bell_sampler(qiskit_sampler):
    from qiskit import QuantumCircuit
    from pytest_quantum import assert_sampler_distribution

    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])

    result = qiskit_sampler.run([qc], shots=2000).result()
    assert_sampler_distribution(result[0], {"00": 0.5, "11": 0.5})
```

---

### `qiskit_estimator` *(NEW in v0.2.0)*

A session-scoped `StatevectorEstimator` from the Qiskit 1.0+ Primitives API.
Use this to compute expectation values of observables.

**Returns:** `qiskit.primitives.StatevectorEstimator`

**Auto-skips when:** `qiskit` is not installed.

```python
def test_z_estimator(qiskit_estimator):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from pytest_quantum import assert_estimator_close

    qc = QuantumCircuit(1)
    qc.x(0)  # |1⟩

    op = SparsePauliOp("Z")
    result = qiskit_estimator.run([(qc, op)]).result()
    assert_estimator_close(result[0], expected=-1.0, atol=1e-6)
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

---

### `cirq_sampler` *(NEW in v0.2.0)*

A session-scoped callable that runs Cirq circuits with shot simulation.
Returns a standard count dictionary (string keys → integer counts).

**Returns:** `Callable[[cirq.Circuit, int], dict[str, int]]`

**Auto-skips when:** `cirq` is not installed.

```python
def test_cirq_bell_sampling(cirq_sampler):
    import cirq
    from pytest_quantum import assert_measurement_distribution

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="result"),
    ])

    counts = cirq_sampler(circuit, shots=2000)
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
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
    counts = {"".join(str(b) for b in k): v for k, v in counts_raw.items()}

    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
    )
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

```python
import numpy as np
from pytest_quantum import assert_state_fidelity_above

BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

def test_graphix_bell(graphix_backend):
    from graphix.transpiler import Circuit

    circuit = Circuit(2)
    circuit.h(0); circuit.cnot(0, 1)

    pattern = circuit.transpile().pattern
    output_state = graphix_backend.run_pattern(pattern)

    assert_state_fidelity_above(output_state, BELL, threshold=0.999)
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

:::warning Important: PennyLane requires circuit execution before gate count assertions

When using `assert_gate_count` with PennyLane QNodes, the QNode must have been
executed at least once before calling `assert_gate_count`. The auto-dry-run handles
this automatically, but be aware it may produce unexpected side effects.
:::

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

---

## Pytket

### `pytket_circuit_factory` *(NEW in v0.3.0)*

A session-scoped fixture that returns the `pytket.Circuit` class itself.
This allows your tests to construct Pytket circuits without importing
pytket directly (the fixture auto-skips if pytket is not installed).

**Returns:** `pytket.Circuit` (the class)

**Auto-skips when:** `pytket` is not installed.
Install: `pip install pytket`

```python
def test_pytket_bell(pytket_circuit_factory):
    from pytest_quantum import assert_unitary
    import numpy as np, math

    Circuit = pytket_circuit_factory
    circ = Circuit(2)
    circ.H(0)
    circ.CX(0, 1)

    BELL = np.array([[1,0,0,1],[0,1,1,0],[0,1,-1,0],[1,0,0,-1]], dtype=complex) / math.sqrt(2)
    assert_unitary(circ, BELL)
```

---

## Stim

### `stim_sampler` *(NEW in v0.3.0)*

A session-scoped callable that runs Stim circuits and returns count
dictionaries.

**Returns:** `Callable[[stim.Circuit, int], dict[str, int]]`

**Auto-skips when:** `stim` is not installed.
Install: `pip install stim`

```python
def test_stim_basic(stim_sampler):
    import stim

    circuit = stim.Circuit("""
        H 0
        CNOT 0 1
        M 0 1
    """)
    counts = stim_sampler(circuit, shots=2000)

    from pytest_quantum import assert_measurement_distribution
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
```

---

## Benchmarking and Utilities

### `quantum_benchmark` *(NEW in v0.3.0)*

A function-scoped fixture that wraps `pytest-benchmark` for circuit timing.
Use this to measure and track circuit execution time across releases.

**Returns:** benchmark wrapper object compatible with `pytest-benchmark`

**Auto-skips when:** `pytest-benchmark` is not installed.

```python
def test_circuit_performance(quantum_benchmark):
    from qiskit import QuantumCircuit
    from pytest_quantum import assert_unitary
    import numpy as np

    def run_circuit():
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(5)
        qc.h(0)
        for i in range(4): qc.cx(i, i+1)
        return qc

    result = quantum_benchmark(run_circuit)
```

---

### `shot_budget` *(NEW in v0.3.0)*

A function-scoped shot counting tracker. Use this when you need to track
the total number of shots used across multiple circuit runs in one test.

**Returns:** a `ShotBudget` object with `.add(n)` and `.total` attributes.

```python
def test_shot_budget(shot_budget):
    shot_budget.add(1000)  # first circuit run
    shot_budget.add(500)   # second circuit run
    assert shot_budget.total <= 2000, "Exceeded shot budget"
```

---

### `quantum_shots` *(NEW in v0.2.0)*

A session-scoped fixture that reads the `--quantum-shots` CLI option.
Use this to write tests that respect the global shot override.

**Returns:** `int | None`

```python
def test_adaptive_shots(aer_simulator, quantum_shots):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_measurement_distribution

    shots = quantum_shots or 1000  # use CLI override or default 1000

    qc = QuantumCircuit(1)
    qc.h(0); qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=shots).result().get_counts()

    assert_measurement_distribution(counts, {"0": 0.5, "1": 0.5})
```

---

### `quantum_significance` *(NEW in v0.2.0)*

A session-scoped fixture that reads the `--quantum-significance` CLI option.

**Returns:** `float | None`

```python
def test_adaptive_significance(aer_simulator, quantum_significance):
    from qiskit import QuantumCircuit, transpile
    from pytest_quantum import assert_measurement_distribution

    sig = quantum_significance or 0.05  # use CLI override or default

    qc = QuantumCircuit(1)
    qc.h(0); qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=1000).result().get_counts()
    assert_measurement_distribution(counts, {"0": 0.5, "1": 0.5}, significance=sig)
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
SKIPPED [1] conftest.py:22: 'qiskit_aer' is not installed.
Install it with: pip install "pytest-quantum[qiskit]"
```

This means you can have a single test suite that covers multiple frameworks
and run it in environments where only some are installed. CI jobs can be
defined per-framework by controlling which extras are installed.
