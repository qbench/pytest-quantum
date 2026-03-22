# Fixtures Reference

pytest-quantum provides **25 fixtures** covering every major quantum SDK, real hardware
backends, and testing utilities. All fixtures are **auto-discovered** with no import
needed; just declare them as test parameters.

```python
def test_bell(aer_simulator):    # ← injected automatically
    ...
```

Fixtures that require an uninstalled SDK are **automatically skipped** with a
helpful message pointing to the install command.

---

## Simulators

### `aer_simulator`

Session-scoped `AerSimulator()` for shot-based Qiskit simulation.

```python
def test_bell(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=1024) \
                          .result().get_counts()
    assert "00" in counts
```

### `aer_statevector_simulator`

Session-scoped `AerSimulator(method="statevector")` for exact statevectors with
no shot noise.

```python
def test_exact(aer_statevector_simulator):
    result = aer_statevector_simulator.run(transpile(qc, aer_statevector_simulator)) \
                                      .result()
    sv = result.get_statevector()
```

### `aer_noise_simulator`

Function-scoped factory: `make_simulator(error_rate)`, an AerSimulator with
configurable depolarizing noise on all gates.

```python
def test_noisy(aer_noise_simulator):
    sim = aer_noise_simulator(error_rate=0.01)   # 1% per-gate depolarizing error
    counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5}, significance=0.001)
```

### `cirq_simulator`

Session-scoped `cirq.Simulator()`.

```python
def test_cirq(cirq_simulator):
    import cirq
    q = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.H(q[0]))
    sv = cirq_simulator.simulate(circuit).final_state_vector
    assert sv.shape == (2,)
```

### `cirq_sampler`

Returns a callable `run(circuit, repetitions=1024) → dict[str, int]` that
runs a Cirq circuit with measurements and returns a count dict.

```python
def test_cirq_bell(cirq_sampler):
    import cirq
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q[0]), cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="result"),
    )
    counts = cirq_sampler(circuit, repetitions=2000)
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
```

### `braket_simulator`

Session-scoped Amazon Braket `LocalSimulator()`.

```python
def test_braket(braket_simulator):
    from braket.circuits import Circuit
    circ = Circuit().h(0).cnot(0, 1).probability()
    result = braket_simulator.run(circ, shots=1000).result()
```

### `pennylane_device`

Factory: `make_device(wires, shots=None)`, creates a `default.qubit` device.

```python
def test_pl(pennylane_device):
    import pennylane as qml
    dev = pennylane_device(wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0); qml.CNOT([0, 1])
        return qml.state()

    state = circuit()
    assert state.shape == (4,)
```

### `pytket_circuit_factory`

Returns the `pytket.Circuit` class for building circuits.

```python
def test_pytket(pytket_circuit_factory):
    Circuit = pytket_circuit_factory
    c = Circuit(2)
    c.H(0); c.CX(0, 1)
    assert c.n_qubits == 2
```

### `stim_sampler`

Returns a callable `sample(circuit, shots=1024) → dict[str, int]` for Stim
stabilizer circuits.

```python
def test_stim(stim_sampler):
    import stim
    c = stim.Circuit("H 0\nCNOT 0 1\nM 0 1")
    counts = stim_sampler(c, shots=10000)
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
```

### `graphix_backend`

Backend for running graphix MBQC patterns. Returns a `_GraphixBackend` with
`.run_pattern(pattern)` method.

```python
def test_graphix(graphix_backend):
    from graphix.transpiler import Circuit
    circuit = Circuit(1); circuit.h(0)
    pattern = circuit.transpile().pattern
    sv = graphix_backend.run_pattern(pattern)
    assert sv.shape == (2,)
```

---

## Qiskit Primitives

### `qiskit_sampler`

Session-scoped `StatevectorSampler` (Qiskit 1.0+ primitives API).

```python
def test_sampler(qiskit_sampler):
    from qiskit.circuit import QuantumCircuit
    qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
    result = qiskit_sampler.run([(qc,)]).result()
    assert_sampler_distribution(result, {"00": 0.5, "11": 0.5})
```

### `qiskit_estimator`

Session-scoped `StatevectorEstimator`.

```python
def test_estimator(qiskit_estimator):
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    qc = QuantumCircuit(1)   # |0⟩: ⟨Z⟩ = 1.0
    result = qiskit_estimator.run([(qc, SparsePauliOp("Z"))]).result()
    assert_estimator_close(result, expected=1.0, atol=0.01)
```

---

## Open-System Solvers

### `qutip_solver`

Returns a callable `solve(H, psi0, tlist, c_ops=None) → np.ndarray` that runs
the Lindblad master equation via QuTiP.

```python
def test_qubit_decay(qutip_solver):
    import qutip, numpy as np
    H = qutip.sigmaz() / 2
    psi0 = qutip.basis(2, 0)
    gamma = 0.1
    tlist = np.linspace(0, 10, 100)
    rho_final = qutip_solver(H, psi0, tlist, c_ops=[np.sqrt(gamma)*qutip.sigmam()])
    assert_purity_above(rho_final, min_purity=0.0)
```

### `tequila_backend`

Returns the `tequila` module for quantum chemistry / VQE tests.

```python
def test_tequila_h(tequila_backend):
    tq = tequila_backend
    U = tq.gates.H(target=0)
    result = tq.simulate(U)
    assert abs(result[0])**2 == pytest.approx(0.5, abs=1e-6)
```

---

## Real Hardware

All hardware fixtures require `--quantum-real` to be passed on the CLI
and skip automatically otherwise.

### `ibm_backend`

Real IBM Quantum backend. Reads `IBM_QUANTUM_TOKEN` env var.
Uses `SamplerV2` primitives.

```bash
export IBM_QUANTUM_TOKEN=your_token
pytest --quantum-real tests/hardware/
```

```python
@pytest.mark.quantum_real
def test_ibm(ibm_backend):
    from qiskit_ibm_runtime import SamplerV2
    from pytest_quantum import assert_measurement_distribution

    sampler = SamplerV2(ibm_backend)
    job = sampler.run([bell_circuit], shots=1024)
    counts = dict(job.result()[0].data.c.get_counts())
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
```

Environment variables:
- `IBM_QUANTUM_TOKEN` (required)
- `IBM_QUANTUM_BACKEND` (optional, defaults to least-busy)
- `IBM_QUANTUM_INSTANCE` (optional, defaults to `ibm-q/open/main`)

### `ionq_backend`

Real IonQ backend. Reads `IONQ_API_KEY` env var.

```bash
export IONQ_API_KEY=your_key
pytest --quantum-real
```

### `quantinuum_backend`

Real Quantinuum backend via pytket-quantinuum. Reads
`QUANTINUUM_USERNAME` and `QUANTINUUM_PASSWORD` env vars.

### `braket_cloud_device`

AWS Braket cloud device. Reads `BRAKET_DEVICE_ARN` and uses AWS credentials
from environment.

### `quantum_hardware_info`

Returns a dict of credential availability, useful for dynamic test skipping.

```python
def test_info(quantum_hardware_info):
    print(quantum_hardware_info)
    # {'ibm_available': True, 'ionq_available': False, ...}
```

---

## Utilities

### `quantum_benchmark`

Quantum-aware benchmark fixture. Uses pytest-benchmark if installed, otherwise
a simple timer.

```python
def test_circuit_speed(quantum_benchmark):
    result = quantum_benchmark(run_circuit, my_circuit, shots=1024, n_qubits=5)
```

### `benchmark_suite`

Collects per-assertion timing data within a test session. *(v1.0.0)*

```python
def test_assertion_overhead(benchmark_suite):
    import time
    with benchmark_suite.record("assert_unitary"):
        assert_unitary(qc, expected_matrix)
    benchmark_suite.print_summary()
```

### `multi_backend_runner`

Runs the same circuit on multiple backends in parallel (thread-based) and
returns a `dict[backend_name, counts]`. *(v1.0.0)*

```python
def test_cross_backend(multi_backend_runner):
    def make_bell():
        qc = QuantumCircuit(2, 2)
        qc.h(0); qc.cx(0, 1); qc.measure([0,1],[0,1])
        return qc

    results = multi_backend_runner.run_all(make_bell, shots=1024)
    for backend, counts in results.items():
        assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
```

### `shot_budget`

Track and enforce total shot usage within a single test.

```python
def test_budget(aer_simulator, shot_budget):
    budget = shot_budget(max_shots=10_000)
    shots = budget.allocate(2000)   # raises if over budget
    # ... run tests ...
    assert budget.remaining == 8000
```

### `quantum_shots` / `quantum_significance`

Returns the `--quantum-shots` / `--quantum-significance` CLI override values.

```python
def test_with_global_shot_override(quantum_shots):
    shots = quantum_shots or 1024   # use CLI override if provided
```

---

## `@pytest.mark.quantum_backends`: parametrised multi-framework tests

Run the same test on multiple backends automatically:

```python
@pytest.mark.quantum_backends("qiskit", "cirq", "pennylane")
def test_h_gate_all_frameworks(quantum_backend, quantum_backend_name):
    # quantum_backend = simulator for current framework
    # quantum_backend_name = "qiskit" | "cirq" | "pennylane"
    print(f"Testing on {quantum_backend_name}")
```
