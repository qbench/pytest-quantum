# Concepts

This page explains the "why" behind pytest-quantum's design choices. Understanding
these concepts will help you write quantum tests that are both correct and reliable.

---

## 1. Why classical assertions don't work for quantum

Quantum programs produce probabilistic outputs. A circuit that applies a Hadamard
gate and measures will produce `0` roughly half the time and `1` roughly half the time
— even with a perfect simulator.

**The wrong way:**

```python
# This test fails ~50% of the time — it's not testing your circuit, it's testing luck
def test_h_gate_bad(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(1)
    qc.h(0); qc.measure_all()

    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=1000).result().get_counts()
    assert counts["0"] == 500  # WRONG: this will fail constantly
```

**The right way:**

```python
from pytest_quantum import assert_measurement_distribution

def test_h_gate_correct(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(1)
    qc.h(0); qc.measure_all()

    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=1000).result().get_counts()
    # Chi-square test: fails only when the distribution is genuinely wrong
    assert_measurement_distribution(counts, {"0": 0.5, "1": 0.5})
```

`assert_measurement_distribution` uses a chi-square goodness-of-fit test under the
hood. It only rejects the null hypothesis (that the counts came from the expected
distribution) when the deviation is statistically significant. Random shot noise will
not cause the test to fail.

### How many shots do I need?

Use the `min_shots` and `recommended_shots` utilities:

```python
from pytest_quantum import min_shots, recommended_shots

# Minimum shots to detect a 5% error (TVD ≥ 0.05) with 80% power
n = min_shots(epsilon=0.05)       # → 293 shots
n = min_shots(epsilon=0.01)       # → 7299 shots

# Shots based on your expected distribution (ensures 5+ expected per bucket)
n = recommended_shots({"00": 0.499, "01": 0.001, "10": 0.001, "11": 0.499})
# → 5000 (driven by the 0.1% outcomes)
```

`min_shots(epsilon)` computes the shot count needed to detect a deviation of size
`epsilon` with 80% statistical power at p=0.05 significance, using chi-square power
analysis.

---

## 2. Shot noise — the most common source of flaky tests

### What is shot noise?

When you measure a qubit in state `α|0⟩ + β|1⟩`, each measurement is an independent
Bernoulli trial with probability `|β|²` of returning `1`. With `N` shots, the number
of `1` outcomes follows a Binomial distribution: `Binomial(N, |β|²)`.

For a perfect Hadamard gate, `|β|² = 0.5`. With 100 shots, you should get about 50
ones — but the standard deviation is `√(100 × 0.5 × 0.5) = 5`. A result of 45 or
55 is perfectly normal.

```
H gate, 100 shots:  Possible outcomes
                    ┌──────────────────────────────────────┐
Expected ≈ 50       │          ████████████████            │
                    │      ████████████████████████        │
                    │  ████████████████████████████████    │
                    │──────────────────────────────────────│
                    │   35    40    45    50    55    60    │
                    └──────────────────────────────────────┘
```

A test that expects exactly 50 and tolerates zero deviation will fail roughly 50% of
the time — even with a perfectly functioning circuit. That's not a test; it's a coin
flip.

### The significance level (p-value)

`assert_measurement_distribution` uses a chi-square test with a configurable
significance level `α`. The test rejects only when `p < α`, where `p` is the
probability of observing counts this far from expected if the circuit is correct.

- `significance=0.05` (default): rejects only when there's less than 5% chance the
  deviation is due to noise. For a suite of 100 tests, you'd expect ~5 false failures
  per run — tolerable.
- `significance=0.001`: very lenient, appropriate for noisy circuits or large shot
  counts where small systematic deviations are acceptable.
- `significance=1e-10`: ultra-strict, appropriate for noiseless simulators where any
  deviation indicates a genuine bug.

### Shot-count table: epsilon vs shots needed

| Goal | Epsilon | Min shots needed |
|---|---|---|
| Detect 10% error | 0.10 | ~74 shots |
| Detect 5% error | 0.05 | ~293 shots |
| Detect 2% error | 0.02 | ~1,832 shots |
| Detect 1% error | 0.01 | ~7,299 shots |
| Detect 0.1% error | 0.001 | ~729,874 shots |

### The `@pytest.mark.quantum_slow` pattern

For tests requiring thousands of shots, use the `quantum_slow` marker to skip them
in fast CI runs:

```python
import pytest
from pytest_quantum import assert_measurement_distribution, min_shots

@pytest.mark.quantum_slow  # skipped unless --quantum-slow is passed
def test_precision_h_gate(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    shots = min_shots(epsilon=0.01)   # 7299 shots — 1% precision
    qc = QuantumCircuit(1)
    qc.h(0); qc.measure_all()
    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=shots
    ).result().get_counts()
    assert_measurement_distribution(counts, {"0": 0.5, "1": 0.5})
```

```bash
pytest                   # fast: quantum_slow tests skipped
pytest --quantum-slow    # full run including slow tests
```

---

## 3. Global phase — why `e^{iθ}|ψ⟩ == |ψ⟩`

In quantum mechanics, a global phase factor `e^{iθ}` is physically unobservable.
The states `|ψ⟩` and `e^{iπ/4}|ψ⟩` are identical in every measurable respect —
they produce the same measurement probabilities, the same fidelity with any target
state, and the same expectation values for all observables.

This matters for testing because different simulators may return unitaries that differ
by a global phase:

```
Qiskit H gate unitary:         Cirq H gate unitary (may differ by e^{iπ}):
[[ 1/√2,  1/√2],               [[-1/√2, -1/√2],
 [ 1/√2, -1/√2]]                [-1/√2,  1/√2]]
```

These are physically identical. `np.allclose(U1, U2)` would return `False`, but
`assert_unitary(circuit, expected)` correctly returns `True` (with the default
`allow_global_phase=True`).

### When to use `allow_global_phase=False`

Set `allow_global_phase=False` when you need to test that a circuit produces an exact
global phase — primarily when:

1. **Phase oracles**: A phase oracle is supposed to map `|x⟩ → (-1)^{f(x)}|x⟩`. The
   "phase" is a *relative* phase between basis states, not a global phase. You must
   distinguish `|0⟩ - |1⟩` from `|0⟩ + |1⟩`.

2. **Controlled operations**: A gate `U` that has global phase `e^{iθ}` will have
   *relative* phase in the controlled version `CU`. Grover's oracle must produce the
   correct sign pattern, and `allow_global_phase=False` enforces this.

```python
from pytest_quantum import assert_unitary
import numpy as np

def test_phase_oracle_exact():
    from qiskit import QuantumCircuit
    # Oracle: marks |1⟩ with a -1 phase
    qc = QuantumCircuit(1)
    qc.z(0)
    Z_exact = np.array([[1, 0], [0, -1]], dtype=complex)
    # Must be exact — allow_global_phase=False
    assert_unitary(qc, Z_exact, allow_global_phase=False)
```

---

## 4. Qubit ordering — the most confusing cross-framework issue

Different quantum frameworks use different conventions for which qubit corresponds to
the most-significant bit (MSB) in the computational basis.

### The two conventions

**Big-endian (Cirq, Pytket, Braket):**
Qubit 0 is the MSB. The state `|q0 q1⟩` is ordered as: `|00⟩, |01⟩, |10⟩, |11⟩`.
The CNOT matrix (control=q0, target=q1) looks like:

```
Big-endian CNOT:
[[1, 0, 0, 0],   |00⟩ → |00⟩
 [0, 1, 0, 0],   |01⟩ → |01⟩
 [0, 0, 0, 1],   |10⟩ → |11⟩
 [0, 0, 1, 0]]   |11⟩ → |10⟩
```

**Little-endian (Qiskit):**
Qubit 0 is the LSB. The state `|q1 q0⟩` is ordered as: `|00⟩, |10⟩, |01⟩, |11⟩`.
The same CNOT (control=q0, target=q1) looks different:

```
Little-endian CNOT (Qiskit):
[[1, 0, 0, 0],   |00⟩ → |00⟩
 [0, 0, 0, 1],   |10⟩ → |11⟩
 [0, 0, 1, 0],   |01⟩ → |01⟩
 [0, 1, 0, 0]]   |11⟩ → |10⟩
```

### What to do about it

**For `assert_circuits_equivalent`:** pytest-quantum handles qubit ordering
automatically for all supported cross-framework comparisons. Just pass both circuits
and the comparison is correct.

**For manual matrix construction:** Always use `to_unitary()` from the same framework
for both matrices, or use pytest-quantum's `assert_unitary` which handles the
conversion internally.

**Example:**

```python
# Correct: assert_circuits_equivalent handles ordering for you
from pytest_quantum import assert_circuits_equivalent
import cirq
from qiskit import QuantumCircuit

qk = QuantumCircuit(2); qk.cx(0, 1)
q0, q1 = cirq.LineQubit.range(2)
cc = cirq.Circuit(cirq.CNOT(q0, q1))
assert_circuits_equivalent(qk, cc)  # passes — ordering handled automatically

# Careful: if you compare the raw unitary matrices, they will look different!
# Get the Qiskit unitary, check with Qiskit's own convention:
from pytest_quantum import assert_unitary
BIG_ENDIAN_CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
assert_unitary(cc, BIG_ENDIAN_CNOT)   # Cirq circuit: big-endian → passes
```

---

## 5. Session scope — why simulators are shared across tests

### The startup cost problem

Creating an `AerSimulator` is not free. The first call triggers:
- Qiskit Aer JIT compilation (~0.5s)
- BLAS library initialization
- Optional GPU detection

For a test suite with 100 tests, creating a new simulator in each test would add
~50 seconds of overhead with no benefit.

### Session scope eliminates this overhead

pytest-quantum's fixtures use `scope="session"`:

```
First test requesting aer_simulator:    creates AerSimulator (once)
All subsequent tests:                   reuse the same instance
```

This is safe because `AerSimulator` is stateless — each `.run()` call is independent
and thread-safe. The simulator object itself carries no mutable state between calls.

### The exception: `aer_noise_simulator`

`aer_noise_simulator` is **function-scoped** because each test passes a different
`error_rate` to the factory:

```python
def test_5pct_noise(aer_noise_simulator):
    sim = aer_noise_simulator(error_rate=0.05)  # fresh each test

def test_1pct_noise(aer_noise_simulator):
    sim = aer_noise_simulator(error_rate=0.01)  # fresh each test
```

If you need an isolated simulator for any other reason, create it directly inside
your test rather than via the fixture:

```python
def test_custom_backend():
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="matrix_product_state")
    # ... use sim directly
```

### Do not modify session fixtures

Because simulators are shared, avoid mutating the fixture object:

```python
# BAD — modifies the shared session fixture!
def test_bad(aer_simulator):
    aer_simulator.set_options(shots=10000)  # affects ALL subsequent tests!

# GOOD — create a local instance if you need custom configuration
def test_good():
    from qiskit_aer import AerSimulator
    sim = AerSimulator(shots=10000)
    # ... use sim locally
```

---

## 6. When to use which assertion

### Decision by state representation

| What you have | What to assert | Why |
|---|---|---|
| Deterministic gate, known matrix | `assert_unitary` | Strongest check; catches phase errors |
| Two frameworks, same algorithm | `assert_circuits_equivalent` | Handles qubit ordering automatically |
| Statevector from noiseless sim | `assert_states_close` | Element-wise, exact |
| Statevector from noisy sim | `assert_state_fidelity_above` | Fidelity tolerates small deviations |
| Density matrix from noisy sim | `assert_density_matrix_close` | Mixed-state equivalent of states_close |
| Want to quantify noisiness | `assert_purity_above` | Tr(ρ²)=1 for pure, 1/d for maximally mixed |
| Shot counts from real/noisy hw | `assert_measurement_distribution` | Chi-square; won't flake on noise |
| Two backends, same circuit | `assert_counts_close` | TVD comparison |

### Decision by test purpose

| Test purpose | Recommended assertion |
|---|---|
| Gate implementation correctness | `assert_unitary` |
| MBQC / Graphix output | `assert_state_fidelity_above` |
| Noise tolerance verification | `assert_purity_above`, `assert_trace_distance_below` |
| Compiler regression | `assert_unitary_snapshot`, `assert_circuit_depth` |
| Hardware-constrained circuits | `assert_circuit_depth`, `assert_gate_count` |
| Cross-framework port validation | `assert_circuits_equivalent` |
| VQE / QAOA result | `assert_ground_state_energy_close` |
| QEC code logical error rate | `assert_stim_logical_error_rate_below` |
| Entanglement structure | `assert_entanglement_entropy_below`, `assert_schmidt_rank_at_most` |
