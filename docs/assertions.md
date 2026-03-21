# Assertions Reference

pytest-quantum provides **38 assertion functions** covering every level of quantum
correctness verification. Import any assertion directly from the top-level package:

```python
from pytest_quantum import assert_unitary, assert_measurement_distribution, ...
```

---

## Decision Guide — Which assertion to use?

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
| Statevector is unit norm | `assert_normalized` | — |

---

## Unitary & Circuit Equivalence

These functions compare circuits at the matrix level — the most rigorous check
for deterministic quantum operations.

### `assert_unitary`

**Purpose:** Assert that a circuit implements the expected unitary matrix.

```python
assert_unitary(
    circuit,
    expected,
    *,
    atol=1e-6,
    allow_global_phase=True,
)
```

**Parameters**

: `circuit` — Any supported quantum circuit (Qiskit, Cirq, Braket, PennyLane, Pytket).
: `expected` — Target unitary as a numpy array of shape `(2**n, 2**n)`.
: `atol` — Absolute tolerance per element (default `1e-6`).
: `allow_global_phase` — If `True` (default), circuits that differ only by
  `e^{iθ}` are considered equivalent.

**When to use:** Whenever you have a deterministic gate or circuit and can write
down the expected matrix. This is the strongest available check.

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

**Failure message looks like:**

```
AssertionError: Unitary mismatch.
  Max |U_actual - U_expected|: 1.41e+00   (tolerance: 1.00e-06)
  Allow global phase: True
```

---

### `assert_circuits_equivalent`

**Purpose:** Assert that two circuits implement the same unitary, up to global phase.
Works across frameworks.

```python
assert_circuits_equivalent(
    circuit_a,
    circuit_b,
    *,
    atol=1e-6,
)
```

**Parameters**

: `circuit_a` — First circuit (any supported framework).
: `circuit_b` — Second circuit (any supported framework).
: `atol` — Absolute tolerance for numpy matrix comparison (default `1e-6`).

**When to use:** Porting an algorithm from one SDK to another, or verifying that a
compiler optimisation preserves circuit semantics. For Qiskit-to-Qiskit comparisons
with `mqt.qcec` installed, uses formal equivalence checking via ZX-calculus.

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

**Failure message looks like:**

```
AssertionError: Circuits are not equivalent.
  Max |U_a - U_b|: 1.00e+00   (tolerance: 1.00e-06)
```

---

### `assert_transpilation_preserves_semantics`

**Purpose:** Assert that a Qiskit transpilation step does not change the unitary.

```python
assert_transpilation_preserves_semantics(original, transpiled, *, atol=1e-6)
```

**When to use:** After calling `qiskit.transpile(qc, backend, optimization_level=3)`,
verify the optimisation was semantics-preserving.

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from pytest_quantum import assert_transpilation_preserves_semantics

def test_transpile_preserves():
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    sim = AerSimulator()
    qc_t = transpile(qc, sim, optimization_level=3)
    assert_transpilation_preserves_semantics(qc, qc_t)
```

---

## State Assertions

Use these when you have access to the full statevector from a simulator.

### `assert_normalized`

**Purpose:** Assert statevector has unit norm: `||ψ||₂ = 1`. *(NEW in v0.3.0)*

```python
assert_normalized(statevector, *, atol=1e-6)
```

**When to use:** After manually constructing a statevector, before passing it to
other assertions. Catches the common bug of forgetting to divide by the norm.

```python
import numpy as np
from pytest_quantum import assert_normalized

def test_manual_state_is_normalized():
    sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # Bell state
    assert_normalized(sv)  # passes
```

**Failure message looks like:**

```
AssertionError: Statevector is not normalized.
  Norm: 1.414214  (expected 1.0, tolerance: 1.00e-06)
  Deviation: 0.414214
  Hint: divide by np.linalg.norm(sv) to normalize.
```

---

### `assert_state_fidelity_above`

**Purpose:** Assert fidelity `F = |⟨actual|target⟩|²` is at or above a threshold.

```python
assert_state_fidelity_above(actual, target, threshold=0.99)
```

**Parameters**

: `actual` — Simulated output statevector (any shape; flattened internally).
: `target` — Ideal target statevector (same number of elements).
: `threshold` — Minimum acceptable fidelity (default `0.99`).

**When to use:** Primary assertion for Graphix MBQC tests, noisy simulations where you
expect slight deviation from ideal, and any test where fidelity is the natural metric.
`F=1.0` means identical states (up to global phase), `F=0.0` means orthogonal.

```python
import numpy as np
from pytest_quantum import assert_state_fidelity_above

BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

def test_bell_graphix(graphix_backend):
    from graphix.transpiler import Circuit
    circuit = Circuit(2)
    circuit.h(0); circuit.cnot(0, 1)
    pattern = circuit.transpile().pattern
    output = graphix_backend.run_pattern(pattern)
    assert_state_fidelity_above(output, BELL, threshold=0.999)
```

**Failure message looks like:**

```
AssertionError: State fidelity too low.
  |⟨actual|target⟩|² = 0.823456
  Required            ≥ 0.99
  Shortfall           = 1.67e-01
```

---

### `assert_states_close`

**Purpose:** Assert two statevectors are element-wise close, up to global phase.

```python
assert_states_close(actual, target, *, atol=1e-6)
```

**When to use:** Exact simulator-to-simulator comparisons where you want bit-for-bit
agreement. Stricter than `assert_state_fidelity_above`.

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

**Failure message looks like:**

```
AssertionError: Statevectors are not close (after global-phase alignment).
  Max |difference|: 7.07e-01   (tolerance: 1.00e-06)
```

---

## Measurement Distributions

Use these for shot-based tests where you run a circuit many times and compare the
resulting probability distribution.

### `assert_measurement_distribution`

**Purpose:** Assert measured counts match the expected probability distribution using
a chi-square goodness-of-fit test. Will not flake on shot noise.

```python
assert_measurement_distribution(
    counts,
    expected_probs,
    *,
    significance=0.05,
    min_expected_per_bucket=5,
)
```

**Parameters**

: `counts` — Measured counts dict, e.g. `{"00": 489, "11": 511}`.
: `expected_probs` — Expected probability dict, e.g. `{"00": 0.5, "11": 0.5}`. Must sum to 1.0.
: `significance` — P-value threshold for rejection (default `0.05`).
: `min_expected_per_bucket` — Warns if any bucket has fewer than 5 expected counts.

**When to use:** The primary assertion for any shot-based test. Prefer this over
comparing counts directly (`assert counts["00"] == 500` will flake ~50% of the time).

```python
from pytest_quantum import assert_measurement_distribution, min_shots

def test_bell_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()

    shots = min_shots(epsilon=0.05)   # 293 shots for 5% sensitivity
    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=shots
    ).result().get_counts()

    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
    )
```

**Failure message looks like:**

```
AssertionError: Measurement distribution does not match expected (p=0.0001 < 0.05).
  Chi-square statistic: 18.23   p-value: 0.0001

  State    Observed    Expected    Observed%   Expected%
  ─────────────────────────────────────────────────────
  00       120         500         12.00%      50.00%
  11       880         500         88.00%      50.00%
```

---

### `assert_counts_close`

**Purpose:** Assert two count dictionaries produce statistically close distributions,
measured by Total Variation Distance (TVD).

```python
assert_counts_close(counts_a, counts_b, *, max_tvd=0.05)
```

**When to use:** Comparing two backends (e.g. Aer vs real hardware), or verifying
that transpilation has not changed a circuit's output distribution. TVD=0 means
identical, TVD=1 means disjoint support.

```python
from pytest_quantum import assert_counts_close

def test_aer_backends_agree(aer_simulator, aer_statevector_simulator):
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

**Failure message looks like:**

```
AssertionError: Count distributions differ by TVD = 0.2345 > max_tvd = 0.05.
  TVD: 0.2345   (max allowed: 0.05)
```

---

## Density Matrix Assertions *(NEW in v0.2.0)*

Use these for mixed-state tests where you have the density matrix from a noisy
simulator. Import: `from pytest_quantum import assert_density_matrix_close, ...`

### `assert_density_matrix_close`

**Purpose:** Assert two density matrices are close element-wise.

```python
assert_density_matrix_close(rho, sigma, *, atol=1e-6)
```

**When to use:** Comparing a noisy simulation density matrix to an ideal density matrix.

```python
import numpy as np
from pytest_quantum import assert_density_matrix_close

def test_bell_density_matrix(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    qc.save_density_matrix()

    sim = aer_noise_simulator(error_rate=0.01)
    rho = sim.run(transpile(qc, sim)).result().data()["density_matrix"].data

    # Ideal Bell state density matrix
    ideal = np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype=complex)
    assert_density_matrix_close(rho, ideal, atol=0.05)
```

**Failure message looks like:**

```
AssertionError: Density matrices are not close.
  Max |ρ - σ|  : 1.50e-01   (tolerance: 1.00e-06)
  Trace distance: 0.1234
  Hint: use assert_trace_distance_below for a physically motivated bound.
```

---

### `assert_trace_distance_below`

**Purpose:** Assert trace distance `T(ρ,σ) = ½ Tr(|ρ-σ|)` is at most `max_distance`.

```python
assert_trace_distance_below(rho, sigma, *, max_distance=0.01)
```

**When to use:** Physically motivated bound on state distinguishability.
`T=0` means identical states, `T=1` means perfectly distinguishable (max probability of
distinguishing them in a single measurement equals `T`).

```python
import numpy as np
from pytest_quantum import assert_trace_distance_below

def test_noisy_state_close_to_ideal(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.save_density_matrix()

    sim = aer_noise_simulator(error_rate=0.005)
    rho = sim.run(transpile(qc, sim)).result().data()["density_matrix"].data

    ideal_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    assert_trace_distance_below(rho, ideal_plus, max_distance=0.05)
```

**Failure message looks like:**

```
AssertionError: Trace distance T(ρ,σ) = 0.1500 exceeds max_distance 0.0500.
  T=0: identical states.  T=1: perfectly distinguishable.
  Hint: check for noise — increase max_distance or reduce error_rate.
```

---

### `assert_purity_above`

**Purpose:** Assert purity `Tr(ρ²) ≥ min_purity`.

```python
assert_purity_above(rho, *, min_purity=0.95)
```

**When to use:** Checking how much noise has mixed a state. `Tr(ρ²)=1.0` for a
pure state, `Tr(ρ²)=1/d` for a maximally mixed d-dimensional state.

```python
from pytest_quantum import assert_purity_above

def test_low_noise_preserves_purity(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    qc.save_density_matrix()

    sim = aer_noise_simulator(error_rate=0.01)
    rho = sim.run(transpile(qc, sim)).result().data()["density_matrix"].data
    assert_purity_above(rho, min_purity=0.95)
```

**Failure message looks like:**

```
AssertionError: Purity Tr(ρ²) = 0.7832 is below min_purity 0.9500.
  Pure state=1.0, maximally mixed (4×4)=0.2500.
  Hint: high depolarizing noise reduces purity.
```

---

### `assert_partial_trace_close`

**Purpose:** Assert the reduced density matrix (partial trace over some qubits)
matches an expected value.

```python
assert_partial_trace_close(rho, keep_qubits, expected, *, atol=1e-6)
```

**When to use:** Verifying marginal state of a subsystem — e.g., checking that tracing
out qubit 1 of a Bell state gives the maximally mixed state `I/2`.

```python
import numpy as np
from pytest_quantum import assert_partial_trace_close

def test_bell_partial_trace():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(bell, bell.conj())
    mixed = np.eye(2, dtype=complex) / 2
    assert_partial_trace_close(rho, keep_qubits=[0], expected=mixed)
```

---

## Quantum Channel Assertions *(NEW in v0.3.0)*

### `assert_hermitian`

**Purpose:** Assert a matrix is Hermitian: `A == A†`.

```python
assert_hermitian(matrix, *, atol=1e-8)
```

**When to use:** Validating that a Hamiltonian or observable matrix is physically
valid (all physical observables must be Hermitian).

```python
import numpy as np
from pytest_quantum import assert_hermitian

H = np.array([[1, 1+1j], [1-1j, 2]], dtype=complex)  # Hermitian
assert_hermitian(H)
```

**Failure message looks like:**

```
AssertionError: Matrix is not Hermitian (A ≠ A†).
  Shape     : (2, 2)
  Max |A - A†| : 2.00e+00   (tolerance: 1.00e-08)
```

---

### `assert_positive_semidefinite`

**Purpose:** Assert all eigenvalues of a matrix are `≥ -atol`.

```python
assert_positive_semidefinite(matrix, *, atol=1e-8)
```

**When to use:** Validating that a density matrix is physically valid (density matrices
must be PSD). Also useful for checking Choi matrices of quantum channels.

**Failure message looks like:**

```
AssertionError: Matrix is not positive semi-definite.
  Minimum eigenvalue: -2.500000e-01   (threshold: -1.00e-08)
  All eigenvalues    : [-0.25, 0.5, 0.75]
```

---

### `assert_commutes_with`

**Purpose:** Assert two square matrices commute: `AB == BA`.

```python
assert_commutes_with(op_a, op_b, *, atol=1e-8)
```

**When to use:** Verifying that a Hamiltonian commutes with its symmetry operators, or
that two Pauli operators in a stabilizer code commute as required.

```python
import numpy as np
from pytest_quantum import assert_commutes_with

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)
# X⊗I commutes with I⊗Z
assert_commutes_with(np.kron(X, I), np.kron(I, Z))
```

**Failure message looks like:**

```
AssertionError: Matrices do not commute: [A, B] ≠ 0.
  Max |AB - BA|: 2.00e+00   (tolerance: 1.00e-08)
```

---

### `assert_channel_is_cptp`

**Purpose:** Assert Kraus operators satisfy completeness: `Σ K†K == I`.

```python
assert_channel_is_cptp(kraus_ops, *, atol=1e-8)
```

**When to use:** Validating custom noise models or quantum channel implementations.
CPTP (completely positive, trace-preserving) is the necessary and sufficient condition
for a physically valid quantum channel.

```python
from pytest_quantum import assert_channel_is_cptp
from pytest_quantum.random import depolarizing_kraus

def test_depolarizing_is_cptp():
    kraus = depolarizing_kraus(n_qubits=1, error_rate=0.1)
    assert_channel_is_cptp(kraus)
```

**Failure message looks like:**

```
AssertionError: Kraus operators do not satisfy the CPTP completeness relation (ΣK†K ≠ I).
  Number of Kraus operators: 2
  Matrix dimension: 2x2
  ||ΣK†K - I||_fro: 0.316228  (tolerance: 1.00e-08)
  Hint: Check that your Kraus operators form a valid quantum channel.
        Use depolarizing_kraus(n_qubits=1, error_rate=0.1) for a valid example.
```

---

### `assert_process_fidelity_above`

**Purpose:** Assert process fidelity `F_process(A, B) ≥ threshold`.

```python
assert_process_fidelity_above(channel_a, channel_b, threshold=0.99, *, atol=1e-8)
```

**When to use:** Comparing a noisy implementation of a channel to its ideal version.
Supports Kraus lists, unitary matrices, and Qiskit quantum info objects.

**Failure message looks like:**

```
AssertionError: Process fidelity 0.8234 < threshold 0.99.
  Fidelity:  0.8234
  Threshold: 0.99
  Gap:       0.1666
```

---

### `assert_noise_fidelity_above`

**Purpose:** Assert state fidelity between a noisy density matrix and an ideal state:
`F(ρ, |ψ⟩) = ⟨ψ|ρ|ψ⟩ ≥ threshold`.

```python
assert_noise_fidelity_above(noisy_dm, ideal_state, threshold=0.99)
```

**When to use:** After running a noisy simulation, checking how close the output density
matrix is to the ideal pure state you would get without noise.

```python
import numpy as np, math
from pytest_quantum import assert_noise_fidelity_above

def test_noise_fidelity(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    qc.save_density_matrix()

    sim = aer_noise_simulator(error_rate=0.01)
    rho = sim.run(transpile(qc, sim)).result().data()["density_matrix"].data

    ideal = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    assert_noise_fidelity_above(rho, ideal, threshold=0.95)
```

---

## Entanglement Assertions *(NEW in v0.3.0)*

### `assert_entanglement_entropy_below`

**Purpose:** Assert von Neumann entanglement entropy `S(ρ_A) ≤ max_entropy`.

```python
assert_entanglement_entropy_below(
    statevector, partition, max_entropy, *, n_qubits=None
)
```

**When to use:** Checking that a circuit produces at most a certain amount of
entanglement. `max_entropy=0` means product state, `max_entropy=1` means at most 1 ebit.

```python
import numpy as np
from pytest_quantum import assert_entanglement_entropy_below

def test_product_state_has_zero_entropy():
    # |0⊗0> = |00> is separable
    sv = np.array([1, 0, 0, 0], dtype=complex)
    assert_entanglement_entropy_below(sv, partition=[0], max_entropy=1e-10)

def test_bell_state_entropy():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    # Bell state has exactly 1 bit of entanglement entropy — allow up to 1.01
    assert_entanglement_entropy_below(bell, partition=[0], max_entropy=1.01)
```

**Failure message looks like:**

```
AssertionError: Entanglement entropy S(ρ_A) = 0.9423 bits exceeds max_entropy = 0.5 bits.
  Partition (kept qubits): [0]
  Full system: 2 qubits
  S = 0.9423 bits  (max allowed: 0.5000 bits)
  Hint: S = 1.0 for a maximally entangled Bell state;
        S = 0.0 for a product (separable) state.
```

---

### `assert_bloch_sphere_close`

**Purpose:** Assert a single-qubit state is close to the expected Bloch sphere position.

```python
assert_bloch_sphere_close(
    statevector, expected_theta, expected_phi, *, atol=0.1
)
```

**Parameters**

: `statevector` — Single-qubit state (length-2 complex array).
: `expected_theta` — Expected polar angle in radians. `θ=0` → |0⟩ (north pole), `θ=π` → |1⟩ (south pole).
: `expected_phi` — Expected azimuthal angle in radians.
: `atol` — Tolerance for Bloch vector Euclidean distance (default `0.1`).

**When to use:** Testing rotation gates where you expect the qubit to end up at a
specific location on the Bloch sphere.

```python
import numpy as np
from pytest_quantum import assert_bloch_sphere_close

def test_rx_pi_puts_qubit_at_south_pole():
    # RX(π)|0⟩ = -i|1⟩ → south pole, θ=π, φ=0
    state = np.array([0, -1j], dtype=complex)
    assert_bloch_sphere_close(state, expected_theta=np.pi, expected_phi=0.0)
```

**Failure message looks like:**

```
AssertionError: Bloch sphere position mismatch.
  Expected: θ=0.000 rad, φ=0.000 rad  (|0⟩ = north pole)
  Actual:   θ=3.142 rad, φ=0.000 rad  (south pole)
  Bloch vector distance: 2.0000  (max allowed: 0.1)
```

---

### `assert_schmidt_rank_at_most`

**Purpose:** Assert Schmidt rank of a bipartite pure state is at most `max_rank`.

```python
assert_schmidt_rank_at_most(
    statevector, partition, max_rank, *, n_qubits=None, tol=1e-10
)
```

**When to use:** Testing that a circuit produces a state with limited entanglement
structure. Schmidt rank 1 means separable (product state); rank > 1 means entangled.

```python
import numpy as np
from pytest_quantum import assert_schmidt_rank_at_most

def test_product_state_rank_1():
    sv = np.array([1, 0, 0, 0], dtype=complex)  # |00>
    assert_schmidt_rank_at_most(sv, partition=[0], max_rank=1)
```

**Failure message looks like:**

```
AssertionError: Schmidt rank 2 exceeds max_rank 1.
  Partition A qubits: [0]
  Non-zero Schmidt coefficients: [0.7071, 0.7071]
  Rank 1 = separable/product state; Rank > 1 = entangled.
```

---

## Information Theory Assertions *(NEW in v0.3.0)*

### `assert_hellinger_close`

**Purpose:** Assert Hellinger distance `H(p,q) ≤ max_distance`. Range `[0, 1]`.

```python
assert_hellinger_close(counts_a, counts_b, *, max_distance=0.1)
```

**When to use:** Comparing two measurement distributions symmetrically and in a
bounded way. Hellinger distance is more symmetric than KL divergence and always bounded.

```python
from pytest_quantum import assert_hellinger_close

def test_two_runs_agree(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(1)
    qc.h(0); qc.measure_all()

    counts_1 = aer_simulator.run(transpile(qc, aer_simulator), shots=2000).result().get_counts()
    counts_2 = aer_simulator.run(transpile(qc, aer_simulator), shots=2000).result().get_counts()

    assert_hellinger_close(counts_1, counts_2, max_distance=0.15)
```

**Failure message looks like:**

```
AssertionError: Hellinger distance H = 0.2345 exceeds max_distance = 0.1000.
  H(p, q) = 0.2345  (max allowed: 0.1000)

  Outcome    p (left)   q (right)        √p        √q
  ────────────────────────────────────────────────────
  00           0.5000      0.3000    0.7071    0.5477
  11           0.5000      0.7000    0.7071    0.8367

  Hint: H=0.0 means identical distributions, H=1.0 means completely different.
```

---

### `assert_kl_divergence_below`

**Purpose:** Assert KL divergence `D_KL(observed || expected) ≤ max_kl`.

```python
assert_kl_divergence_below(counts, expected_probs, *, max_kl=0.1)
```

**When to use:** When you have a specific expected distribution and want to measure
how much the observed distribution diverges from it. Note: KL is asymmetric and undefined
if `expected_probs` assigns zero probability to an observed outcome.

**Failure message looks like:**

```
AssertionError: KL divergence D(observed||expected) = 0.4567 exceeds max_kl = 0.1000.
  D_KL = 0.4567  (max allowed: 0.1000)

  Outcome    observed   expected   contribution
  ──────────────────────────────────────────────
  00           0.8000     0.5000        +0.3520
  11           0.2000     0.5000        +0.1047
```

---

### `assert_cross_entropy_below`

**Purpose:** Assert cross-entropy `H(P, Q) = -Σ P(x) log₂ Q(x) ≤ max_ce`.

```python
assert_cross_entropy_below(counts, expected_probs, *, max_ce=1.0)
```

**When to use:** Cross-entropy benchmarking (XEB) for quantum supremacy experiments.
Measures how well the expected circuit distribution predicts the observed samples.

---

## Observable / Expectation Value Assertions *(NEW in v0.2.0)*

### `assert_expectation_value_close`

**Purpose:** Assert an expectation value is close to the expected value.

```python
assert_expectation_value_close(actual, expected, *, atol=0.01)
```

**When to use:** Checking results from PennyLane `qml.expval()` or Qiskit Estimator.

```python
import pennylane as qml
import numpy as np
from pytest_quantum import assert_expectation_value_close

def test_z_expectation(pennylane_device):
    dev = pennylane_device(wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(0)
        return qml.expval(qml.PauliZ(0))

    # X|0⟩ = |1⟩, ⟨Z⟩ = -1
    assert_expectation_value_close(circuit(), expected=-1.0, atol=1e-6)
```

---

### `assert_ground_state_energy_close`

**Purpose:** Assert a computed ground state energy is close to the expected value.

```python
assert_ground_state_energy_close(actual_energy, expected_energy, *, atol=0.01)
```

**When to use:** VQE / QAOA energy results where you allow a tolerance for
optimizer convergence error.

```python
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

---

## Qiskit Primitives *(NEW in v0.2.0)*

### `assert_sampler_distribution`

**Purpose:** Assert Qiskit 1.0+ `StatevectorSampler` results match expected distribution.

```python
assert_sampler_distribution(sampler_result, expected_probs, *, significance=0.05)
```

**When to use:** Testing circuits using the Qiskit 1.0+ Primitives API.

```python
from pytest_quantum import assert_sampler_distribution

def test_bell_sampler(qiskit_sampler):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])

    result = qiskit_sampler.run([qc], shots=2000).result()
    assert_sampler_distribution(result[0], {"00": 0.5, "11": 0.5})
```

---

### `assert_estimator_close`

**Purpose:** Assert Qiskit 1.0+ `StatevectorEstimator` expectation value is close to expected.

```python
assert_estimator_close(estimator_result, expected, *, atol=0.01)
```

```python
from pytest_quantum import assert_estimator_close

def test_z_estimator(qiskit_estimator):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(1)
    qc.x(0)  # |1⟩

    op = SparsePauliOp("Z")
    result = qiskit_estimator.run([(qc, op)]).result()
    assert_estimator_close(result[0], expected=-1.0, atol=1e-6)
```

---

## Circuit Structure

These assertions check static properties without executing the circuit.

### `assert_circuit_depth`

**Purpose:** Assert a circuit's depth is within specified bounds.

```python
assert_circuit_depth(circuit, *, max_depth=None, min_depth=None)
```

**When to use:** Regression testing compiler output to ensure depth doesn't increase,
or verifying circuits meet hardware depth constraints. Supported: Qiskit, Cirq, Braket.

```python
from pytest_quantum import assert_circuit_depth

def test_bell_depth():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    assert_circuit_depth(qc, max_depth=3)

def test_transpiled_depth(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(5)
    for i in range(4): qc.cx(i, i+1)
    qc_t = transpile(qc, aer_simulator, optimization_level=3)
    assert_circuit_depth(qc_t, max_depth=10, min_depth=2)
```

**Failure message looks like:**

```
AssertionError: Circuit depth 15 exceeds max_depth 10.
```

---

### `assert_circuit_width`

**Purpose:** Assert a circuit acts on exactly `expected_qubits` qubits.

```python
assert_circuit_width(circuit, expected_qubits)
```

**When to use:** Ensuring a circuit generator produces the right number of qubits.
Supported: Qiskit, Cirq, Braket, PennyLane.

```python
from pytest_quantum import assert_circuit_width

def test_ghz_width():
    from qiskit import QuantumCircuit
    n = 4
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1): qc.cx(i, i+1)
    assert_circuit_width(qc, expected_qubits=4)
```

**Failure message looks like:**

```
AssertionError: Circuit has 3 qubits, expected 4.
```

---

### `assert_gate_count`

**Purpose:** Assert a circuit contains exactly `expected` occurrences of `gate_name`.

```python
assert_gate_count(circuit, gate_name, expected)
```

**When to use:** T-count optimization verification, or ensuring a specific gate was
not duplicated by a compiler pass. Supported: Qiskit, Cirq, PennyLane, Pytket.

```python
from pytest_quantum import assert_gate_count

def test_t_count():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.t(0); qc.t(1); qc.cx(0, 1)
    assert_gate_count(qc, "t", 2)
    assert_gate_count(qc, "cx", 1)
```

**Failure message looks like:**

```
AssertionError: Gate count for "t" is 3, expected 2.
```

---

### `assert_circuit_is_clifford`

**Purpose:** Assert a circuit uses only Clifford gates (H, S, CNOT, CZ, Pauli, etc.).

```python
assert_circuit_is_clifford(circuit)
```

**When to use:** Verifying that a circuit can be efficiently simulated classically via
the Gottesman-Knill theorem, or checking that a QEC syndrome extraction circuit uses
only Clifford operations. Supported: Qiskit, Cirq, Braket, PennyLane, Pytket.

```python
from pytest_quantum import assert_circuit_is_clifford

def test_bell_is_clifford():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    assert_circuit_is_clifford(qc)

def test_t_gate_not_clifford():
    from qiskit import QuantumCircuit
    import pytest
    qc = QuantumCircuit(1)
    qc.t(0)
    with pytest.raises(AssertionError, match="non-Clifford"):
        assert_circuit_is_clifford(qc)
```

**Failure message looks like:**

```
AssertionError: Circuit contains non-Clifford gates: ['t', 'tdg']
  Non-Clifford gates found: t (1 occurrence)
```

---

### `assert_has_diagram`

**Purpose:** Assert a circuit's text diagram matches the expected string.

```python
assert_has_diagram(circuit, expected_diagram)
```

**When to use:** Snapshot-style regression testing of circuit structure — catch when a
compiler accidentally reorders gates.

---

## Snapshot / Golden-File Testing *(NEW in v0.2.0)*

### `assert_unitary_snapshot`

**Purpose:** Assert a circuit's unitary matches its saved snapshot.

```python
assert_unitary_snapshot(
    circuit, name, *, update=False, atol=1e-6, allow_global_phase=True
)
```

**How it works:** First run saves the unitary to `.pytest-quantum-snapshots/<name>.npy`.
Subsequent runs load and compare. Use `--quantum-update-snapshots` CLI flag to regenerate.

**When to use:** Regression testing — catch when a refactor accidentally changes a
circuit's unitary without needing to type out the full matrix.

```python
from pytest_quantum import assert_unitary_snapshot

def test_grover_oracle_unchanged():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h([0, 1]); qc.cz(0, 1); qc.h([0, 1])
    assert_unitary_snapshot(qc, name="grover_2qubit_oracle")
    # Regenerate: pytest --quantum-update-snapshots
```

---

### `assert_distribution_snapshot`

**Purpose:** Assert a count dictionary matches its saved snapshot.

```python
assert_distribution_snapshot(counts, name, *, update=False, max_tvd=0.05)
```

**When to use:** Catching changes in output distribution after compiler or noise model
changes.

---

## OpenQASM Round-Trip *(NEW in v0.3.0)*

### `assert_qasm_roundtrip`

**Purpose:** Assert a circuit survives QASM export and re-import with equivalent unitary.

```python
assert_qasm_roundtrip(circuit, *, atol=1e-6)
```

**When to use:** Testing that a custom gate correctly implements QASM serialization,
or that your circuit library's QASM export is lossless.

```python
from pytest_quantum import assert_qasm_roundtrip

def test_bell_qasm_roundtrip():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    assert_qasm_roundtrip(qc)
```

---

## QEC / Stim Assertions *(NEW in v0.3.0)*

These assertions require `stim` to be installed: `pip install stim`.

### `assert_stim_logical_error_rate_below`

**Purpose:** Assert that a Stim circuit's logical error rate is below a threshold.

```python
assert_stim_logical_error_rate_below(circuit, max_error_rate, shots=10000)
```

**When to use:** Testing that a QEC code provides sufficient protection against
physical errors for a given physical error rate.

```python
import importlib.util
import pytest

@pytest.mark.skipif(not importlib.util.find_spec("stim"), reason="stim not installed")
def test_repetition_code_logical_error_rate(stim_sampler):
    import stim
    from pytest_quantum import assert_stim_logical_error_rate_below

    circuit = stim.Circuit("""
        X_ERROR(0.01) 0 1 2
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-3]
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-2] rec[-1]
    """)
    assert_stim_logical_error_rate_below(circuit, max_error_rate=0.05, shots=5000)
```

**Failure message looks like:**

```
AssertionError: Logical error rate 0.0823 exceeds max_error_rate 0.05.
  Observed logical errors: 823 / 10000 shots
```

---

### `assert_stim_detector_error_rate_below`

**Purpose:** Assert that the detector error rate in a Stim circuit is below a threshold.

```python
assert_stim_detector_error_rate_below(circuit, max_error_rate, shots=10000)
```

---

### `assert_stabilizer_state`

**Purpose:** Assert that a statevector is a stabilizer state (i.e., a +1 eigenstate of
all stabilizer operators in the given list).

```python
assert_stabilizer_state(statevector, stabilizers)
```

**When to use:** Verifying that a QEC code preparation circuit correctly encodes the
logical |0̄⟩ state.
