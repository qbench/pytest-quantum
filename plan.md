# pytest-quantum v0.2.0 — Plan

## Where We Are (v0.1.1)

Live on PyPI. 38 tests passing from a fresh install. Core plugin works.
Framework support is uneven — some assertions silently don't apply to certain
frameworks, CLI options are registered but not wired up, and cross-framework
multi-qubit comparison is broken due to qubit ordering.

---

## What v0.2.0 Must Solve

### 1. Broken / Incomplete Things (must fix)

**A. `--quantum-shots` and `--quantum-significance` are no-ops**
The CLI options are registered but the fixtures ignore them entirely.
Users who pass `--quantum-shots 4000` see zero effect.

Fix: read `config.getoption("quantum_shots")` inside each fixture's factory
and pass it to `shots=` when running circuits. Read `quantum_significance`
inside `assert_measurement_distribution` via the pytest request object.

**B. Cirq has no shot-based simulation**
`assert_measurement_distribution` and `assert_counts_close` raise TypeError
on Cirq circuits. But Cirq *does* support shot simulation via
`cirq.Simulator().run(circuit, repetitions=N)` — we just never wired it.

Fix: add a `cirq_sampler` fixture (distinct from `cirq_simulator`) that
wraps `cirq.Simulator().run()` and returns count dicts. Then
`assert_measurement_distribution` works on Cirq too.

**C. Braket has no gate count support**
`assert_gate_count` raises NotImplementedError for Braket circuits.
Braket circuits do expose `circuit.instructions` which is iterable.

Fix: iterate `circuit.instructions`, match `instruction.operator.name`
case-insensitively.

**D. PennyLane has no circuit depth**
`assert_circuit_depth` raises TypeError for PennyLane QNodes.
PennyLane exposes `qml.specs(qnode)()["depth"]` after execution.

Fix: call `qml.specs(circuit)()["depth"]` in `_get_depth()`.

**E. PennyLane gate count requires prior execution**
`assert_gate_count` requires the QNode tape to exist, which means the user
must call the circuit before asserting. This is surprising and undocumented.

Fix: execute the circuit with a dry-run (zero shots) to populate the tape
automatically, then count. No longer requires prior execution.

**F. Cross-framework multi-qubit comparison is wrong**
Qiskit uses little-endian qubit ordering, Cirq uses big-endian.
A CNOT in Qiskit produces a different 4×4 matrix than a CNOT in Cirq.
Currently `assert_circuits_equivalent` gives wrong results silently.

Fix: detect when comparing across frameworks, apply qubit reversal
permutation to one matrix before comparing. Add `qubit_order` parameter
to `to_unitary()` so callers can request `"big_endian"` or `"little_endian"`.

---

### 2. New Features

**A. Density Matrix Assertions** — most important new module

Noisy quantum computing produces mixed states (density matrices), not
statevectors. Every noise test and open-system simulation needs this.
Currently there is no way to test density matrix outputs.

New module: `assertions/density.py`

```python
assert_density_matrix_close(rho, sigma, atol=1e-6)
  # element-wise comparison after global phase alignment

assert_trace_distance_below(rho, sigma, max_distance=0.01)
  # T(ρ,σ) = 0.5 * Tr(|ρ-σ|), measures distinguishability

assert_purity_above(rho, min_purity=0.95)
  # Tr(ρ²) ≥ min_purity (1.0 = pure state, 1/d = maximally mixed)

assert_partial_trace_close(rho, target_subsystem, expected_rho, atol=1e-6)
  # Trace out subsystem, compare reduced density matrix
```

Why this matters: every `aer_noise_simulator` test produces density matrices.
The current assertion suite has no way to inspect them.

**B. Expectation Value Assertions** — essential for VQE / QAOA users

The single most common quantum chemistry and optimization test is:
"does my variational circuit produce the right expectation value?"
Currently there is no assertion for this.

New module: `assertions/observables.py`

```python
assert_expectation_value_close(circuit, observable, expected, atol=0.1,
                                shots=None, backend=None)
  # Runs circuit, measures <observable>, checks |result - expected| <= atol
  # observable: numpy matrix, Qiskit SparsePauliOp, PennyLane Hamiltonian, etc.

assert_ground_state_energy_close(circuit, hamiltonian, expected_energy,
                                  atol=0.1)
  # Convenience: checks VQE convergence to known ground state energy
```

Why this matters: VQE, QAOA, and quantum chemistry are the biggest quantum
computing use cases. These researchers have NO good way to write regression
tests today.

**C. Snapshot / Golden File Testing** — killer feature for compiler testing

The biggest pain in quantum compiler development: after a compiler pass,
"did the circuit's behavior change?" Currently developers manually inspect
matrices or counts after every change.

New module: `assertions/snapshot.py`

```python
assert_unitary_snapshot(circuit, name, *, update=False, atol=1e-6)
  # First run: saves circuit unitary to .pytest-quantum-snapshots/<name>.npy
  # Subsequent runs: loads and compares — fails if circuit changed
  # --quantum-update-snapshots flag: regenerates all snapshots

assert_distribution_snapshot(counts, name, *, update=False, max_tvd=0.05)
  # Same idea but for shot distributions
```

Why this matters: this is what `syrupy` does for object snapshots and
`pytest-regressions` does for data — but neither understands quantum circuits.
Compiler teams would use this for every PR.

CLI additions:
- `--quantum-update-snapshots` — regenerate all snapshot files

**D. Qiskit Primitives Support** — Qiskit's modern execution model

Qiskit 1.0+ moved to Sampler/Estimator primitives. The old
`backend.run()` API still works but the community is moving to primitives.
We have no assertion support for primitive results.

New fixtures:
```python
@pytest.fixture(scope="session")
def qiskit_sampler():
    from qiskit.primitives import StatevectorSampler
    return StatevectorSampler()

@pytest.fixture(scope="session")
def qiskit_estimator():
    from qiskit.primitives import StatevectorEstimator
    return StatevectorEstimator()
```

New assertions:
```python
assert_sampler_distribution(sampler_result, pub_idx=0,
                             expected_probs=..., significance=0.05)
assert_estimator_close(estimator_result, expected_value, atol=0.1)
```

**E. Improved Failure Messages** — developer experience

Currently only numpy array `==` comparisons get improved messages.
Most framework objects (QuantumCircuit, cirq.Circuit) get generic diffs.

Add `pytest_assertrepr_compare` branches for:
- Qiskit `QuantumCircuit` — show circuit diagram side by side
- `dict` counts comparisons — show TVD and per-key difference table
- numpy arrays with `<` / `>` — show actual vs expected with tolerance

**F. `assert_circuit_is_clifford(circuit)`** — fault-tolerance research

Clifford circuits (H, S, CNOT gates only) are efficiently simulable and
form the basis of error correction research. Researchers need to verify
their circuits stay within the Clifford group after compilation.

```python
assert_circuit_is_clifford(circuit)
  # Verifies circuit only uses Clifford gates
  # Supported: Qiskit (via qiskit.synthesis.clifford), Cirq
```

---

### 3. Infrastructure

**A. Add `conftest.py` snapshot directory to `.gitignore`**
Snapshot files should be committed (they're the ground truth), but the
directory structure needs to be consistent.

**B. Add `--quantum-update-snapshots` to CLI options**

**C. Bump Python minimum to 3.11** — already done, keep it.

**D. Add `py.typed` marker** — make the package PEP 561 compliant for
downstream mypy users.

**E. Parallel test support with `pytest-xdist`**
Session-scoped fixtures need to be safe under `pytest-xdist` parallel
execution. Add `scope="session"` lock or document incompatibility.

---

## v0.2.0 Scope (what we actually build)

In priority order — stop when scope is too large:

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | Wire `--quantum-shots` / `--quantum-significance` to fixtures | Small | High |
| 2 | Fix Cirq shot simulation (`cirq_sampler` fixture) | Small | High |
| 3 | Fix Braket `assert_gate_count` | Small | Medium |
| 4 | Fix PennyLane `assert_circuit_depth` | Small | Medium |
| 5 | Fix PennyLane gate count (no prior execution required) | Small | Medium |
| 6 | Fix cross-framework qubit ordering | Medium | High |
| 7 | Density matrix assertions | Medium | High |
| 8 | Expectation value assertions | Medium | High |
| 9 | Snapshot / golden file testing | Medium | Very High |
| 10 | Qiskit Primitives fixtures + assertions | Medium | High |
| 11 | Improved failure messages | Small | Medium |
| 12 | `assert_circuit_is_clifford` | Small | Medium |
| 13 | `py.typed` marker | Tiny | Low |

All 13 items are in scope for v0.2.0.

---

## What v0.3.0 Gets (not now)

- IBM Quantum real hardware fixture (`ibm_backend`, `--quantum-real`)
- AWS Braket cloud device fixture
- QuTiP support (density matrix evolution, Lindblad master equation)
- Tequila support (quantum chemistry)
- OpenQASM round-trip assertions
- `pytest-xdist` compatibility

---

## File Changes

### New files
```
src/pytest_quantum/assertions/density.py
src/pytest_quantum/assertions/observables.py
src/pytest_quantum/assertions/snapshot.py
tests/test_assertions_density.py
tests/test_assertions_observables.py
tests/test_assertions_snapshot.py
tests/test_integration_cirq_shots.py
tests/test_integration_primitives.py
```

### Modified files
```
src/pytest_quantum/__init__.py            — new exports
src/pytest_quantum/plugin.py             — wire CLI opts, new fixtures, new CLI flag
src/pytest_quantum/assertions/structure.py — PL depth, Braket gate count, PL auto-tape
src/pytest_quantum/converters/to_unitary.py — qubit ordering parameter
src/pytest_quantum/assertions/unitary.py    — use new qubit ordering
pyproject.toml                           — bump to 0.2.0, add py.typed
```

---

## Public API After v0.2.0

```python
from pytest_quantum import (
    # Unitary (unchanged)
    assert_unitary,
    assert_circuits_equivalent,

    # States (unchanged)
    assert_state_fidelity_above,
    assert_states_close,

    # Density matrices (NEW)
    assert_density_matrix_close,
    assert_trace_distance_below,
    assert_purity_above,
    assert_partial_trace_close,

    # Observables / expectation values (NEW)
    assert_expectation_value_close,
    assert_ground_state_energy_close,

    # Distributions (unchanged)
    assert_measurement_distribution,
    assert_counts_close,

    # Primitives (NEW)
    assert_sampler_distribution,
    assert_estimator_close,

    # Structure (unchanged API, wider framework support)
    assert_circuit_depth,
    assert_circuit_width,
    assert_gate_count,
    assert_circuit_is_clifford,   # NEW

    # Snapshots (NEW)
    assert_unitary_snapshot,
    assert_distribution_snapshot,

    # Stats (unchanged)
    min_shots,
    recommended_shots,
    fidelity,
    tvd,
    tvd_from_counts,
    chi_square_test,
)
```

New fixtures:
```
cirq_sampler           — Cirq shot-based simulation
qiskit_sampler         — Qiskit StatevectorSampler (primitives)
qiskit_estimator       — Qiskit StatevectorEstimator (primitives)
```

New markers:
```
@pytest.mark.quantum_snapshot   — marks snapshot tests (for selective update)
```

New CLI options:
```
--quantum-update-snapshots   — regenerate all snapshot golden files
```

---

## Version Bump

`pyproject.toml`: `version = "0.1.1"` → `version = "0.2.0"`

Release command (once all tests pass):
```bash
./release.sh 0.2.0
```
