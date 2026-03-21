# pytest-quantum v0.3.0 — Plan

## What v0.2.0 Delivered

v0.2.0 fixed several broken behaviors and added four major feature areas:

- **Bug fixes**: Wired `--quantum-shots` and `--quantum-significance` CLI flags,
  fixed Cirq shot simulation (added `cirq_sampler` fixture), fixed Braket
  `assert_gate_count`, fixed PennyLane `assert_circuit_depth` and gate count
  (no prior execution required), fixed cross-framework multi-qubit unitary
  comparison (Qiskit little-endian ↔ Cirq big-endian qubit ordering).
- **Density matrix assertions**: `assert_density_matrix_close`,
  `assert_trace_distance_below`, `assert_purity_above`, `assert_partial_trace_close`.
- **Expectation value assertions**: `assert_expectation_value_close`,
  `assert_ground_state_energy_close`.
- **Snapshot / golden file testing**: `assert_unitary_snapshot`,
  `assert_distribution_snapshot`, `--quantum-update-snapshots` CLI flag.
- **Qiskit Primitives**: `qiskit_sampler`, `qiskit_estimator` fixtures,
  `assert_sampler_distribution`, `assert_estimator_close` assertions.
- **Clifford circuit check**: `assert_circuit_is_clifford` (Qiskit + Cirq).
- **Improved assertion messages** for `QuantumCircuit` and count dict comparisons.

---

## What v0.3.0 Adds

### 1. Quantum Channel / Operator Assertions (pure numpy)

New module `assertions/channels.py`:

```python
assert_hermitian(matrix, *, atol=1e-8)
assert_positive_semidefinite(matrix, *, atol=1e-8)
assert_commutes_with(op_a, op_b, *, atol=1e-8)
assert_channel_is_cptp(kraus_ops, *, atol=1e-8)
assert_process_fidelity_above(channel_a, channel_b, threshold=0.99, *, atol=1e-8)
assert_noise_fidelity_above(noisy_dm, ideal_state, threshold=0.99)
```

No quantum SDK required. Supports Qiskit `process_fidelity` delegation for
Qiskit channel objects.

### 2. Entanglement Assertions (pure numpy)

New module `assertions/entanglement.py`:

```python
assert_entanglement_entropy_below(statevector, partition, max_entropy, *, n_qubits=None)
assert_bloch_sphere_close(statevector, expected_theta, expected_phi, *, atol=0.1)
assert_schmidt_rank_at_most(statevector, partition, max_rank, *, n_qubits=None, tol=1e-10)
```

Uses `_partial_trace` from `assertions/density.py`.

### 3. Information-Theoretic Distribution Assertions (pure numpy)

New module `assertions/information.py`:

```python
assert_hellinger_close(counts_a, counts_b, *, max_distance=0.1)
assert_kl_divergence_below(counts, expected_probs, *, max_kl=0.1)
assert_cross_entropy_below(counts, expected_probs, *, max_ce=1.0)
```

### 4. OpenQASM Round-Trip Assertions

New module `assertions/qasm.py`:

```python
assert_qasm_roundtrip(circuit, *, atol=1e-6, allow_global_phase=True)
```

Supports Qiskit (via `qiskit.qasm3`) and Cirq (via `cirq.contrib.qasm_import`).

### 5. Pytket Framework Support

- `converters/to_unitary.py`: `_from_pytket()`, `_is_pytket()` helper.
- `assertions/structure.py`: Pytket depth, width, gate count, Clifford check.
- `assertions/unitary.py`: Pytket big-endian qubit ordering (same as Cirq).
- Cross-framework `assert_circuits_equivalent` for Pytket ↔ Qiskit.

### 6. Expanded Clifford Support

`assert_circuit_is_clifford` now supports:
- **Amazon Braket** — checks against `_CLIFFORD_BRAKET` gate set.
- **PennyLane** — checks tape operations against `_CLIFFORD_PENNYLANE`.
- **Pytket** — delegates to `pytket.tableau.UnitaryTableau`.

### 7. New Fixtures (plugin.py)

```python
pytket_circuit_factory  # returns pytket.circuit.Circuit class
stim_sampler            # stim stabilizer circuit sampler → count dict
quantum_benchmark       # wraps pytest-benchmark or simple timer
shot_budget             # ShotBudget class to track shots per test
```

### 8. Enhanced Assertion Reporter

`pytest_assertrepr_compare` now handles:
- Qiskit `QuantumCircuit` — shows circuit diagrams side-by-side.
- `dict` count distributions — shows TVD and per-key probability diff table.
- `numpy.ndarray` (original) — fidelity and max-diff.

### 9. New `quantum_backends` Marker

```python
@pytest.mark.quantum_backends(backends)
```

### 10. Updated pyproject.toml

- Version bumped to `0.3.0`.
- New optional deps: `pytket = ["pytket>=1.30"]`, `stim = ["stim>=1.13"]`.
- Both added to `all` group and `dev` group.

---

## File Changes

### New files
```
src/pytest_quantum/assertions/channels.py
src/pytest_quantum/assertions/entanglement.py
src/pytest_quantum/assertions/information.py
src/pytest_quantum/assertions/qasm.py
tests/test_assertions_channels.py
tests/test_assertions_entanglement.py
tests/test_assertions_information.py
tests/test_integration_braket.py
tests/test_integration_pytket.py
tests/test_integration_stim.py
```

### Modified files
```
src/pytest_quantum/__init__.py        — version 0.3.0, new exports, new assert_rewrite calls
src/pytest_quantum/plugin.py          — new fixtures, enhanced assertrepr_compare, new marker
src/pytest_quantum/assertions/structure.py  — Pytket + Braket + PennyLane Clifford
src/pytest_quantum/assertions/unitary.py    — Pytket qubit ordering
src/pytest_quantum/converters/to_unitary.py — _from_pytket(), _is_pytket()
pyproject.toml                        — version 0.3.0, pytket/stim deps
tests/test_plugin.py                  — version string updated to 0.3.0
```

---

## Public API After v0.3.0

```python
from pytest_quantum import (
    # Unitary
    assert_unitary,
    assert_circuits_equivalent,

    # States
    assert_state_fidelity_above,
    assert_states_close,

    # Density matrices (v0.2.0)
    assert_density_matrix_close,
    assert_trace_distance_below,
    assert_purity_above,
    assert_partial_trace_close,

    # Observables (v0.2.0)
    assert_expectation_value_close,
    assert_ground_state_energy_close,

    # Distributions
    assert_measurement_distribution,
    assert_counts_close,

    # Primitives (v0.2.0)
    assert_sampler_distribution,
    assert_estimator_close,

    # Structure
    assert_circuit_depth,
    assert_circuit_width,
    assert_gate_count,
    assert_circuit_is_clifford,

    # Snapshots (v0.2.0)
    assert_unitary_snapshot,
    assert_distribution_snapshot,

    # Channels / operators (v0.3.0)
    assert_hermitian,
    assert_positive_semidefinite,
    assert_commutes_with,
    assert_channel_is_cptp,
    assert_process_fidelity_above,
    assert_noise_fidelity_above,

    # Entanglement (v0.3.0)
    assert_entanglement_entropy_below,
    assert_bloch_sphere_close,
    assert_schmidt_rank_at_most,

    # Information theory (v0.3.0)
    assert_hellinger_close,
    assert_kl_divergence_below,
    assert_cross_entropy_below,

    # QASM round-trip (v0.3.0)
    assert_qasm_roundtrip,

    # Stats
    min_shots,
    recommended_shots,
    fidelity,
    tvd,
    tvd_from_counts,
    chi_square_test,
)
```

Fixtures:
```
aer_simulator              — Qiskit AerSimulator (shots)
aer_statevector_simulator  — Qiskit AerSimulator (statevector)
aer_noise_simulator        — factory: make_simulator(error_rate)
qiskit_sampler             — Qiskit StatevectorSampler (primitives)
qiskit_estimator           — Qiskit StatevectorEstimator (primitives)
cirq_simulator             — cirq.Simulator
cirq_sampler               — Cirq shot sampler → count dict
braket_simulator           — Braket LocalSimulator
graphix_backend            — graphix pattern runner
pennylane_device           — factory: make_device(wires, shots=None)
pytket_circuit_factory     — pytket.circuit.Circuit class (v0.3.0)
stim_sampler               — stim circuit sampler → count dict (v0.3.0)
quantum_benchmark          — benchmark wrapper (v0.3.0)
shot_budget                — ShotBudget class (v0.3.0)
quantum_shots              — --quantum-shots CLI value
quantum_significance       — --quantum-significance CLI value
```

Markers:
```
@pytest.mark.quantum            — generic quantum test
@pytest.mark.quantum_slow       — skipped unless --quantum-slow
@pytest.mark.quantum_snapshot   — snapshot tests
@pytest.mark.quantum_backends   — specify target backends (v0.3.0)
@pytest.mark.shots(n)           — override shot count
@pytest.mark.significance(p)    — override p-value threshold
```

CLI options:
```
--quantum-slow              — run quantum_slow tests
--quantum-shots N           — override default shot count
--quantum-significance P    — override default p-value threshold
--quantum-update-snapshots  — regenerate snapshot golden files
```

---

## Version Bump

`pyproject.toml`: `version = "0.2.0"` → `version = "0.3.0"`
