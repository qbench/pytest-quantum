# Assertions Reference

pytest-quantum gives you **80+ assertion functions** that understand quantum semantics —
statistical noise, global phase, density matrices, hardware constraints, and more.

```python
from pytest_quantum import assert_unitary, assert_measurement_distribution  # and 78 more
```

Every assertion raises `AssertionError` with a detailed human-readable message when it
fails, so you know *exactly* what went wrong in CI.

---

## Quick decision guide

| I want to test... | Use this |
|---|---|
| Gate implements a specific unitary | `assert_unitary` |
| Two circuits do the same thing | `assert_circuits_equivalent` |
| Same circuit, different frameworks | `assert_cross_platform_equivalent` |
| Output state fidelity | `assert_state_fidelity_above` |
| Measurement distribution (shot-noise-safe) | `assert_measurement_distribution` |
| Noisy density matrix | `assert_density_matrix_close` |
| Quantum channel is physically valid | `assert_channel_is_cptp` |
| VQE / QAOA converged | `assert_vqe_converges` |
| Circuit didn't regress | `assert_unitary_snapshot` |
| Circuit uses hardware-native gates only | `assert_gates_in_basis_set` |
| No mid-circuit measurements (hardware constraint) | `assert_no_mid_circuit_measurement` |
| Backend meets quantum volume spec | `assert_quantum_volume` |
| Gate error from RB experiment | `assert_randomized_benchmarking` |
| T1 / T2 / T2* coherence times | `assert_t1_above` / `assert_t2_above` / `assert_t2star_above` |
| XEB fidelity (Google-style benchmark) | `assert_xeb_fidelity_above` |
| VQA circuit is expressive enough | `assert_expressibility_above` |
| No barren plateaus in VQA training | `assert_no_barren_plateau` |
| Error mitigation improves result | `assert_zne_reduces_error` |
| Logical error rate of QEC code | `assert_stim_logical_error_rate_below` |

---

## Unitary & Circuit Equivalence

The most rigorous checks: compare circuits at the matrix level.

### `assert_unitary`

Assert a circuit implements an expected unitary matrix. Global-phase safe.
Works with Qiskit, Cirq, Amazon Braket, PennyLane, Pytket.

```python
from pytest_quantum import assert_unitary
import numpy as np

def test_hadamard():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    assert_unitary(qc, H)                        # e^(iθ)·H also passes ✓
    assert_unitary(qc, H, atol=1e-6)             # tighter tolerance
    assert_unitary(qc, H, allow_global_phase=False)  # strict phase matching
```

### `assert_circuits_equivalent`

Assert two circuits implement the same unitary, up to global phase.
Cross-framework: compare a Qiskit circuit against a Cirq circuit.

```python
def test_same_gate_two_ways():
    from qiskit import QuantumCircuit
    import cirq

    qc = QuantumCircuit(1); qc.h(0); qc.t(0)
    q = cirq.LineQubit.range(1)
    cc = cirq.Circuit(cirq.H(q[0]), cirq.T(q[0]))

    assert_circuits_equivalent(qc, cc)
```

### `assert_transpilation_preserves_semantics`

Assert a transpiled/compiled circuit is semantically equivalent to the original.

```python
from qiskit import transpile
from qiskit_aer import AerSimulator

def test_transpile_preserves():
    compiled = transpile(my_circuit, AerSimulator(), optimization_level=3)
    assert_transpilation_preserves_semantics(my_circuit, compiled)
```

### `assert_cross_platform_equivalent`

Assert two circuits from *different* frameworks implement the same unitary.
Handles qubit-endianness differences between Qiskit (little-endian) and
Cirq/Pytket (big-endian) automatically.

```python
assert_cross_platform_equivalent(qiskit_circuit, cirq_circuit)
assert_qiskit_cirq_equivalent(qiskit_circuit, cirq_circuit)
assert_qiskit_pytket_equivalent(qiskit_circuit, pytket_circuit)
```

---

## State Assertions

### `assert_state_fidelity_above`

Assert `|⟨actual|target⟩|² ≥ threshold`. Accepts numpy arrays or statevectors.

```python
from pytest_quantum import assert_state_fidelity_above

def test_rx_pi():
    import numpy as np
    sv = run_rx_pi_circuit()     # should be close to -i|1⟩
    assert_state_fidelity_above(sv, np.array([0, -1j]), threshold=0.999)
```

### `assert_states_close`

Elementwise comparison up to global phase and tolerance.

```python
assert_states_close(actual, target, atol=1e-6)
assert_states_close(actual, target, atol=1e-4, allow_global_phase=False)
```

### `assert_normalized`

Assert a statevector has unit norm `‖ψ‖₂ = 1`.

```python
assert_normalized(statevector)
assert_normalized(statevector, atol=1e-8)
```

---

## Measurement Distributions

### `assert_measurement_distribution` ⭐

The workhorse of quantum testing. Uses a chi-square goodness-of-fit test
so it **never fails due to shot noise**, only when the distribution is
genuinely wrong.

```python
from pytest_quantum import assert_measurement_distribution

def test_bell_state(aer_simulator):
    counts = run_bell_circuit(aer_simulator, shots=2000)
    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
        significance=0.01,   # p-value threshold (default 0.01)
    )
```

### `assert_counts_close`

Assert two count dicts are close using Total Variation Distance (TVD).

```python
assert_counts_close(counts_a, counts_b, max_tvd=0.05)
```

---

## Density Matrix Assertions

For noisy simulation and open quantum systems.

```python
from pytest_quantum import (
    assert_density_matrix_close,
    assert_trace_distance_below,
    assert_purity_above,
    assert_partial_trace_close,
)

def test_noisy_channel():
    rho = run_noisy_circuit()              # density matrix from noise model

    assert_density_matrix_close(rho, ideal_rho, atol=0.01)
    assert_trace_distance_below(rho, ideal_rho, max_distance=0.05)
    assert_purity_above(rho, min_purity=0.90)   # not too noisy

def test_bell_entanglement():
    rho_AB = get_bell_density_matrix()
    rho_A_expected = np.eye(2) / 2            # maximally mixed subsystem
    assert_partial_trace_close(rho_AB, keep_qubits=[0], expected=rho_A_expected)
```

---

## Quantum Channel Assertions

```python
from pytest_quantum import (
    assert_hermitian, assert_positive_semidefinite, assert_commutes_with,
    assert_channel_is_cptp, assert_process_fidelity_above, assert_noise_fidelity_above,
)

def test_custom_channel(kraus_ops):
    assert_channel_is_cptp(kraus_ops)              # completely positive + trace preserving
    assert_process_fidelity_above(kraus_ops, ideal_channel, threshold=0.99)

def test_hamiltonian(H):
    assert_hermitian(H)
    assert_positive_semidefinite(H + np.eye(H.shape[0]) * np.abs(np.min(np.linalg.eigvalsh(H))))

def test_pauli_commutation():
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    # X and Y anti-commute — this should raise AssertionError:
    # assert_commutes_with(X, Y)
```

---

## Noise Channel Assertions

Verify noise model Kraus operators match physical channels.

```python
from pytest_quantum import (
    assert_depolarizing_channel,
    assert_amplitude_damping_channel,
    assert_dephasing_channel,
    assert_no_leakage,
    assert_channel_preserves_trace,
    assert_channel_diamond_norm_below,
)

def test_noise_model(noise_kraus_ops):
    assert_channel_preserves_trace(noise_kraus_ops)      # always a sanity check
    assert_depolarizing_channel(noise_kraus_ops, error_rate=0.01, atol=1e-4)
    assert_no_leakage(noise_kraus_ops, computational_dim=2)

def test_channels_are_close(kraus_a, kraus_b):
    # Diamond norm: operational distance between two quantum channels
    assert_channel_diamond_norm_below(kraus_a, kraus_b, max_norm=0.05)
```

---

## Observable & Expectation Value

```python
from pytest_quantum import (
    assert_expectation_value_close,
    assert_ground_state_energy_close,
    assert_vqe_converges,
    assert_cost_decreases,
)

def test_z_expectation(qiskit_estimator):
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(1)        # |0⟩ → ⟨Z⟩ = 1.0
    result = qiskit_estimator.run([(qc, SparsePauliOp("Z"))]).result()
    assert_estimator_close(result, expected=1.0, atol=0.01)

def test_h2_ground_state():
    energy = run_vqe_h2()
    assert_ground_state_energy_close(energy, expected_energy=-1.137, atol=0.05)

def test_vqe_converges():
    assert_vqe_converges(vqe_fn, h2_hamiltonian, target_energy=-1.137, atol=0.05)
    assert_cost_decreases(cost_fn, initial_params)
```

---

## Circuit Structure

Static checks on gate counts, depth, and composition. No simulation needed.

```python
from pytest_quantum import (
    assert_circuit_depth, assert_circuit_width, assert_gate_count,
    assert_gates_in_basis_set, assert_circuit_is_clifford,
    assert_has_diagram, assert_no_mid_circuit_measurement,
)

def test_compiled_circuit(backend):
    from qiskit import transpile
    qc = QuantumCircuit(3); qc.h(0); qc.cx(0,1); qc.cx(1,2)
    compiled = transpile(qc, backend, basis_gates=["cx","u3"])

    assert_circuit_depth(compiled, max_depth=6)
    assert_circuit_width(compiled, expected_qubits=3)
    assert_gate_count(compiled, "cx", expected=2)
    assert_gates_in_basis_set(compiled, {"cx", "u3"})
    assert_circuit_is_clifford(qc)                    # original is Clifford

def test_no_mid_circuit():
    # Fail fast before submitting to hardware that doesn't support reset
    assert_no_mid_circuit_measurement(circuit)
```

---

## Transpilation & Compilation

```python
from pytest_quantum import (
    assert_transpilation_equivalent,
    assert_transpilation_depth_below,
    assert_gate_count_after_transpilation,
)

def test_transpilation(aer_simulator):
    assert_transpilation_equivalent(my_circuit, aer_simulator)
    assert_transpilation_depth_below(my_circuit, aer_simulator, max_depth=20)
    assert_gate_count_after_transpilation(my_circuit, aer_simulator, "cx", expected=2)
```

---

## Sweeps / Parametrised Circuits

```python
from pytest_quantum import (
    assert_circuit_sweep, assert_circuit_sweep_states,
    assert_parametrized_unitary_continuous,
)

def test_ry_sweep(aer_simulator):
    from qiskit.circuit import Parameter, QuantumCircuit

    theta = Parameter("θ")
    qc = QuantumCircuit(1); qc.ry(theta, 0); qc.measure_all()

    assert_circuit_sweep(
        qc,
        param_values=[{"θ": 0}, {"θ": np.pi}, {"θ": np.pi/2}],
        expected_probs_list=[{"0": 1.0}, {"1": 1.0}, {"0": 0.5, "1": 0.5}],
        backend=aer_simulator,
    )
    assert_parametrized_unitary_continuous(qc, param_range=(0, 2*np.pi))
```

---

## Entanglement

```python
from pytest_quantum import (
    assert_entanglement_entropy_below,
    assert_bloch_sphere_close,
    assert_schmidt_rank_at_most,
)

def test_bell_entanglement():
    sv = get_bell_statevector()
    # Bell state has maximum entanglement entropy = 1 ebit
    assert_entanglement_entropy_below(sv, partition=[0], max_entropy=1.01)
    assert_schmidt_rank_at_most(sv, partition=[0], max_rank=2)

def test_single_qubit_bloch():
    sv = get_plus_state()   # |+⟩ = (|0⟩+|1⟩)/√2
    # θ=π/2, φ=0 on Bloch sphere
    assert_bloch_sphere_close(sv, theta=np.pi/2, phi=0.0, atol=0.01)
```

---

## Information Theory

```python
from pytest_quantum import assert_hellinger_close, assert_kl_divergence_below, assert_cross_entropy_below

# Hellinger distance ∈ [0,1]: 0=identical, 1=disjoint
assert_hellinger_close(counts_a, counts_b, max_distance=0.1)

# KL divergence (asymmetric)
assert_kl_divergence_below(counts, expected_probs, max_kl=0.1)

# Cross-entropy
assert_cross_entropy_below(counts, expected_probs, max_ce=1.0)
```

---

## Snapshots / Golden Files

Regression testing for circuits: run once to create the golden file,
then future runs compare against it.

```python
from pytest_quantum import assert_unitary_snapshot, assert_distribution_snapshot

def test_no_regression():
    # First run: writes .snapshots/my_circuit.npy
    # Subsequent runs: compares against it
    assert_unitary_snapshot(qc, "my_circuit")

def test_distribution_no_regression(aer_simulator):
    counts = run_circuit(aer_simulator, shots=2000)
    assert_distribution_snapshot(counts, "my_dist_snapshot", max_tvd=0.05)
```

Regenerate golden files: `pytest --quantum-update-snapshots`

---

## QASM Round-Trips

```python
from pytest_quantum import assert_qasm_roundtrip, assert_qasm2_roundtrip

def test_qasm3_roundtrip(qc):
    assert_qasm_roundtrip(qc)       # OpenQASM 3 (Qiskit) or Cirq JSON

def test_qasm2_roundtrip(qc):
    assert_qasm2_roundtrip(qc)      # OpenQASM 2.0 (Qiskit only)
```

---

## QEC / Stim

```python
from pytest_quantum import (
    assert_stim_logical_error_rate_below,
    assert_stim_detector_error_rate_below,
    assert_stabilizer_state,
)

def test_surface_code(stim_sampler):
    import stim
    code = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    assert_stim_logical_error_rate_below(code, max_error_rate=0.01, shots=100_000)

def test_bell_stabilizer():
    sv = get_bell_statevector()
    assert_stabilizer_state(sv, ["XX", "ZZ"])   # Bell state stabilizers
```

---

## Benchmarking *(v1.0.0)*

### Quantum Volume

```python
def test_qv16(aer_simulator):
    qv = assert_quantum_volume(aer_simulator, target_qv=16, num_trials=100)
    print(f"Measured QV: {qv}")   # 16
```

### Randomized Benchmarking

```python
def test_1q_rb(aer_simulator):
    result = assert_randomized_benchmarking(
        aer_simulator, qubit=0,
        clifford_lengths=[1, 10, 20, 50, 100],
        num_sequences=20,
        min_fidelity_per_clifford=0.999,
    )
    print(f"Average gate fidelity: {result['fidelity']:.5f}")
```

### T1 / T2 / T2* Characterisation

```python
@pytest.mark.quantum_real
def test_coherence_times(ibm_backend):
    t1   = assert_t1_above(ibm_backend, qubit=0, target_t1_us=50.0)
    t2   = assert_t2_above(ibm_backend, qubit=0, target_t2_us=30.0)   # Hahn echo
    t2s  = assert_t2star_above(ibm_backend, qubit=0, target_t2star_us=20.0)  # Ramsey
    print(f"T1={t1:.1f}µs  T2={t2:.1f}µs  T2*={t2s:.1f}µs")
```

### Interleaved RB (single gate fidelity)

```python
def test_cx_gate_fidelity(aer_simulator):
    from qiskit import QuantumCircuit
    x_gate = QuantumCircuit(1); x_gate.x(0)

    result = assert_interleaved_rb(
        aer_simulator, qubit=0,
        gate_name="X", gate_circuit=x_gate,
        min_gate_fidelity=0.999,
    )
    print(f"X gate fidelity: {result['fidelity']:.5f}")
```

---

## Quantum ML *(v1.0.0)*

### Cross-Entropy Benchmarking (XEB)

```python
def test_xeb(aer_simulator):
    fidelity = assert_xeb_fidelity_above(
        aer_simulator, num_qubits=4, target_fidelity=0.9,
        num_circuits=50, depth=20,
    )
    print(f"XEB fidelity: {fidelity:.3f}")
```

### Expressibility

Is your variational ansatz expressive enough to represent the solution?

```python
def test_ansatz_expressibility():
    def my_ansatz(params):
        qc = QuantumCircuit(2)
        qc.ry(params[0], 0); qc.ry(params[1], 1); qc.cx(0, 1)
        return qc

    score = assert_expressibility_above(
        my_ansatz, num_qubits=2, num_params=2, target_expressibility=0.3,
    )
    print(f"Expressibility score: {score:.3f}  (1.0 = Haar-random)")
```

### Entanglement Capability

```python
def test_ansatz_entanglement():
    q = assert_entanglement_capability_above(
        my_ansatz, num_qubits=2, num_params=4, target_capability=0.3,
    )
    print(f"Meyer-Wallach Q: {q:.3f}  (1.0 = maximally entangled)")
```

### Barren Plateau Detection

```python
def test_no_barren_plateau():
    var = assert_no_barren_plateau(
        deep_ansatz, num_qubits=8, num_params=64,
        min_gradient_variance=1e-4,
    )
    print(f"Gradient variance: {var:.2e}  (higher = trainable)")
```

---

## Hardware Assertions *(v1.0.0)*

```python
@pytest.mark.quantum_real
def test_backend_ready(ibm_backend):
    assert_backend_calibration(ibm_backend, min_t1_us=30.0, min_cx_fidelity=0.99)
    assert_circuit_fits_backend(my_circuit, ibm_backend)   # qubit count + connectivity
    assert_mirror_fidelity(ibm_backend, qubit=0, target_fidelity=0.95)
```

---

## Error Mitigation (Mitiq) *(v1.0.0)*

```python
from pytest_quantum import (
    assert_zne_expectation_close, assert_zne_reduces_error,
    assert_cdr_reduces_error, assert_pec_reduces_error,
    assert_pec_expectation_close, assert_error_mitigation_benchmark,
)

def test_zne_improves_result():
    assert_zne_reduces_error(
        circuit, observable, noisy_val=0.85, ideal_val=1.0,
    )
    assert_zne_expectation_close(
        circuit, observable, expected=1.0, atol=0.05,
    )

def test_all_methods():
    assert_error_mitigation_benchmark(
        circuit, observable, methods=["zne", "cdr", "pec"]
    )
```

---

## Primitives (Qiskit 1.0+)

```python
from pytest_quantum import assert_sampler_distribution, assert_estimator_close

def test_sampler(qiskit_sampler):
    result = qiskit_sampler.run([(bell_circuit,)]).result()
    assert_sampler_distribution(result, {"00": 0.5, "11": 0.5})

def test_estimator(qiskit_estimator):
    result = qiskit_estimator.run([(ground_state, SparsePauliOp("Z"))]).result()
    assert_estimator_close(result, expected=1.0, atol=0.01)
```
