# Cookbook

Self-contained, copy-paste recipes for common quantum testing patterns.
Each recipe is a complete, runnable test.

---

## Recipe 1: Testing a single-qubit gate

```python
"""test_gates.py — assert gate unitaries are correct."""
import numpy as np
import pytest
from pytest_quantum import assert_unitary, assert_state_fidelity_above


def test_hadamard():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    assert_unitary(qc, H)


def test_t_gate():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.t(0)
    T = np.diag([1, np.exp(1j * np.pi / 4)])
    assert_unitary(qc, T)


def test_rx_at_pi():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(1)
    qc.rx(np.pi, 0)                              # RX(π) ≈ -iX: |0⟩ → -i|1⟩
    sv = Statevector(qc).data
    assert_state_fidelity_above(sv, np.array([0, -1j]), threshold=0.9999)
```

---

## Recipe 2: Shot-noise-safe Bell state test

```python
"""test_bell.py — the classic Bell state, done correctly."""
from pytest_quantum import assert_measurement_distribution, assert_circuits_equivalent
import numpy as np


def test_bell_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=2000) \
                          .result().get_counts()

    # Chi-square test — never fails due to shot noise alone
    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
        significance=0.01,
    )


def test_bell_cirq_vs_qiskit():
    """Same circuit in two frameworks should give the same unitary."""
    from qiskit import QuantumCircuit
    import cirq

    qk = QuantumCircuit(2); qk.h(0); qk.cx(0, 1)

    q = cirq.LineQubit.range(2)
    cq = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))

    assert_circuits_equivalent(qk, cq)
```

---

## Recipe 3: Noisy simulation with density matrices

```python
"""test_noisy.py — test circuits under realistic noise."""
from pytest_quantum import (
    assert_density_matrix_close,
    assert_purity_above,
    assert_trace_distance_below,
)


def test_noisy_rx(aer_noise_simulator):
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import DensityMatrix

    sim = aer_noise_simulator(error_rate=0.01)
    qc = QuantumCircuit(1); qc.rx(np.pi/2, 0)
    transpiled = transpile(qc, sim)

    # Get density matrix from 10000 shots
    result = sim.run(transpiled, shots=10000).result()
    rho = DensityMatrix(result.get_statevector()).data

    # Even with 1% noise, purity should be > 0.9
    assert_purity_above(rho, min_purity=0.90)

    # Should be close to ideal |+y⟩ state
    ideal = DensityMatrix.from_label("+").data
    assert_trace_distance_below(rho, ideal, max_distance=0.05)
```

---

## Recipe 4: Snapshot regression testing

```python
"""test_regression.py — catch accidental circuit changes."""
from pytest_quantum import assert_unitary_snapshot, assert_distribution_snapshot


def test_my_circuit_no_regression():
    """First run: creates .snapshots/my_algo.npy. All future runs compare."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.s(0)
    assert_unitary_snapshot(qc, "my_algo")


def test_distribution_no_regression(aer_simulator):
    from qiskit import QuantumCircuit, transpile
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1); qc.measure_all()
    counts = aer_simulator.run(transpile(qc, aer_simulator), shots=2000) \
                          .result().get_counts()
    assert_distribution_snapshot(counts, "bell_dist", max_tvd=0.05)
```

Regenerate all snapshots: `pytest --quantum-update-snapshots`

---

## Recipe 5: VQE testing end-to-end

```python
"""test_vqe.py — validate a variational quantum eigensolver."""
import numpy as np
from pytest_quantum import (
    assert_circuit_sweep_states,
    assert_vqe_converges,
    assert_ground_state_energy_close,
)


def test_ry_sweep(aer_simulator):
    from qiskit.circuit import Parameter, QuantumCircuit

    theta = Parameter("θ")
    qc = QuantumCircuit(1); qc.ry(theta, 0)

    # θ=0 → |0⟩, θ=π → |1⟩
    assert_circuit_sweep_states(
        qc,
        param_values=[{theta: 0}, {theta: np.pi}],
        expected_states=[np.array([1, 0]), np.array([0, 1])],
        backend=aer_simulator,
        atol=0.01,
    )


def build_h2_ansatz():
    from qiskit.circuit import Parameter, QuantumCircuit
    theta = Parameter("θ")
    qc = QuantumCircuit(2)
    qc.x(0); qc.ry(theta, 1); qc.cx(1, 0)
    return qc, theta


def test_h2_vqe_converges(qiskit_estimator):
    from qiskit.quantum_info import SparsePauliOp
    H = SparsePauliOp.from_list([("ZZ", 0.5), ("XX", -0.5)])

    def run_vqe():
        from scipy.optimize import minimize
        qc, theta = build_h2_ansatz()

        def cost(params):
            bound = qc.assign_parameters({theta: params[0]})
            result = qiskit_estimator.run([(bound, H)]).result()
            return float(result[0].data.evs)

        res = minimize(cost, x0=[0.0], method="COBYLA")
        return res.fun

    energy = run_vqe()
    assert_ground_state_energy_close(energy, expected_energy=-1.0, atol=0.1)
```

---

## Recipe 6: Multi-framework parametrised test

```python
"""test_multi_fw.py — run the same test on every installed framework."""
import pytest
from pytest_quantum import assert_unitary
import numpy as np


@pytest.mark.quantum_backends("qiskit", "cirq", "pennylane")
def test_hadamard_gate(quantum_backend_name):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    if quantum_backend_name == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1); qc.h(0)
        assert_unitary(qc, H)

    elif quantum_backend_name == "cirq":
        import cirq
        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.H(q[0]))
        assert_unitary(circuit, H)

    elif quantum_backend_name == "pennylane":
        import pennylane as qml
        import numpy as np

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.Hadamard(0)
            return qml.state()

        sv = np.array(circuit())
        expected = H[:, 0]  # H|0⟩ = |+⟩
        from pytest_quantum import assert_state_fidelity_above
        assert_state_fidelity_above(sv, expected, threshold=0.999)
```

---

## Recipe 7: QEC with Stim

```python
"""test_qec.py — validate a quantum error correcting code."""
from pytest_quantum import (
    assert_stim_logical_error_rate_below,
    assert_stabilizer_state,
)


def test_repetition_code():
    import stim

    # Distance-3 repetition code, 3 rounds
    code = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
    )
    # Physical error rate 0.1% → logical error rate should be < 0.01%
    assert_stim_logical_error_rate_below(code, max_error_rate=0.0001, shots=100_000)


def test_bell_stabilizers():
    import numpy as np
    # Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 is stabilised by XX and ZZ
    sv = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert_stabilizer_state(sv, stabilizers=["XX", "ZZ"])
```

---

## Recipe 8: Randomised benchmarking

```python
"""test_rb.py — characterise gate quality via RB."""
from pytest_quantum import assert_randomized_benchmarking, assert_quantum_volume


def test_1q_clifford_fidelity(aer_simulator):
    result = assert_randomized_benchmarking(
        aer_simulator,
        qubit=0,
        clifford_lengths=[1, 10, 20, 50, 100],
        num_sequences=20,
        shots=1024,
        min_fidelity_per_clifford=0.999,
    )
    print(f"Average gate fidelity: {result['fidelity']:.5f}")
    print(f"Decay parameter p: {result['decay_rate']:.5f}")


def test_qv4(aer_simulator):
    qv = assert_quantum_volume(
        aer_simulator,
        target_qv=4,
        num_trials=20,
        shots=1024,
    )
    assert qv >= 4
```

---

## Recipe 9: Error mitigation with Mitiq

```python
"""test_mitiq.py — verify ZNE improves expectation values."""
pytest.importorskip("mitiq", reason="mitiq not installed")

from pytest_quantum import assert_zne_reduces_error, assert_zne_expectation_close


def test_zne_on_bell(aer_noise_simulator):
    from qiskit import QuantumCircuit
    import numpy as np

    noisy_sim = aer_noise_simulator(error_rate=0.02)
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)

    def executor(circuit):
        from qiskit import transpile
        transpiled = transpile(circuit, noisy_sim, optimization_level=0)
        result = noisy_sim.run(transpiled, shots=1024).result()
        counts = result.get_counts()
        total = sum(counts.values())
        return (counts.get("00", 0) + counts.get("11", 0)) / total - 0.5

    assert_zne_expectation_close(
        qc, executor, expected=0.5, atol=0.1,
    )
```

---

## Recipe 10: Real hardware test (IBM Quantum)

```python
"""test_hardware.py — run on real IBM Quantum hardware."""
import pytest
from pytest_quantum import (
    assert_circuit_fits_backend,
    assert_backend_calibration,
    assert_measurement_distribution,
    assert_real_counts_close,
)


@pytest.mark.quantum_real
def test_bell_on_ibm(ibm_backend):
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import SamplerV2

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)

    # Static check: does the circuit fit the backend?
    assert_circuit_fits_backend(qc, ibm_backend)
    assert_backend_calibration(ibm_backend, min_t1_us=20.0)

    # Run on real hardware
    transpiled = transpile(qc, ibm_backend, optimization_level=3)
    transpiled.measure_all()

    sampler = SamplerV2(ibm_backend)
    job = sampler.run([transpiled], shots=1024)

    # Hardware is noisy — use wider tolerance
    counts = dict(job.result()[0].data.meas.get_counts())
    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
        significance=0.001,   # looser for hardware noise
    )
```

---

## Recipe 11: Shot budget enforcement

```python
"""test_budget.py — cap total shots in an integration test."""
from pytest_quantum import assert_measurement_distribution


def test_budget_enforced(aer_simulator, shot_budget):
    budget = shot_budget(max_shots=5000)

    for gate in ["h", "x", "y", "z"]:
        shots = budget.allocate(1000)
        counts = run_single_qubit_gate(aer_simulator, gate, shots=shots)
        assert_measurement_distribution(counts, expected_probs_for(gate))

    assert budget.remaining == 1000
    print(f"Used {budget.used} / {budget.max_shots} shots")
```
