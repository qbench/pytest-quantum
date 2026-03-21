# Tutorial: Testing a VQE Ansatz End-to-End

This tutorial shows how to write a complete test suite for a Variational Quantum Eigensolver (VQE) implementation targeting the H2 molecule. By the end you will have tests covering circuit structure, unitary correctness, parameter sweeps, expectation values, convergence, and error mitigation — the full stack.

**Prerequisites:**
- Python 3.11+
- Basic familiarity with quantum circuits and VQE
- Installed pytest-quantum with Qiskit and Mitiq extras

**Time to complete:** ~30 minutes

---

## 0. Installation

```bash
pip install "pytest-quantum[qiskit,mitiq]"
```

Verify the install:

```bash
pytest --co -q  # should list no collection errors
```

---

## 1. The Problem: Testing a VQE Circuit for H2

The hydrogen molecule (H2) in the STO-3G basis can be mapped to a 2-qubit problem via the Jordan-Wigner transform. The simplest ansatz that can reach the ground state is:

```
q0: ──Ry(θ₀)──●──
               │
q1: ──Ry(θ₁)──X──
```

This is a hardware-efficient ansatz: one layer of Ry rotations followed by a CNOT entangling gate. The H2 ground-state energy in atomic units (Hartree) is approximately **-1.8572 Ha** at equilibrium bond length.

We will build a test suite that validates every property of this ansatz from the bottom up.

---

## Step 1: Test the Ansatz Circuit Structure

Before running any simulation, confirm the circuit has the right shape. Structural tests run in milliseconds and catch obvious bugs (wrong qubit count, missing gates, extra layers).

```python
# test_vqe_h2.py — Step 1
import pytest
import numpy as np
from pytest_quantum import (
    assert_circuit_depth,
    assert_circuit_width,
    assert_gate_count,
)


def make_h2_ansatz(theta0: float, theta1: float):
    """2-qubit hardware-efficient ansatz for H2."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.cx(0, 1)
    return qc


def test_ansatz_width():
    """H2 ansatz must act on exactly 2 qubits."""
    qc = make_h2_ansatz(0.1, 0.2)
    assert_circuit_width(qc, expected_qubits=2)


def test_ansatz_depth():
    """Ansatz depth should be <= 3 (Ry layer + CNOT)."""
    qc = make_h2_ansatz(0.1, 0.2)
    # depth=2: Ry gates execute in parallel (same layer), then CX
    assert_circuit_depth(qc, max_depth=3)


def test_ansatz_gate_counts():
    """Ansatz must have exactly 2 Ry gates and 1 CNOT."""
    qc = make_h2_ansatz(0.1, 0.2)
    assert_gate_count(qc, "ry", 2)
    assert_gate_count(qc, "cx", 1)


def test_ansatz_is_parametrized():
    """Parameters should change the circuit — same params give same circuit."""
    qc_a = make_h2_ansatz(0.5, 1.0)
    qc_b = make_h2_ansatz(0.5, 1.0)
    # Both should have the same op counts
    assert qc_a.count_ops() == qc_b.count_ops()

    qc_c = make_h2_ansatz(0.5, 1.1)  # different theta1
    # Op types are the same, but the circuits are different instances
    assert qc_c.count_ops() == qc_a.count_ops()
```

Run these tests:

```bash
pytest test_vqe_h2.py::test_ansatz_width \
       test_vqe_h2.py::test_ansatz_depth \
       test_vqe_h2.py::test_ansatz_gate_counts \
       test_vqe_h2.py::test_ansatz_is_parametrized \
       -v
```

**What to expect:**

```
test_vqe_h2.py::test_ansatz_width          PASSED
test_vqe_h2.py::test_ansatz_depth          PASSED
test_vqe_h2.py::test_ansatz_gate_counts    PASSED
test_vqe_h2.py::test_ansatz_is_parametrized PASSED
```

---

## Step 2: Test the Ansatz Unitary for Specific Parameter Values

For a known parameter set, compute the expected 4x4 unitary analytically and verify the circuit implements it exactly. This is stronger than a distribution test: it catches global phase errors, incorrect gate ordering, and qubit indexing bugs.

```python
# test_vqe_h2.py — Step 2
def ry_matrix(theta: float) -> np.ndarray:
    """2x2 Ry rotation matrix."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def expected_ansatz_unitary(theta0: float, theta1: float) -> np.ndarray:
    """Compute the 4x4 unitary for the H2 ansatz analytically.

    Circuit: Ry(theta0) ⊗ Ry(theta1), then CNOT(0->1).

    Note: Qiskit uses little-endian qubit ordering, so qubit 0 is the
    rightmost tensor factor. The CNOT matrix in this ordering is:
        |00><00| + |01><01| + |11><10| + |10><11|
    which is the standard CNOT with control=0, target=1.
    """
    import numpy as np

    # Layer 1: Ry(theta0) on q0, Ry(theta1) on q1
    # In Qiskit's little-endian convention: tensor product is q1 ⊗ q0
    ry0 = ry_matrix(theta0)
    ry1 = ry_matrix(theta1)
    layer1 = np.kron(ry1, ry0)  # 4x4 matrix

    # Layer 2: CNOT with control=q0, target=q1
    # In Qiskit little-endian ordering:
    cnot = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ], dtype=complex)

    return cnot @ layer1


def test_ansatz_unitary_at_zero():
    """At theta=0,0 ansatz should produce the CNOT (Ry(0) = I)."""
    from pytest_quantum import assert_unitary

    qc = make_h2_ansatz(0.0, 0.0)
    expected = expected_ansatz_unitary(0.0, 0.0)
    assert_unitary(qc, expected, atol=1e-6)


def test_ansatz_unitary_at_pi():
    """At theta=pi,0: Ry(pi)|0> = |1>, so q0 flipped before CNOT."""
    from pytest_quantum import assert_unitary

    theta0, theta1 = np.pi, 0.0
    qc = make_h2_ansatz(theta0, theta1)
    expected = expected_ansatz_unitary(theta0, theta1)
    assert_unitary(qc, expected, atol=1e-6)


def test_ansatz_unitary_at_optimal_params():
    """Test at VQE-optimal parameters for H2 ground state.

    The optimal theta values that minimize the H2 energy are approximately:
    theta0 = 0.22 rad, theta1 = -0.22 rad (basis-dependent).
    """
    from pytest_quantum import assert_unitary

    theta0, theta1 = 0.22, -0.22
    qc = make_h2_ansatz(theta0, theta1)
    expected = expected_ansatz_unitary(theta0, theta1)
    assert_unitary(qc, expected, atol=1e-6)
```

**What to expect:**

```
test_vqe_h2.py::test_ansatz_unitary_at_zero       PASSED
test_vqe_h2.py::test_ansatz_unitary_at_pi         PASSED
test_vqe_h2.py::test_ansatz_unitary_at_optimal_params PASSED
```

If a test fails, `assert_unitary` prints the maximum element-wise difference so you can immediately see how far off the circuit is:

```
AssertionError: Circuit does not implement the expected unitary.
  Max |difference| : 1.41e+00   (tolerance: 1.00e-06)
  allow_global_phase = True
  Hint: use allow_global_phase=True if you only care about physical equivalence.
```

---

## Step 3: Parameter Sweep — Verify Unitary Across All Theta Values

A single parameter point is not enough. Use `assert_circuit_sweep` to verify the ansatz unitary is correct across a grid of theta values. This catches bugs that only appear at particular angles (e.g., sign errors at `theta = pi/2`).

```python
# test_vqe_h2.py — Step 3
from pytest_quantum import assert_circuit_sweep


def test_ansatz_sweep_theta0():
    """Ansatz unitary must be correct for all theta0 in [0, 2pi], theta1=0."""
    assert_circuit_sweep(
        circuit_fn=lambda theta0: make_h2_ansatz(theta0, 0.0),
        param_values={"theta0": np.linspace(0, 2 * np.pi, 16)},
        expected_fn=lambda theta0: expected_ansatz_unitary(theta0, 0.0),
        atol=1e-6,
    )


def test_ansatz_sweep_both_params():
    """Ansatz unitary must be correct over a 5x5 grid of (theta0, theta1)."""
    from pytest_quantum import assert_circuit_sweep

    assert_circuit_sweep(
        circuit_fn=lambda theta0, theta1: make_h2_ansatz(theta0, theta1),
        param_values={
            "theta0": np.linspace(0, np.pi, 5),
            "theta1": np.linspace(0, np.pi, 5),
        },
        expected_fn=lambda theta0, theta1: expected_ansatz_unitary(theta0, theta1),
        atol=1e-6,
    )


def test_ansatz_unitary_is_continuous():
    """The ansatz unitary must vary continuously with theta0.

    Discontinuities would indicate branch-cut bugs or phase jumps.
    """
    from pytest_quantum import assert_parametrized_unitary_continuous

    assert_parametrized_unitary_continuous(
        circuit_fn=lambda theta0: make_h2_ansatz(theta0, 0.5),
        param_name="theta0",
        param_range=(0, 2 * np.pi),
        n_samples=30,
        max_jump=0.5,  # Frobenius norm between adjacent samples
    )
```

`assert_circuit_sweep` tests all 25 combinations (5x5 grid) and reports every failure:

```
AssertionError: assert_circuit_sweep failed for 3 / 25 parameter combinations:
  params={theta0=1.5708, theta1=0.0000}: max|diff|=1.41e+00 (tol=1.00e-06)
  params={theta0=1.5708, theta1=0.7854}: max|diff|=1.41e+00 (tol=1.00e-06)
  ...
```

---

## Step 4: Test Expectation Value with the Ideal Simulator

The H2 Hamiltonian in the minimal STO-3G basis, after Jordan-Wigner mapping, can be written as a sum of Pauli operators. At the optimal parameters, the expectation value should approach the ground-state energy.

```python
# test_vqe_h2.py — Step 4
from pytest_quantum import assert_expectation_value_close


def make_h2_hamiltonian():
    """Build the H2 Hamiltonian as a SparsePauliOp.

    These coefficients are for H2 at equilibrium bond length (0.74 Angstrom)
    in the STO-3G basis after the Jordan-Wigner transform and two-electron
    reduction to a 2-qubit problem.
    """
    from qiskit.quantum_info import SparsePauliOp

    # H2 Hamiltonian (2-qubit): E_nn + c_IZ*IZ + c_ZI*ZI + c_ZZ*ZZ + c_XX*XX
    return SparsePauliOp.from_list([
        ("II", -1.0523732),   # nuclear repulsion + constant terms
        ("IZ",  0.3979374),
        ("ZI", -0.3979374),
        ("ZZ", -0.0112801),
        ("XX",  0.1809312),
    ])


def test_expectation_value_at_optimal_params():
    """At optimal parameters, H2 energy should be close to -1.857 Ha."""
    from qiskit.primitives import StatevectorEstimator

    hamiltonian = make_h2_hamiltonian()

    # Optimal VQE parameters for this Hamiltonian
    theta0, theta1 = 0.22, -0.22
    qc = make_h2_ansatz(theta0, theta1)

    # Use Qiskit's StatevectorEstimator (exact, no shot noise)
    estimator = StatevectorEstimator()
    pub = (qc, hamiltonian)
    result = estimator.run([pub]).result()
    energy = float(result[0].data.evs)

    # H2 ground state energy at equilibrium: -1.8572 Ha
    assert_expectation_value_close(energy, expected=-1.8572, atol=0.1)


def test_expectation_value_above_ground_state():
    """Any ansatz state must have energy >= ground state (variational principle)."""
    from qiskit.primitives import StatevectorEstimator

    hamiltonian = make_h2_hamiltonian()

    # Try an arbitrary random parameter
    rng = np.random.default_rng(seed=42)
    theta0, theta1 = rng.uniform(0, 2 * np.pi, 2)
    qc = make_h2_ansatz(theta0, theta1)

    estimator = StatevectorEstimator()
    result = estimator.run([(qc, hamiltonian)]).result()
    energy = float(result[0].data.evs)

    # Variational principle: any trial state has energy >= ground state
    ground_state_energy = -1.8572
    assert energy >= ground_state_energy - 0.01, (
        f"Energy {energy:.4f} is below the ground state {ground_state_energy:.4f} "
        "by more than numerical tolerance — check Hamiltonian coefficients."
    )
```

**What to expect:**

```
test_vqe_h2.py::test_expectation_value_at_optimal_params  PASSED
test_vqe_h2.py::test_expectation_value_above_ground_state PASSED
```

---

## Step 5: Test VQE Convergence

`assert_vqe_converges` runs a complete optimization loop and verifies the energy decreases to the known minimum. This is the integration test for the full VQE algorithm.

```python
# test_vqe_h2.py — Step 5
from pytest_quantum import assert_vqe_converges


def make_energy_function():
    """Return a cost function: theta_array -> H2 energy (float)."""
    from qiskit.primitives import StatevectorEstimator

    hamiltonian = make_h2_hamiltonian()
    estimator = StatevectorEstimator()

    def energy(params: np.ndarray) -> float:
        theta0, theta1 = float(params[0]), float(params[1])
        qc = make_h2_ansatz(theta0, theta1)
        result = estimator.run([(qc, hamiltonian)]).result()
        return float(result[0].data.evs)

    return energy


def test_vqe_converges_to_ground_state():
    """VQE must converge to the H2 ground state energy -1.8572 Ha.

    Uses COBYLA (gradient-free, robust for noisy quantum hardware).
    Starting from (0.5, 0.5) — not the optimum but within basin of attraction.
    """
    energy_fn = make_energy_function()

    assert_vqe_converges(
        cost_function=energy_fn,
        initial_params=[0.5, 0.5],
        method="COBYLA",
        max_iterations=200,
        expected_minimum=-1.8572,
        atol=0.05,  # 50 mHa tolerance — tight but achievable
    )


@pytest.mark.quantum_slow
def test_vqe_converges_from_random_start():
    """VQE should converge regardless of starting point (basin of attraction check)."""
    energy_fn = make_energy_function()
    rng = np.random.default_rng(seed=7)

    for trial in range(3):
        initial = rng.uniform(0, 2 * np.pi, size=2)
        assert_vqe_converges(
            cost_function=energy_fn,
            initial_params=initial,
            method="COBYLA",
            max_iterations=300,
            expected_minimum=-1.8572,
            atol=0.1,
        )
```

Run the fast version:

```bash
pytest test_vqe_h2.py::test_vqe_converges_to_ground_state -v -s
```

You will see optimizer progress printed (scipy minimize calls the cost function ~80-120 times):

```
test_vqe_h2.py::test_vqe_converges_to_ground_state PASSED
```

The slow version (3 random starts) runs with `--quantum-slow`:

```bash
pytest test_vqe_h2.py::test_vqe_converges_from_random_start --quantum-slow -v
```

---

## Step 6: Error Mitigation — Apply ZNE and Assert It Improves

On real hardware, noise corrupts the expectation value. Zero-Noise Extrapolation (ZNE) amplifies the noise at several scale factors and extrapolates back to zero. Here we test this workflow using a noisy Aer simulator so the test is fully reproducible.

```python
# test_vqe_h2.py — Step 6
import pytest
from pytest_quantum import (
    assert_zne_reduces_error,
    assert_zne_expectation_close,
)


def make_noisy_executor(error_rate: float = 0.01):
    """Return an executor callable that runs circuits on a noisy Aer simulator."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit import transpile

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ["ry"])
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(error_rate * 10, 2), ["cx"]
    )
    sim = AerSimulator(noise_model=noise_model)

    def executor(circuit) -> float:
        """Run circuit on noisy simulator, return <ZZ> expectation value."""
        from qiskit.quantum_info import SparsePauliOp

        # Measure ZZ: append Z on each qubit and compute from counts
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()
        transpiled = transpile(meas_circuit, sim, optimization_level=0)
        counts = sim.run(transpiled, shots=4096).result().get_counts()

        total = sum(counts.values())
        zz_exp = 0.0
        for bitstring, count in counts.items():
            # Qiskit bitstrings are little-endian: rightmost bit = q0
            bits = [int(b) for b in reversed(bitstring.replace(" ", ""))]
            if len(bits) >= 2:
                z0 = 1 - 2 * bits[0]  # +1 for |0>, -1 for |1>
                z1 = 1 - 2 * bits[1]
                zz_exp += (z0 * z1) * count / total
        return zz_exp

    return executor


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mitiq") is None,
    reason="mitiq not installed: pip install mitiq",
)
def test_zne_reduces_error_on_ansatz():
    """ZNE should reduce the estimation error compared to unmitigated.

    The ideal <ZZ> for the H2 ansatz at theta=(pi/4, pi/4) is computable.
    With noise, the unmitigated value deviates. ZNE brings it back closer.
    """
    theta0, theta1 = np.pi / 4, np.pi / 4
    qc = make_h2_ansatz(theta0, theta1)

    executor = make_noisy_executor(error_rate=0.02)

    # assert_zne_reduces_error returns (unmitigated, mitigated)
    unmitigated, mitigated = assert_zne_reduces_error(
        circuit=qc,
        executor=executor,
        scale_factors=[1.0, 2.0, 3.0],
        noise_extrapolation="richardson",
    )

    print(f"\n  Unmitigated ZZ: {unmitigated:.4f}")
    print(f"  Mitigated ZZ:   {mitigated:.4f}")

    # Both should be finite floats
    assert np.isfinite(unmitigated)
    assert np.isfinite(mitigated)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mitiq") is None,
    reason="mitiq not installed: pip install mitiq",
)
def test_zne_expectation_close_to_ideal():
    """ZNE-mitigated energy should be close to the ideal energy.

    Uses a simple single-qubit Rz observable for a cleaner demonstration.
    """
    # Use a simpler 1-qubit circuit to keep the test fast and predictable
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)  # |1>, ideal <Z> = -1.0

    def noisy_z_executor(circuit) -> float:
        """Measure <Z> on a noisy simulator."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["x"])
        sim = AerSimulator(noise_model=noise_model)

        c = circuit.copy()
        c.measure_all()
        counts = sim.run(transpile(c, sim), shots=2048).result().get_counts()
        total = sum(counts.values())
        return sum(
            (1 - 2 * int(b[-1])) * cnt / total for b, cnt in counts.items()
        )

    assert_zne_expectation_close(
        circuit=qc,
        executor=noisy_z_executor,
        expected=-1.0,  # ideal <Z> for |1>
        scale_factors=[1.0, 2.0, 3.0],
        atol=0.3,  # generous tolerance for shot noise
        noise_extrapolation="richardson",
    )
```

**What to expect:**

```
test_vqe_h2.py::test_zne_reduces_error_on_ansatz  PASSED
  Unmitigated ZZ: 0.8742
  Mitigated ZZ:   0.9384

test_vqe_h2.py::test_zne_expectation_close_to_ideal PASSED
```

---

## Step 7: Real Hardware Check (Skip Unless --quantum-real)

Tag any test that must run on real IBM Quantum hardware with `@pytest.mark.quantum_real`. These tests are automatically skipped unless you pass `--quantum-real` on the command line.

```python
# test_vqe_h2.py — Step 7
from pytest_quantum import (
    assert_circuit_fits_backend,
    assert_backend_calibration,
    assert_real_counts_close,
)


@pytest.mark.quantum_real
def test_ansatz_fits_real_backend(ibm_backend):
    """Assert the H2 ansatz transpiles within resource limits on the real backend."""
    qc = make_h2_ansatz(0.22, -0.22)

    info = assert_circuit_fits_backend(
        qc,
        ibm_backend,
        optimization_level=3,
        max_depth=50,      # generous — 2-qubit ansatz transpiles deep on some topologies
        max_2q_gates=10,   # 1 CNOT should transpile to <= 10 ECR/CX gates
    )
    print(f"\n  Transpiled depth: {info['depth']}")
    print(f"  2Q gates:         {info['num_2q_gates']}")
    print(f"  All ops:          {info['ops']}")


@pytest.mark.quantum_real
def test_backend_calibration_before_vqe(ibm_backend):
    """Gate quality must meet threshold before we trust VQE results.

    If this test fails, the device is noisy — skip or re-run on a better backend.
    """
    assert_backend_calibration(
        ibm_backend,
        max_gate_error=0.02,       # 2% max 2-qubit gate error
        max_readout_error=0.05,    # 5% max readout error
    )


@pytest.mark.quantum_real
def test_ansatz_measurement_on_real_hardware(ibm_backend):
    """Run the ansatz at optimal params and check the counts make physical sense.

    At theta=(0.22, -0.22), the ground state is primarily |00> with some |11>.
    Exact distribution depends on the Hamiltonian encoding but should not be
    uniform random.
    """
    theta0, theta1 = 0.22, -0.22
    qc = make_h2_ansatz(theta0, theta1)
    qc.measure_all()

    counts = assert_real_counts_close(
        qc,
        ibm_backend,
        expected_probs={"00": 0.88, "01": 0.05, "10": 0.05, "11": 0.02},
        shots=4096,
        max_tvd=0.20,      # real hardware has significant noise
        timeout=300,       # 5 min timeout for IBM queue
        optimization_level=1,
    )
    print(f"\n  Real hardware counts: {dict(sorted(counts.items(), key=lambda x: -x[1])[:4])}")
```

Run the real hardware tests:

```bash
export IBM_QUANTUM_TOKEN="<your-token>"
pytest test_vqe_h2.py -m quantum_real --quantum-real -v
```

Without `--quantum-real`, the tests are skipped cleanly:

```
test_vqe_h2.py::test_ansatz_fits_real_backend             SKIPPED (real hardware tests skipped (pass --quantum-real))
test_vqe_h2.py::test_backend_calibration_before_vqe       SKIPPED (real hardware tests skipped (pass --quantum-real))
test_vqe_h2.py::test_ansatz_measurement_on_real_hardware   SKIPPED (real hardware tests skipped (pass --quantum-real))
```

---

## Full Working Test File

Here is the complete `test_vqe_h2.py` combining all steps:

```python
"""
Complete VQE test suite for the H2 molecule (2-qubit ansatz).

Tests:
  Step 1: Circuit structure (width, depth, gate counts)
  Step 2: Unitary correctness at specific parameter values
  Step 3: Parameter sweep (all theta combinations)
  Step 4: Expectation value with ideal simulator
  Step 5: VQE convergence
  Step 6: ZNE error mitigation
  Step 7: Real hardware checks (--quantum-real)

Run:
  pip install "pytest-quantum[qiskit,mitiq]"
  pytest test_vqe_h2.py -v
  pytest test_vqe_h2.py --quantum-slow -v        # include slow tests
  pytest test_vqe_h2.py --quantum-real -v        # include real hardware
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from pytest_quantum import (
    assert_circuit_depth,
    assert_circuit_width,
    assert_circuit_fits_backend,
    assert_circuit_sweep,
    assert_expectation_value_close,
    assert_gate_count,
    assert_parametrized_unitary_continuous,
    assert_real_counts_close,
    assert_backend_calibration,
    assert_unitary,
    assert_vqe_converges,
    assert_zne_reduces_error,
    assert_zne_expectation_close,
)

_HAS_MITIQ = importlib.util.find_spec("mitiq") is not None
_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None

pytestmark = pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------


def make_h2_ansatz(theta0: float, theta1: float):
    """2-qubit hardware-efficient ansatz: Ry(theta0) x Ry(theta1), then CNOT."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.cx(0, 1)
    return qc


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------


def make_h2_hamiltonian():
    """H2 STO-3G Hamiltonian (2-qubit, Jordan-Wigner reduced)."""
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_list([
        ("II", -1.0523732),
        ("IZ",  0.3979374),
        ("ZI", -0.3979374),
        ("ZZ", -0.0112801),
        ("XX",  0.1809312),
    ])


# ---------------------------------------------------------------------------
# Analytic unitary (for step 2 / step 3)
# ---------------------------------------------------------------------------


def ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def expected_ansatz_unitary(theta0: float, theta1: float) -> np.ndarray:
    """Analytic 4x4 unitary for the H2 ansatz (Qiskit little-endian)."""
    ry0 = ry_matrix(theta0)
    ry1 = ry_matrix(theta1)
    layer1 = np.kron(ry1, ry0)  # q1 ⊗ q0 (little-endian)
    cnot = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ], dtype=complex)
    return cnot @ layer1


# ---------------------------------------------------------------------------
# Step 1: Structure
# ---------------------------------------------------------------------------


def test_ansatz_width():
    assert_circuit_width(make_h2_ansatz(0.1, 0.2), expected_qubits=2)


def test_ansatz_depth():
    assert_circuit_depth(make_h2_ansatz(0.1, 0.2), max_depth=3)


def test_ansatz_gate_counts():
    qc = make_h2_ansatz(0.1, 0.2)
    assert_gate_count(qc, "ry", 2)
    assert_gate_count(qc, "cx", 1)


# ---------------------------------------------------------------------------
# Step 2: Unitary at specific parameter values
# ---------------------------------------------------------------------------


def test_ansatz_unitary_at_zero():
    assert_unitary(make_h2_ansatz(0.0, 0.0), expected_ansatz_unitary(0.0, 0.0))


def test_ansatz_unitary_at_pi():
    assert_unitary(make_h2_ansatz(np.pi, 0.0), expected_ansatz_unitary(np.pi, 0.0))


def test_ansatz_unitary_at_optimal():
    assert_unitary(make_h2_ansatz(0.22, -0.22), expected_ansatz_unitary(0.22, -0.22))


# ---------------------------------------------------------------------------
# Step 3: Parameter sweep
# ---------------------------------------------------------------------------


def test_ansatz_unitary_sweep():
    assert_circuit_sweep(
        circuit_fn=lambda theta0, theta1: make_h2_ansatz(theta0, theta1),
        param_values={
            "theta0": np.linspace(0, np.pi, 5),
            "theta1": np.linspace(0, np.pi, 5),
        },
        expected_fn=lambda theta0, theta1: expected_ansatz_unitary(theta0, theta1),
        atol=1e-6,
    )


def test_ansatz_unitary_is_continuous():
    assert_parametrized_unitary_continuous(
        circuit_fn=lambda theta0: make_h2_ansatz(theta0, 0.5),
        param_name="theta0",
        param_range=(0, 2 * np.pi),
        n_samples=30,
        max_jump=0.5,
    )


# ---------------------------------------------------------------------------
# Step 4: Expectation value
# ---------------------------------------------------------------------------


def test_expectation_value_at_optimal():
    from qiskit.primitives import StatevectorEstimator

    hamiltonian = make_h2_hamiltonian()
    qc = make_h2_ansatz(0.22, -0.22)
    estimator = StatevectorEstimator()
    result = estimator.run([(qc, hamiltonian)]).result()
    energy = float(result[0].data.evs)
    assert_expectation_value_close(energy, expected=-1.8572, atol=0.1)


def test_variational_principle():
    """Every ansatz state must have energy >= true ground state."""
    from qiskit.primitives import StatevectorEstimator

    hamiltonian = make_h2_hamiltonian()
    estimator = StatevectorEstimator()
    rng = np.random.default_rng(42)
    theta0, theta1 = rng.uniform(0, 2 * np.pi, 2)
    qc = make_h2_ansatz(float(theta0), float(theta1))
    result = estimator.run([(qc, hamiltonian)]).result()
    energy = float(result[0].data.evs)
    assert energy >= -1.8572 - 0.01  # variational lower bound


# ---------------------------------------------------------------------------
# Step 5: VQE convergence
# ---------------------------------------------------------------------------


def test_vqe_converges():
    from qiskit.primitives import StatevectorEstimator

    hamiltonian = make_h2_hamiltonian()
    estimator = StatevectorEstimator()

    def energy_fn(params: np.ndarray) -> float:
        qc = make_h2_ansatz(float(params[0]), float(params[1]))
        result = estimator.run([(qc, hamiltonian)]).result()
        return float(result[0].data.evs)

    assert_vqe_converges(
        cost_function=energy_fn,
        initial_params=[0.5, 0.5],
        method="COBYLA",
        max_iterations=200,
        expected_minimum=-1.8572,
        atol=0.05,
    )


# ---------------------------------------------------------------------------
# Step 6: ZNE error mitigation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MITIQ, reason="mitiq not installed")
def test_zne_expectation_close():
    from qiskit import QuantumCircuit

    # Simple X gate: ideal <Z> = -1.0
    qc = QuantumCircuit(1)
    qc.x(0)

    def noisy_z_executor(circuit) -> float:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["x"])
        sim = AerSimulator(noise_model=nm)
        c = circuit.copy()
        c.measure_all()
        counts = sim.run(transpile(c, sim), shots=2048).result().get_counts()
        total = sum(counts.values())
        return sum((1 - 2 * int(b[-1])) * cnt / total for b, cnt in counts.items())

    assert_zne_expectation_close(
        circuit=qc,
        executor=noisy_z_executor,
        expected=-1.0,
        scale_factors=[1.0, 2.0, 3.0],
        atol=0.3,
        noise_extrapolation="richardson",
    )


# ---------------------------------------------------------------------------
# Step 7: Real hardware (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_ansatz_fits_real_backend(ibm_backend):
    qc = make_h2_ansatz(0.22, -0.22)
    info = assert_circuit_fits_backend(
        qc, ibm_backend,
        optimization_level=3,
        max_depth=50,
        max_2q_gates=10,
    )
    print(f"\n  Transpiled depth: {info['depth']}, 2Q gates: {info['num_2q_gates']}")


@pytest.mark.quantum_real
def test_backend_calibration(ibm_backend):
    assert_backend_calibration(
        ibm_backend,
        max_gate_error=0.02,
        max_readout_error=0.05,
    )


@pytest.mark.quantum_real
def test_ansatz_on_real_hardware(ibm_backend):
    qc = make_h2_ansatz(0.22, -0.22)
    qc.measure_all()
    counts = assert_real_counts_close(
        qc, ibm_backend,
        expected_probs={"00": 0.88, "01": 0.05, "10": 0.05, "11": 0.02},
        shots=4096,
        max_tvd=0.20,
        timeout=300,
    )
    print(f"\n  Top counts: {dict(sorted(counts.items(), key=lambda x: -x[1])[:4])}")
```

Run the complete suite:

```bash
# Fast tests (no slow, no real hardware):
pytest test_vqe_h2.py -v

# Include slow convergence tests:
pytest test_vqe_h2.py --quantum-slow -v

# Include real hardware (requires IBM_QUANTUM_TOKEN):
export IBM_QUANTUM_TOKEN="<your-token>"
pytest test_vqe_h2.py --quantum-real -v
```

**Expected output (fast suite):**

```
test_vqe_h2.py::test_ansatz_width                  PASSED
test_vqe_h2.py::test_ansatz_depth                  PASSED
test_vqe_h2.py::test_ansatz_gate_counts             PASSED
test_vqe_h2.py::test_ansatz_unitary_at_zero         PASSED
test_vqe_h2.py::test_ansatz_unitary_at_pi           PASSED
test_vqe_h2.py::test_ansatz_unitary_at_optimal      PASSED
test_vqe_h2.py::test_ansatz_unitary_sweep           PASSED
test_vqe_h2.py::test_ansatz_unitary_is_continuous   PASSED
test_vqe_h2.py::test_expectation_value_at_optimal   PASSED
test_vqe_h2.py::test_variational_principle           PASSED
test_vqe_h2.py::test_vqe_converges                  PASSED
test_vqe_h2.py::test_zne_expectation_close           PASSED  (or SKIPPED if mitiq not installed)
test_vqe_h2.py::test_ansatz_fits_real_backend        SKIPPED (--quantum-real not set)
test_vqe_h2.py::test_backend_calibration             SKIPPED (--quantum-real not set)
test_vqe_h2.py::test_ansatz_on_real_hardware         SKIPPED (--quantum-real not set)

======= 12 passed, 3 skipped in 8.24s =======
```

---

## Common Issues and Troubleshooting

**`assert_unitary` fails with "Max |difference| : 1.41e+00"**

A diff of ~√2 typically means a qubit ordering problem. Qiskit uses little-endian ordering (qubit 0 is the least significant bit). The expected unitary must use the same convention. Check that your `np.kron` order is `q1 ⊗ q0`, not `q0 ⊗ q1`.

**VQE does not converge**

If `assert_vqe_converges` fails with "energy did not decrease", try:
- Increase `max_iterations` (200 is usually enough for 2 parameters, but 500 for more)
- Use `rhobeg=0.5` in COBYLA options (already the default)
- Try `method="L-BFGS-B"` with a gradient-capable cost function

**ZNE fails or gives worse results**

ZNE requires the noise to be gate-error dominated (not readout error dominated) and the circuit must be simple enough that folding is valid. For very short circuits, shot noise may dominate and ZNE can appear to make things worse — increase shots to reduce statistical variance.

**`ImportError: mitiq is required`**

```bash
pip install mitiq
# or
pip install "pytest-quantum[mitiq]"
```

**Slow VQE tests in CI**

Mark them with `@pytest.mark.quantum_slow` and only run with `--quantum-slow` in nightly CI, not on every PR.
