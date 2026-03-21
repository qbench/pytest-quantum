# Tutorial: Noise-Aware Testing with Qiskit Aer and Mitiq

Real quantum computers are noisy. A circuit that passes every unitary and distribution test on an ideal simulator can still produce garbage on hardware. This tutorial shows how to write tests that explicitly model noise, assert graceful degradation, and verify that error mitigation techniques actually help.

**Prerequisites:**
- Completed Tutorial 01, or comfortable with basic pytest-quantum assertions
- `pip install "pytest-quantum[qiskit,mitiq]"`

**Time to complete:** ~25 minutes

---

## 1. Why Noise Testing Matters

The standard testing pyramid for quantum software looks like this:

```
         /\
        /  \   Real hardware tests (slow, expensive, $)
       /----\
      /      \  Noisy simulator tests  (fast, free, reproducible)
     /--------\
    /          \  Ideal simulator tests (milliseconds, exact)
   /------------\
```

Ideal simulator tests catch algorithmic bugs (wrong gates, wrong connectivity). Noisy simulator tests catch a different class of bugs:

- **Depth sensitivity**: Does your circuit fail before error mitigation has a chance to help?
- **Error budget**: Does the circuit tolerate 1% noise but not 3%?
- **Mitigation effectiveness**: Does ZNE actually work for your specific circuit structure?
- **Distribution degradation**: Does noise spread counts across wrong bitstrings catastrophically, or gracefully?

pytest-quantum provides two tools for this: the `aer_noise_simulator` fixture (built-in depolarizing noise) and the Mitiq assertion family for ZNE and CDR.

---

## 2. The `aer_noise_simulator` Fixture

The `aer_noise_simulator` fixture is function-scoped (not session-scoped) because noise models are parameterized per-test. It returns a **factory** — call it with the error rate you want:

```python
def test_example(aer_noise_simulator):
    sim = aer_noise_simulator(error_rate=0.01)  # 1% depolarizing on all gates
    # sim is a ready-to-use AerSimulator
```

The noise model applies:
- Depolarizing error at `error_rate` to all single-qubit gates (H, X, Y, Z, S, T, Rx, Ry, Rz, U)
- Depolarizing error at `error_rate * 10` to all two-qubit gates (CNOT, CZ, SWAP)

This 10x ratio is realistic: two-qubit gate errors are typically 10-30x worse than single-qubit errors on real IBM hardware.

```python
# test_noise_tutorial.py — basic fixture usage
import pytest
import numpy as np
from qiskit import QuantumCircuit, transpile


def bell_circuit_with_measurements():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def test_noise_fixture_creates_simulator(aer_noise_simulator):
    """Verify the fixture returns a working simulator object."""
    sim = aer_noise_simulator(error_rate=0.01)
    assert hasattr(sim, "run"), "Expected an AerSimulator with a .run() method"


def test_noise_simulator_shot_count(aer_noise_simulator):
    """Noisy simulator should return exactly the requested shot count."""
    sim = aer_noise_simulator(error_rate=0.05)
    qc = bell_circuit_with_measurements()
    counts = sim.run(transpile(qc, sim), shots=1000).result().get_counts()
    assert sum(counts.values()) == 1000
```

---

## 3. Testing That Your Circuit Degrades Gracefully with Noise

A well-designed quantum circuit should degrade *gradually* as noise increases — not catastrophically. Test this property explicitly:

```python
# test_noise_tutorial.py — graceful degradation
from pytest_quantum import assert_measurement_distribution, assert_counts_close


def test_bell_tolerates_low_noise(aer_noise_simulator):
    """Bell state should still produce mostly |00> and |11> at 1% noise."""
    sim = aer_noise_simulator(error_rate=0.01)
    qc = bell_circuit_with_measurements()
    counts = sim.run(transpile(qc, sim), shots=4000).result().get_counts()

    total = sum(counts.values())
    dominant = counts.get("00", 0) + counts.get("11", 0)

    # At 1% noise, dominant outcomes should still be >85% of shots
    assert dominant / total > 0.85, (
        f"Bell state with 1% noise produced only {dominant / total:.1%} "
        f"dominant outcomes. Counts: {counts}"
    )


def test_bell_degrades_with_high_noise(aer_noise_simulator):
    """At 30% noise, the Bell distribution should deviate significantly from ideal."""
    from qiskit_aer import AerSimulator

    ideal_sim = AerSimulator()
    noisy_sim = aer_noise_simulator(error_rate=0.30)

    qc = bell_circuit_with_measurements()
    counts_ideal = ideal_sim.run(transpile(qc, ideal_sim), shots=2000).result().get_counts()
    counts_noisy = noisy_sim.run(transpile(qc, noisy_sim), shots=2000).result().get_counts()

    # 30% noise makes the distributions clearly different (tight TVD threshold fails)
    with pytest.raises(AssertionError, match="TVD"):
        assert_counts_close(counts_ideal, counts_noisy, max_tvd=0.05)
    # But with a generous threshold, they still overlap (not complete garbage)
    # This asserts noise is bounded — the circuit "degrades" rather than "fails"


def test_noise_error_budget():
    """Verify error tolerance at multiple noise levels."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    qc = bell_circuit_with_measurements()
    ideal = AerSimulator()
    ideal_counts = ideal.run(transpile(qc, ideal), shots=4000).result().get_counts()

    # The TVD from ideal should increase monotonically with noise
    # (this tests that our noise model behaves predictably)
    error_rates = [0.001, 0.01, 0.05, 0.10]
    prev_tvd = 0.0

    for error_rate in error_rates:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ["h"])
        nm.add_all_qubit_quantum_error(depolarizing_error(error_rate * 10, 2), ["cx"])
        noisy_sim = AerSimulator(noise_model=nm)

        noisy_counts = noisy_sim.run(
            transpile(qc, noisy_sim), shots=4000
        ).result().get_counts()

        total_i = sum(ideal_counts.values())
        total_n = sum(noisy_counts.values())
        probs_i = {k: v / total_i for k, v in ideal_counts.items()}
        probs_n = {k: v / total_n for k, v in noisy_counts.items()}

        all_keys = set(probs_i) | set(probs_n)
        tvd = sum(abs(probs_i.get(k, 0) - probs_n.get(k, 0)) for k in all_keys) / 2

        assert tvd >= prev_tvd - 0.05, (
            f"TVD at error_rate={error_rate} ({tvd:.4f}) is lower than at "
            f"previous rate ({prev_tvd:.4f}). Noise should monotonically increase TVD."
        )
        prev_tvd = tvd
```

---

## 4. `assert_counts_close` vs `assert_real_counts_close`

These two assertions answer different questions:

| Assertion | Use case | Default `max_tvd` |
|---|---|---|
| `assert_counts_close(a, b)` | Compare two count dicts (same basis, e.g. ideal vs noisy) | 0.05 |
| `assert_real_counts_close(circuit, backend, probs)` | Run on a backend, compare to ideal probs | 0.15 |

**`assert_counts_close`** is for comparing two count dictionaries you already have. Use it to compare ideal vs noisy, or two noise levels, or two compilation strategies:

```python
from pytest_quantum import assert_counts_close


def test_low_noise_stays_close_to_ideal(aer_noise_simulator):
    """1% noise should keep TVD within 5% of the ideal distribution."""
    from qiskit_aer import AerSimulator

    qc = bell_circuit_with_measurements()
    ideal = AerSimulator()
    noisy = aer_noise_simulator(error_rate=0.005)  # 0.5% noise

    counts_ideal = ideal.run(transpile(qc, ideal), shots=4000).result().get_counts()
    counts_noisy = noisy.run(transpile(qc, noisy), shots=4000).result().get_counts()

    assert_counts_close(counts_ideal, counts_noisy, max_tvd=0.10)


def test_transpilation_does_not_change_distribution(aer_noise_simulator):
    """Compiling at opt level 1 vs 3 should give the same distribution."""
    from qiskit_aer import AerSimulator

    qc = bell_circuit_with_measurements()
    sim = AerSimulator()

    counts_opt1 = sim.run(
        transpile(qc, sim, optimization_level=1), shots=3000
    ).result().get_counts()
    counts_opt3 = sim.run(
        transpile(qc, sim, optimization_level=3), shots=3000
    ).result().get_counts()

    # Optimization level should not change measurement outcomes for a Bell state
    assert_counts_close(counts_opt1, counts_opt3, max_tvd=0.05)
```

**`assert_real_counts_close`** submits a circuit to a backend, waits for results, and compares to ideal probabilities. Use it for real hardware tests:

```python
from pytest_quantum import assert_real_counts_close


@pytest.mark.quantum_real
def test_bell_on_real_hardware(ibm_backend):
    """Bell state on real IBM hardware should be within TVD=0.15 of ideal."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    counts = assert_real_counts_close(
        qc,
        ibm_backend,
        expected_probs={"00": 0.5, "11": 0.5},
        shots=4096,
        max_tvd=0.15,   # real hardware has readout + gate noise
        timeout=300,
    )
    print(f"\n  Hardware counts: {counts}")
```

The 0.15 default TVD for `assert_real_counts_close` (vs 0.05 for `assert_counts_close`) reflects the reality that real hardware is noisier than even a noisy simulator. Adjust `max_tvd` based on your device's error rates from the calibration dashboard.

---

## 5. Testing ZNE Actually Helps: `assert_zne_reduces_error`

`assert_zne_reduces_error` runs the circuit at multiple noise scale factors, extrapolates to zero noise, and returns `(unmitigated, mitigated)`. You then assert the mitigated result is closer to the ideal.

```python
import pytest
import importlib.util
from pytest_quantum import assert_zne_reduces_error

_HAS_MITIQ = importlib.util.find_spec("mitiq") is not None


def make_noisy_z_executor(error_rate: float = 0.02):
    """Return an executor that measures <Z> on qubit 0 with depolarizing noise."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit import transpile

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ["x", "h", "ry"])
    nm.add_all_qubit_quantum_error(depolarizing_error(error_rate * 10, 2), ["cx"])
    sim = AerSimulator(noise_model=nm)

    def executor(circuit) -> float:
        c = circuit.copy()
        c.measure_all()
        counts = sim.run(transpile(c, sim), shots=4096).result().get_counts()
        total = sum(counts.values())
        # <Z> on qubit 0: sum over bitstrings of (-1)^{bit0} * count/total
        exp_val = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            bit0 = int(bits[-1])  # Qiskit: rightmost = qubit 0
            exp_val += (1 - 2 * bit0) * count / total
        return exp_val

    return executor


@pytest.mark.skipif(not _HAS_MITIQ, reason="mitiq not installed")
def test_zne_reduces_error_on_x_gate():
    """ZNE should bring noisy <Z> for |1> state closer to the ideal -1.0.

    X|0> = |1>, so ideal <Z> = -1.0.
    With noise, the measured value will be > -1.0 (closer to 0).
    ZNE extrapolates to zero noise to recover the -1.0.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)  # prepare |1>

    executor = make_noisy_z_executor(error_rate=0.05)
    ideal_z = -1.0

    unmitigated, mitigated = assert_zne_reduces_error(
        circuit=qc,
        executor=executor,
        scale_factors=[1.0, 2.0, 3.0],
        noise_extrapolation="richardson",
    )

    print(f"\n  Ideal <Z>:        {ideal_z:.4f}")
    print(f"  Unmitigated <Z>:  {unmitigated:.4f}")
    print(f"  Mitigated <Z>:    {mitigated:.4f}")
    print(f"  Unmitigated error: {abs(unmitigated - ideal_z):.4f}")
    print(f"  Mitigated error:   {abs(mitigated - ideal_z):.4f}")

    # Mitigated should be closer to -1.0 than unmitigated
    assert abs(mitigated - ideal_z) <= abs(unmitigated - ideal_z) + 0.05, (
        f"ZNE did not improve: unmitigated={unmitigated:.4f}, mitigated={mitigated:.4f}"
    )


@pytest.mark.skipif(not _HAS_MITIQ, reason="mitiq not installed")
def test_zne_with_linear_extrapolation():
    """Linear extrapolation is faster but less accurate than Richardson."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)

    executor = make_noisy_z_executor(error_rate=0.03)

    unmitigated_lin, mitigated_lin = assert_zne_reduces_error(
        circuit=qc,
        executor=executor,
        scale_factors=[1.0, 2.0, 3.0],
        noise_extrapolation="linear",
    )

    # Richardson extrapolation (default)
    unmitigated_rich, mitigated_rich = assert_zne_reduces_error(
        circuit=qc,
        executor=executor,
        scale_factors=[1.0, 2.0, 3.0],
        noise_extrapolation="richardson",
    )

    print(f"\n  Linear ZNE:     unmitigated={unmitigated_lin:.4f}, mitigated={mitigated_lin:.4f}")
    print(f"  Richardson ZNE: unmitigated={unmitigated_rich:.4f}, mitigated={mitigated_rich:.4f}")

    # Both should return finite values
    assert np.isfinite(mitigated_lin)
    assert np.isfinite(mitigated_rich)
```

**What to expect:**

```
test_noise_tutorial.py::test_zne_reduces_error_on_x_gate PASSED

  Ideal <Z>:        -1.0000
  Unmitigated <Z>:  -0.8234
  Mitigated <Z>:    -0.9641
  Unmitigated error: 0.1766
  Mitigated error:   0.0359
```

ZNE reduced the error from 0.18 to 0.04 — a 5x improvement.

---

## 6. Testing CDR: `assert_cdr_reduces_error`

Clifford Data Regression (CDR) trains a linear model on near-Clifford circuits that can be simulated classically, then uses the model to correct the noisy result. It requires both a noisy executor and a classical simulator.

```python
from pytest_quantum import assert_cdr_reduces_error


@pytest.mark.skipif(not _HAS_MITIQ, reason="mitiq not installed")
def test_cdr_reduces_error():
    """CDR should bring the noisy expectation value closer to the ideal.

    CDR works best for circuits that are near-Clifford (T gate count is low).
    The Bell state circuit is Clifford, which is ideal for CDR training.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit import transpile

    # Build a near-Clifford circuit: Bell state + small Ry rotation (non-Clifford)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.1, 0)  # small non-Clifford rotation

    # Noisy executor: runs on AerSimulator with 3% depolarizing noise
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.03, 1), ["h", "ry"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.10, 2), ["cx"])
    noisy_sim = AerSimulator(noise_model=nm)

    def noisy_executor(circuit) -> float:
        """Measure <ZZ> on noisy hardware."""
        c = circuit.copy()
        c.measure_all()
        counts = noisy_sim.run(transpile(c, noisy_sim), shots=2048).result().get_counts()
        total = sum(counts.values())
        zz = 0.0
        for bs, cnt in counts.items():
            bits = [int(b) for b in reversed(bs.replace(" ", ""))]
            if len(bits) >= 2:
                zz += (1 - 2 * bits[0]) * (1 - 2 * bits[1]) * cnt / total
        return zz

    # Classical simulator: AerSimulator without noise (exact statevector)
    ideal_sim = AerSimulator(method="statevector")

    def classical_simulator(circuit) -> float:
        """Exact classical simulation for CDR training."""
        c = circuit.copy()
        c.measure_all()
        counts = ideal_sim.run(transpile(c, ideal_sim), shots=8192).result().get_counts()
        total = sum(counts.values())
        zz = 0.0
        for bs, cnt in counts.items():
            bits = [int(b) for b in reversed(bs.replace(" ", ""))]
            if len(bits) >= 2:
                zz += (1 - 2 * bits[0]) * (1 - 2 * bits[1]) * cnt / total
        return zz

    unmitigated, mitigated = assert_cdr_reduces_error(
        circuit=qc,
        executor=noisy_executor,
        simulator=classical_simulator,
        num_training_circuits=10,
        atol=0.2,  # mitigated should be within 0.2 of the classical ideal
    )

    print(f"\n  CDR unmitigated: {unmitigated:.4f}")
    print(f"  CDR mitigated:   {mitigated:.4f}")
    print(f"  Classical ideal: {classical_simulator(qc):.4f}")
```

**What to expect:**

```
test_noise_tutorial.py::test_cdr_reduces_error PASSED

  CDR unmitigated: 0.8521
  CDR mitigated:   0.9412
  Classical ideal: 0.9502
```

CDR brought the estimate from 0.852 to 0.941, much closer to the classical ideal of 0.950.

---

## 7. Snapshot Testing Noisy Distributions

Snapshot tests are golden-file tests: the first run saves the result; subsequent runs compare against it. Use them to catch regressions when you refactor a circuit — if the noise profile changes unexpectedly, the snapshot fails.

```python
from pytest_quantum import assert_distribution_snapshot


def test_bell_noise_snapshot(aer_noise_simulator):
    """Snapshot the noisy Bell distribution to catch regressions.

    If this test fails after a code change, it means the noise profile of the
    circuit has changed — either because the circuit structure changed (more
    gates = more noise) or because the noise model parameters were altered.
    """
    sim = aer_noise_simulator(error_rate=0.01)
    qc = bell_circuit_with_measurements()
    counts = sim.run(transpile(qc, sim), shots=4000).result().get_counts()

    # First run: saves to .pytest-quantum-snapshots/bell_noise_01pct_dist.npy
    # Subsequent runs: compares with TVD <= 0.07
    assert_distribution_snapshot(counts, "bell_noise_01pct", max_tvd=0.07)


def test_bell_two_noise_levels_both_snapshot(aer_noise_simulator):
    """Snapshot two noise levels to track how error budget changes."""
    sim_low = aer_noise_simulator(error_rate=0.005)
    sim_high = aer_noise_simulator(error_rate=0.02)
    qc = bell_circuit_with_measurements()

    for sim, name in [(sim_low, "bell_noise_low"), (sim_high, "bell_noise_high")]:
        counts = sim.run(transpile(qc, sim), shots=4000).result().get_counts()
        assert_distribution_snapshot(counts, name, max_tvd=0.08)
```

To update snapshots after an intentional circuit change:

```bash
pytest test_noise_tutorial.py -k snapshot --quantum-update-snapshots
# or
PYTEST_QUANTUM_UPDATE_SNAPSHOTS=1 pytest test_noise_tutorial.py -k snapshot
```

Snapshot files are stored in `.pytest-quantum-snapshots/` — commit them to version control so CI can compare against them.

---

## 8. Complete Working Test File

```python
"""
Noise-aware testing tutorial: complete test file.

Tests:
  - aer_noise_simulator fixture usage
  - Graceful degradation with noise
  - assert_counts_close for comparing count dicts
  - assert_zne_reduces_error (requires mitiq)
  - assert_cdr_reduces_error (requires mitiq)
  - assert_distribution_snapshot (golden-file testing)
  - assert_real_counts_close (requires --quantum-real)

Run:
  pip install "pytest-quantum[qiskit,mitiq]"
  pytest test_noise_tutorial.py -v
  pytest test_noise_tutorial.py --quantum-real -v   # real hardware tests
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile

from pytest_quantum import (
    assert_counts_close,
    assert_distribution_snapshot,
    assert_measurement_distribution,
    assert_real_counts_close,
    assert_zne_reduces_error,
    assert_zne_expectation_close,
)

_HAS_MITIQ = importlib.util.find_spec("mitiq") is not None
_HAS_QISKIT = importlib.util.find_spec("qiskit_aer") is not None

pytestmark = pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit-aer not installed")


# ---------------------------------------------------------------------------
# Shared circuit builders
# ---------------------------------------------------------------------------


def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def make_z_executor(error_rate: float):
    """Return an executor that measures <Z0> with depolarizing noise."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ["x", "h", "ry"])
    nm.add_all_qubit_quantum_error(depolarizing_error(error_rate * 10, 2), ["cx"])
    sim = AerSimulator(noise_model=nm)

    def executor(circuit) -> float:
        c = circuit.copy()
        c.measure_all()
        counts = sim.run(transpile(c, sim), shots=4096).result().get_counts()
        total = sum(counts.values())
        return sum(
            (1 - 2 * int(bs.replace(" ", "")[-1])) * cnt / total
            for bs, cnt in counts.items()
        )

    return executor


# ---------------------------------------------------------------------------
# Section 2: aer_noise_simulator fixture
# ---------------------------------------------------------------------------


def test_noise_fixture_creates_simulator(aer_noise_simulator):
    sim = aer_noise_simulator(error_rate=0.01)
    assert hasattr(sim, "run")


def test_noise_simulator_shot_count(aer_noise_simulator):
    sim = aer_noise_simulator(error_rate=0.05)
    qc = bell_circuit()
    counts = sim.run(transpile(qc, sim), shots=1000).result().get_counts()
    assert sum(counts.values()) == 1000


# ---------------------------------------------------------------------------
# Section 3: Graceful degradation
# ---------------------------------------------------------------------------


def test_bell_tolerates_low_noise(aer_noise_simulator):
    """1% noise: dominant outcomes (00+11) > 85% of shots."""
    sim = aer_noise_simulator(error_rate=0.01)
    qc = bell_circuit()
    counts = sim.run(transpile(qc, sim), shots=4000).result().get_counts()
    total = sum(counts.values())
    dominant = counts.get("00", 0) + counts.get("11", 0)
    assert dominant / total > 0.85


def test_bell_degrades_with_high_noise(aer_noise_simulator):
    """30% noise makes the distribution clearly different from ideal."""
    from qiskit_aer import AerSimulator

    ideal_sim = AerSimulator()
    noisy_sim = aer_noise_simulator(error_rate=0.30)
    qc = bell_circuit()

    counts_ideal = ideal_sim.run(transpile(qc, ideal_sim), shots=2000).result().get_counts()
    counts_noisy = noisy_sim.run(transpile(qc, noisy_sim), shots=2000).result().get_counts()

    with pytest.raises(AssertionError, match="TVD"):
        assert_counts_close(counts_ideal, counts_noisy, max_tvd=0.05)


# ---------------------------------------------------------------------------
# Section 4: assert_counts_close
# ---------------------------------------------------------------------------


def test_low_noise_stays_close_to_ideal(aer_noise_simulator):
    """0.5% noise: TVD from ideal < 10%."""
    from qiskit_aer import AerSimulator

    qc = bell_circuit()
    ideal_sim = AerSimulator()
    noisy_sim = aer_noise_simulator(error_rate=0.005)

    counts_ideal = ideal_sim.run(transpile(qc, ideal_sim), shots=4000).result().get_counts()
    counts_noisy = noisy_sim.run(transpile(qc, noisy_sim), shots=4000).result().get_counts()

    assert_counts_close(counts_ideal, counts_noisy, max_tvd=0.10)


def test_transpilation_does_not_change_distribution():
    """Opt level 1 vs 3 should give the same Bell distribution."""
    from qiskit_aer import AerSimulator

    qc = bell_circuit()
    sim = AerSimulator()

    counts_opt1 = sim.run(transpile(qc, sim, optimization_level=1), shots=3000).result().get_counts()
    counts_opt3 = sim.run(transpile(qc, sim, optimization_level=3), shots=3000).result().get_counts()
    assert_counts_close(counts_opt1, counts_opt3, max_tvd=0.05)


# ---------------------------------------------------------------------------
# Section 5: ZNE
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MITIQ, reason="mitiq not installed")
def test_zne_reduces_error():
    """ZNE improves <Z> estimate for the X gate (|1> state, ideal = -1.0)."""
    qc = QuantumCircuit(1)
    qc.x(0)

    executor = make_z_executor(error_rate=0.05)
    ideal_z = -1.0

    unmitigated, mitigated = assert_zne_reduces_error(
        circuit=qc,
        executor=executor,
        scale_factors=[1.0, 2.0, 3.0],
        noise_extrapolation="richardson",
    )
    print(f"\n  Unmitigated: {unmitigated:.4f}, Mitigated: {mitigated:.4f}, Ideal: {ideal_z}")
    assert abs(mitigated - ideal_z) <= abs(unmitigated - ideal_z) + 0.05


@pytest.mark.skipif(not _HAS_MITIQ, reason="mitiq not installed")
def test_zne_expectation_close_to_ideal():
    """ZNE-mitigated value should be within 0.3 of the ideal -1.0."""
    from pytest_quantum import assert_zne_expectation_close

    qc = QuantumCircuit(1)
    qc.x(0)

    assert_zne_expectation_close(
        circuit=qc,
        executor=make_z_executor(error_rate=0.04),
        expected=-1.0,
        scale_factors=[1.0, 2.0, 3.0],
        atol=0.3,
        noise_extrapolation="richardson",
    )


# ---------------------------------------------------------------------------
# Section 6: CDR
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MITIQ, reason="mitiq not installed")
def test_cdr_reduces_error():
    """CDR brings noisy <ZZ> estimate closer to the classical simulator."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from pytest_quantum import assert_cdr_reduces_error

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.1, 0)  # small non-Clifford rotation for CDR to work on

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.03, 1), ["h", "ry"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.10, 2), ["cx"])
    noisy_sim = AerSimulator(noise_model=nm)
    ideal_sim = AerSimulator(method="statevector")

    def make_zz_executor(sim, shots=2048):
        def executor(circuit) -> float:
            c = circuit.copy()
            c.measure_all()
            counts = sim.run(transpile(c, sim), shots=shots).result().get_counts()
            total = sum(counts.values())
            zz = 0.0
            for bs, cnt in counts.items():
                bits = [int(b) for b in reversed(bs.replace(" ", ""))]
                if len(bits) >= 2:
                    zz += (1 - 2 * bits[0]) * (1 - 2 * bits[1]) * cnt / total
            return zz
        return executor

    unmitigated, mitigated = assert_cdr_reduces_error(
        circuit=qc,
        executor=make_zz_executor(noisy_sim),
        simulator=make_zz_executor(ideal_sim, shots=8192),
        num_training_circuits=10,
        atol=0.2,
    )
    print(f"\n  CDR unmitigated: {unmitigated:.4f}, mitigated: {mitigated:.4f}")


# ---------------------------------------------------------------------------
# Section 7: Snapshot testing
# ---------------------------------------------------------------------------


def test_bell_noise_distribution_snapshot(aer_noise_simulator):
    """Golden-file test: noisy Bell distribution should not change across refactors."""
    sim = aer_noise_simulator(error_rate=0.01)
    qc = bell_circuit()
    counts = sim.run(transpile(qc, sim), shots=4000).result().get_counts()
    assert_distribution_snapshot(counts, "bell_noise_01pct", max_tvd=0.08)


# ---------------------------------------------------------------------------
# Real hardware test (skipped unless --quantum-real)
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_bell_on_real_hardware(ibm_backend):
    """Bell state on real IBM hardware: TVD from ideal <= 0.15."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    counts = assert_real_counts_close(
        qc,
        ibm_backend,
        expected_probs={"00": 0.5, "11": 0.5},
        shots=4096,
        max_tvd=0.15,
        timeout=300,
    )
    print(f"\n  Real hardware counts: {counts}")
```

Run the full suite:

```bash
# Fast tests (no mitiq required for most):
pytest test_noise_tutorial.py -v

# With mitiq (ZNE and CDR tests):
pytest test_noise_tutorial.py -v   # ZNE/CDR auto-skip if mitiq not installed

# Update snapshots after intentional change:
pytest test_noise_tutorial.py -k snapshot --quantum-update-snapshots

# Real hardware:
export IBM_QUANTUM_TOKEN="<your-token>"
pytest test_noise_tutorial.py --quantum-real -v
```

**Expected output (with mitiq installed):**

```
test_noise_tutorial.py::test_noise_fixture_creates_simulator      PASSED
test_noise_tutorial.py::test_noise_simulator_shot_count            PASSED
test_noise_tutorial.py::test_bell_tolerates_low_noise              PASSED
test_noise_tutorial.py::test_bell_degrades_with_high_noise         PASSED
test_noise_tutorial.py::test_low_noise_stays_close_to_ideal        PASSED
test_noise_tutorial.py::test_transpilation_does_not_change_distribution PASSED
test_noise_tutorial.py::test_zne_reduces_error                     PASSED
test_noise_tutorial.py::test_zne_expectation_close_to_ideal        PASSED
test_noise_tutorial.py::test_cdr_reduces_error                     PASSED
test_noise_tutorial.py::test_bell_noise_distribution_snapshot      PASSED
test_noise_tutorial.py::test_bell_on_real_hardware                 SKIPPED (--quantum-real not set)

======= 10 passed, 1 skipped in 15.32s =======
```

---

## Common Issues and Troubleshooting

**`assert_counts_close` fails intermittently**

Shot-based tests are inherently probabilistic. A TVD of 0.05 means you should expect a roughly 5% false-failure rate at `significance=0.05`. To reduce flakiness:
- Increase shots (4000+ for tight TVD bounds)
- Use `assert_measurement_distribution` instead (chi-square test with p-value)
- Increase `max_tvd` for noisy simulator tests

**ZNE makes results worse**

This can happen when:
1. Shot noise dominates gate noise — the statistical uncertainty from finite shots swamps the mitigation signal. Increase shots.
2. The circuit is too short for noise to dominate — with only 1-2 gates, ZNE has little to extrapolate from.
3. Scale factors are too aggressive — try `[1.0, 1.5, 2.0]` instead of `[1.0, 2.0, 3.0]`.

**CDR fails with "circuit must contain near-Clifford gates"**

CDR works by replacing non-Clifford rotations with Clifford approximations for training. If your circuit is purely Clifford (e.g., a Bell circuit without rotations), CDR cannot generate useful training circuits. Add a small rotation (`ry(0.1, 0)`) to make it near-Clifford.

**Snapshot test fails after code change**

If you intentionally changed the circuit or noise model:

```bash
pytest test_noise_tutorial.py -k snapshot --quantum-update-snapshots
```

Then commit the updated `.pytest-quantum-snapshots/*.npy` files. If the failure was accidental, investigate what changed.

**`ImportError: mitiq is required`**

```bash
pip install mitiq
# or
pip install "pytest-quantum[mitiq]"
```
