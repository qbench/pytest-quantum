# Tutorial: Running Tests on Real IBM Quantum Hardware

This tutorial walks through every step needed to run pytest-quantum tests on real IBM Quantum hardware — from account setup and environment variables through to a complete test suite with queue management, cost control, and CI integration.

**Prerequisites:**
- Completed Tutorials 01 and 02, or comfortable with basic pytest-quantum
- An IBM Quantum account (free tier is sufficient)
- `pip install "pytest-quantum[qiskit]"`

**Time to complete:** ~40 minutes (most of it waiting in the IBM queue)

---

## 1. Setup: Get an IBM Quantum Account and API Key

1. Go to [https://quantum.ibm.com](https://quantum.ibm.com) and click **Sign Up**.
2. Complete email verification.
3. After login, click your profile icon (top right) and select **Copy token**.
4. Save the token — it looks like `d1a2b3c4...` (64 hex characters).

You also need to install the IBM runtime SDK:

```bash
pip install "pytest-quantum[qiskit]"
# qiskit-ibm-runtime is included as a core dependency
```

Verify the installation:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
# Should import without errors
```

---

## 2. Setting Environment Variables

pytest-quantum's `ibm_backend` fixture reads credentials from environment variables. Never hard-code API tokens in test files.

```bash
# Required
export IBM_QUANTUM_TOKEN="d1a2b3c4..."   # your token from quantum.ibm.com

# Optional — controls which backend is selected
export IBM_QUANTUM_CHANNEL="ibm_quantum_platform"  # default; use this for quantum.ibm.com accounts
export IBM_QUANTUM_INSTANCE="ibm-q/open/main"      # default Open Plan instance
export IBM_QUANTUM_BACKEND=""                       # empty = least_busy (recommended)
export IBM_QUANTUM_MIN_QUBITS="5"                  # minimum qubits on backend (default 5)
```

For CI (GitHub Actions, GitLab CI), set these as repository secrets:

```yaml
# .github/workflows/quantum-hardware.yml
env:
  IBM_QUANTUM_TOKEN: ${{ secrets.IBM_QUANTUM_TOKEN }}
  IBM_QUANTUM_CHANNEL: ibm_quantum_platform
  IBM_QUANTUM_INSTANCE: ibm-q/open/main
```

To use a specific backend (e.g., if you know `ibm_torino` is available and has low error rates):

```bash
export IBM_QUANTUM_BACKEND="ibm_torino"
```

If `IBM_QUANTUM_BACKEND` is empty, the fixture selects the least-busy backend with at least `IBM_QUANTUM_MIN_QUBITS` qubits. This is the recommended approach for most test suites.

---

## 3. The `ibm_backend` Fixture

The `ibm_backend` fixture is session-scoped: it connects to IBM once per test run, selects a backend, and reuses it for all hardware tests. You do not need to import or configure it — just declare it as a test parameter.

```python
# test_ibm_hardware.py — fixture usage
import pytest


@pytest.mark.quantum_real
def test_backend_is_connected(ibm_backend):
    """Verify the ibm_backend fixture returns a real backend object."""
    assert hasattr(ibm_backend, "name"), "ibm_backend should have a .name attribute"
    name = ibm_backend.name
    print(f"\n  Connected to: {name}")
    print(f"  Num qubits:   {ibm_backend.num_qubits}")
```

**What happens internally:**

1. Checks for `--quantum-real` flag — skips if not set.
2. Reads `IBM_QUANTUM_TOKEN` — skips with a helpful message if missing.
3. Connects to `QiskitRuntimeService` via the configured channel.
4. Calls `service.least_busy(min_num_qubits=5, simulator=False, operational=True)` if no backend name is specified.
5. Returns the backend object for use in all `quantum_real` tests in the session.

If `--quantum-real` is not set, all tests with `ibm_backend` in their signature are skipped:

```
test_ibm_hardware.py::test_backend_is_connected  SKIPPED (real hardware tests skipped (pass --quantum-real))
```

---

## 4. `assert_backend_calibration`: Gate Your Tests on Device Quality

Before trusting VQE results or Bell state counts, verify the backend's current calibration data meets your quality threshold. IBM hardware calibration changes hourly — a backend that passed yesterday might have a degraded qubit today.

```python
from pytest_quantum import assert_backend_calibration


@pytest.mark.quantum_real
def test_device_quality(ibm_backend):
    """Check device calibration before running expensive hardware tests.

    If this test fails, the device has high error rates today.
    Options:
      1. Re-run later (IBM recalibrates devices throughout the day)
      2. Use a different backend (set IBM_QUANTUM_BACKEND=ibm_brisbane)
      3. Relax the thresholds if your circuit can tolerate more noise
    """
    assert_backend_calibration(
        ibm_backend,
        max_gate_error=0.01,       # 1% max 2-qubit gate error rate
        max_readout_error=0.05,    # 5% max per-qubit readout error
    )
```

Typical IBM Quantum calibration data (as of early 2025):
- **ibm_brisbane** (127 qubits): avg 2Q gate error ~0.4%, avg readout error ~1%
- **ibm_torino** (133 qubits): avg 2Q gate error ~0.3%, avg readout error ~1%

The assertion reads `backend.properties()` and finds the worst error rate across all qubits. If any single qubit's readout error or any gate pair's error exceeds your threshold, the test fails with a diagnostic message:

```
AssertionError: Backend readout error too high: worst=0.0812 > 0.0500.
  Average readout error : 0.0124
  Per-qubit errors      : ['0.0082', '0.0124', '0.0097', '0.0812', '0.0089']
  Backend               : ibm_sherbrooke
  Hint: choose a lower-noise device or relax max_readout_error.
```

---

## 5. `assert_circuit_fits_backend`: Check Your Circuit Compiles

Transpilation maps your abstract circuit onto the device's native gate set and qubit connectivity. A circuit that looks simple in your code might become deep after transpilation if it requires many SWAP insertions for routing.

```python
from pytest_quantum import assert_circuit_fits_backend


@pytest.mark.quantum_real
def test_bell_fits_backend(ibm_backend):
    """Bell circuit should transpile within tight resource limits."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    info = assert_circuit_fits_backend(
        qc,
        ibm_backend,
        optimization_level=3,     # use maximum optimization
        max_depth=20,             # allow generous depth for 2-qubit circuit
        max_2q_gates=5,           # should need only 1-3 ECR/CX gates
    )

    print(f"\n  Transpiled depth: {info['depth']}")
    print(f"  2Q gates:         {info['num_2q_gates']}")
    print(f"  All ops:          {info['ops']}")
    # e.g.: depth=7, 2Q gates=2, ops={'rz': 8, 'sx': 4, 'ecr': 2, 'measure': 2}


@pytest.mark.quantum_real
def test_vqe_ansatz_fits_backend(ibm_backend):
    """2-qubit VQE ansatz should stay within 50 layers and 10 2Q gates."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.ry(0.22, 0)
    qc.ry(-0.22, 1)
    qc.cx(0, 1)

    info = assert_circuit_fits_backend(
        qc,
        ibm_backend,
        optimization_level=3,
        max_depth=50,
        max_2q_gates=10,
    )

    # Log for debugging / tracking resource usage over time
    print(f"\n  VQE ansatz on {ibm_backend.name}:")
    print(f"    Depth:    {info['depth']}")
    print(f"    2Q gates: {info['num_2q_gates']}")
```

`assert_circuit_fits_backend` returns a dict with keys `depth`, `num_2q_gates`, `num_qubits`, and `ops` (gate count dictionary). Use these values to track resource usage over time — if a circuit's transpiled depth grows from 7 to 20 after a compiler upgrade, you want to know.

---

## 6. `assert_real_counts_close`: Verify Quantum Behavior on Hardware

`assert_real_counts_close` submits a circuit, waits for results, and compares the empirical distribution to expected probabilities using Total Variation Distance (TVD). The default `max_tvd=0.15` is deliberately generous because real hardware has noise.

```python
from pytest_quantum import assert_real_counts_close


@pytest.mark.quantum_real
def test_h_gate_on_hardware(ibm_backend):
    """Single H gate should produce 50/50 within TVD=0.10 on hardware."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts = assert_real_counts_close(
        qc,
        ibm_backend,
        expected_probs={"0": 0.5, "1": 0.5},
        shots=4096,
        max_tvd=0.10,   # H gate is very robust — expect tight distribution
        timeout=300,
        optimization_level=1,
    )
    print(f"\n  H gate counts: {counts}")
    # Typical output: {'0': 2018, '1': 2078}


@pytest.mark.quantum_real
def test_bell_state_on_hardware(ibm_backend):
    """Bell state should be within TVD=0.15 of ideal 50/50."""
    from qiskit import QuantumCircuit

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
        optimization_level=1,
    )

    total = sum(counts.values())
    dominant_frac = (counts.get("00", 0) + counts.get("11", 0)) / total
    print(f"\n  Bell state counts: {counts}")
    print(f"  Dominant outcomes: {dominant_frac:.1%} (expected ~98%+ for ideal)")
    # Typical output on ibm_torino:
    # Bell state counts: {'00': 1897, '11': 1994, '01': 97, '10': 108}
    # Dominant outcomes: 94.8%
```

**Choosing `max_tvd`:**

| Circuit type | Suggested `max_tvd` |
|---|---|
| Single-qubit gates (H, X, S) | 0.05 — 0.10 |
| 2-qubit Bell state | 0.10 — 0.15 |
| 3+ qubit circuits | 0.15 — 0.25 |
| VQE ansatz with rotations | 0.20 |

More qubits = more transpilation = more gate errors = more noise. Be realistic about what hardware can deliver.

---

## 7. `assert_mirror_fidelity`: Measure Actual Gate Fidelity

Mirror benchmarking runs a circuit followed by its inverse. In the ideal case, the combined circuit is the identity and all qubits return to `|0>`. The fraction of `|00...0>` outcomes is a proxy for gate fidelity.

This is a self-contained fidelity check that requires no classical simulator reference.

```python
from pytest_quantum import assert_mirror_fidelity


@pytest.mark.quantum_real
def test_cx_mirror_fidelity(ibm_backend):
    """CNOT mirror fidelity should be > 0.85 on a healthy device.

    Circuit: CX — CX† = CX — CX (CNOT is self-inverse)
    Ideal outcome: both qubits return to |0>, giving counts = {'00': shots}
    A fidelity of 0.85 means 85% of shots returned |00>.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    fidelity = assert_mirror_fidelity(
        qc,
        ibm_backend,
        shots=4096,
        min_fidelity=0.85,
        timeout=300,
    )
    print(f"\n  CX mirror fidelity: {fidelity:.3f}")
    # Typical output on ibm_torino: 0.917


@pytest.mark.quantum_real
def test_bell_mirror_fidelity(ibm_backend):
    """Bell preparation + inverse should return to |00> with fidelity > 0.80."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    # circuit.inverse() appends CX† then H† automatically

    fidelity = assert_mirror_fidelity(
        qc,
        ibm_backend,
        shots=4096,
        min_fidelity=0.80,   # lower threshold: 2 gates in each direction
        timeout=300,
    )
    print(f"\n  Bell mirror fidelity: {fidelity:.3f}")
```

Use `assert_mirror_fidelity` when:
- You want a device-quality benchmark without needing an expected probability distribution
- You want to test that a specific pair of qubits is healthy before running a longer experiment
- You want to track fidelity over time (run this in a nightly CI job and log the result)

---

## 8. Using the `--quantum-real` Flag and CI Setup

**Local development:**

```bash
# Default: real hardware tests are skipped
pytest test_ibm_hardware.py -v
# All quantum_real tests show SKIPPED

# Enable real hardware tests
export IBM_QUANTUM_TOKEN="..."
pytest test_ibm_hardware.py --quantum-real -v
```

**CI setup (GitHub Actions):**

Split hardware tests into a separate workflow that runs on a schedule (e.g., nightly) rather than on every PR. Real hardware tests are slow and use your monthly quota.

```yaml
# .github/workflows/nightly-hardware.yml
name: Nightly Hardware Tests

on:
  schedule:
    - cron: "0 2 * * *"   # 2 AM UTC every night
  workflow_dispatch:        # allow manual trigger

jobs:
  hardware-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install "pytest-quantum[qiskit]"

      - name: Run hardware tests
        env:
          IBM_QUANTUM_TOKEN: ${{ secrets.IBM_QUANTUM_TOKEN }}
          IBM_QUANTUM_CHANNEL: ibm_quantum_platform
          IBM_QUANTUM_INSTANCE: ibm-q/open/main
        run: |
          pytest tests/test_ibm_hardware.py \
            --quantum-real \
            --timeout=600 \
            -v \
            --tb=short
```

**conftest.py for hardware tests:**

```python
# conftest.py (project root)
import pytest

# Register the quantum_real marker to suppress PytestUnknownMarkWarning
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "quantum_real: mark test as requiring real quantum hardware "
        "(run with --quantum-real; skipped by default).",
    )
```

Or in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "quantum: mark test as quantum-related",
    "quantum_slow: mark test as requiring many shots",
    "quantum_real: mark test as requiring real quantum hardware",
]
```

---

## 9. Dealing with Queue Times

IBM Quantum uses a fair-share queue. During peak hours, jobs can wait 5-30+ minutes. pytest-quantum handles this with built-in polling and configurable timeouts.

The `ibm_backend` fixture selects the **least-busy** backend by default, which minimizes expected queue time. Still, plan for delays:

```python
@pytest.mark.quantum_real
def test_with_long_timeout(ibm_backend):
    """Use a generous timeout for long queue times."""
    from qiskit import QuantumCircuit
    from pytest_quantum import assert_backend_executes

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    counts = assert_backend_executes(
        qc,
        ibm_backend,
        shots=1024,
        timeout=1800,   # 30 minutes — generous for peak hours
    )
    assert sum(counts.values()) == 1024
```

pytest-quantum prints job status updates while polling:

```
  [pytest-quantum] Job submitted to ibm_torino: abc123def456
  [pytest-quantum] abc123def456: QUEUED (1790s remaining) ...
  [pytest-quantum] abc123def456: QUEUED (1748s remaining) ...
  [pytest-quantum] abc123def456: RUNNING (1724s remaining) ...
```

The polling interval starts at 5 seconds and grows exponentially (up to 30 seconds) to avoid hammering the API.

**Tips for reducing queue time:**

1. **Use `IBM_QUANTUM_BACKEND=""` (empty):** The `least_busy` selector picks the backend with the shortest queue right now. A backend with 5 jobs waiting may finish in 10 minutes; one with 50 jobs may take an hour.

2. **Run during off-peak hours:** US Eastern night (3-7 AM UTC) tends to have shorter queues.

3. **Use fewer shots:** IBM queues are depth-limited, not shot-limited. 1024 shots takes the same queue time as 8192 shots.

4. **Batch jobs when possible:** Submit multiple circuits in one job rather than one test per job.

---

## 10. Cost and Time Management with the Open Plan

The IBM Quantum Open Plan provides **10 minutes of quantum execution time per month** (as of 2025). This is the time the circuit is actually running on the QPU, not queue wait time.

**How much can you do with 10 minutes?**

| Circuit | Shots | Runtime per job |
|---|---|---|
| 2-qubit Bell state | 4096 | ~5 seconds |
| 5-qubit GHZ state | 4096 | ~10 seconds |
| 10-qubit VQE ansatz | 4096 | ~20 seconds |
| 127-qubit circuit at full depth | 4096 | ~60 seconds |

With 10 minutes (600 seconds), you can run roughly:
- 120 x Bell state jobs, or
- 60 x 5-qubit jobs, or
- 10 x full 127-qubit jobs

**Strategies to stay within the budget:**

```python
# strategy 1: use shot_budget fixture to track total shots per test
def test_with_budget(ibm_backend, shot_budget):
    """Track total shots to stay within the monthly quota."""
    budget = shot_budget(max_shots=8192)  # total for this test

    from qiskit import QuantumCircuit
    from pytest_quantum import assert_backend_executes

    qc1 = QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure_all()

    qc2 = QuantumCircuit(2, 2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure_all()

    shots1 = budget.allocate(4096)
    shots2 = budget.allocate(4096)

    counts1 = assert_backend_executes(qc1, ibm_backend, shots=shots1)
    counts2 = assert_backend_executes(qc2, ibm_backend, shots=shots2)

    print(f"\n  Shots used: {budget.used} / {budget.max_shots}")
```

```python
# strategy 2: use @pytest.mark.quantum_slow to prevent accidental hardware runs in PR CI
@pytest.mark.quantum_real
@pytest.mark.quantum_slow          # also requires --quantum-slow
def test_expensive_vqe(ibm_backend):
    """Run only in nightly CI with --quantum-slow --quantum-real."""
    ...
```

```python
# strategy 3: mark calibration checks with quantum_real but skip if error rate is high
@pytest.mark.quantum_real
def test_calibration_gate(ibm_backend):
    """Skip the rest of the suite if the device is underperforming."""
    try:
        assert_backend_calibration(ibm_backend, max_gate_error=0.015)
    except AssertionError as exc:
        pytest.skip(f"Device quality too low today — skipping hardware suite: {exc}")
```

---

## 11. Complete Working Test File

The following test file matches the patterns used when testing on `ibm_torino`. It is structured to run cheaply (only a few jobs) while covering the key hardware assertions.

```python
"""
Complete IBM Quantum hardware test suite.

Prerequisites:
    export IBM_QUANTUM_TOKEN="<your-token>"
    pip install "pytest-quantum[qiskit]"

Run:
    pytest test_ibm_hardware.py --quantum-real -v

All tests are tagged @pytest.mark.quantum_real and are skipped unless
--quantum-real is passed. This makes it safe to include this file in
your normal test suite without accidentally hitting hardware.
"""

from __future__ import annotations

import pytest

from pytest_quantum import (
    assert_backend_calibration,
    assert_backend_executes,
    assert_circuit_fits_backend,
    assert_measurement_distribution,
    assert_mirror_fidelity,
    assert_real_counts_close,
)


# ---------------------------------------------------------------------------
# Prerequisite: verify device quality
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_device_calibration(ibm_backend):
    """Verify backend calibration before running any circuit tests.

    This should be the FIRST test in the hardware suite. If it fails,
    the device is noisy and subsequent tests may be unreliable.

    IBM typical values (early 2025):
      ibm_torino:  gate_error ~0.003, readout_error ~0.01
      ibm_brisbane: gate_error ~0.005, readout_error ~0.012
    """
    print(f"\n  Backend: {ibm_backend.name}")
    print(f"  Qubits: {ibm_backend.num_qubits}")

    assert_backend_calibration(
        ibm_backend,
        max_gate_error=0.02,       # generous: allow up to 2% 2Q gate error
        max_readout_error=0.08,    # generous: allow up to 8% readout error
    )


# ---------------------------------------------------------------------------
# Circuit fit check (no shots used)
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_h_gate_fits_backend(ibm_backend):
    """Single H gate should transpile to <= 5 layers on any backend."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)

    info = assert_circuit_fits_backend(
        qc,
        ibm_backend,
        optimization_level=3,
        max_depth=5,
        max_2q_gates=0,    # no 2Q gates in a single-qubit circuit
    )
    print(f"\n  H gate transpiled: depth={info['depth']}, ops={info['ops']}")


@pytest.mark.quantum_real
def test_bell_fits_backend(ibm_backend):
    """Bell circuit transpiles within 20 layers and uses <= 5 2Q gates."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    info = assert_circuit_fits_backend(
        qc,
        ibm_backend,
        optimization_level=3,
        max_depth=20,
        max_2q_gates=5,
    )
    print(
        f"\n  Bell transpiled: depth={info['depth']}, "
        f"2Q gates={info['num_2q_gates']}, ops={info['ops']}"
    )


# ---------------------------------------------------------------------------
# Single-qubit gate tests
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_h_gate_on_hardware(ibm_backend):
    """H gate should produce ~50/50 within TVD=0.10."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts = assert_real_counts_close(
        qc,
        ibm_backend,
        expected_probs={"0": 0.5, "1": 0.5},
        shots=4096,
        max_tvd=0.10,
        timeout=600,
        optimization_level=1,
    )
    print(f"\n  H gate: {counts}")


@pytest.mark.quantum_real
def test_x_gate_on_hardware(ibm_backend):
    """X gate should flip |0> to |1> with fidelity > 95%."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)

    counts = assert_real_counts_close(
        qc,
        ibm_backend,
        expected_probs={"1": 1.0},
        shots=2048,
        max_tvd=0.05,   # X gate is a Clifford — should be very accurate
        timeout=300,
        optimization_level=1,
    )
    total = sum(counts.values())
    fidelity = counts.get("1", 0) / total
    print(f"\n  X gate: {counts}, fidelity={fidelity:.3f}")


# ---------------------------------------------------------------------------
# Two-qubit Bell state
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_bell_state_on_hardware(ibm_backend):
    """Bell state should be within TVD=0.15 of ideal 50/50."""
    from qiskit import QuantumCircuit

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
        timeout=600,
        optimization_level=1,
    )

    total = sum(counts.values())
    dominant = counts.get("00", 0) + counts.get("11", 0)
    print(f"\n  Bell state: {counts}")
    print(f"  Dominant outcomes: {dominant / total:.1%}")
    # Expected on ibm_torino: ~94-97%


# ---------------------------------------------------------------------------
# Mirror fidelity benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_cx_mirror_fidelity(ibm_backend):
    """CNOT mirror fidelity should be > 0.85.

    This measures the practical gate fidelity on the actual physical qubits
    assigned by the transpiler, accounting for routing overhead.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    fidelity = assert_mirror_fidelity(
        qc,
        ibm_backend,
        shots=4096,
        min_fidelity=0.85,
        timeout=600,
    )
    print(f"\n  CX mirror fidelity: {fidelity:.3f}")
    # Expected on ibm_torino: 0.91 - 0.95


@pytest.mark.quantum_real
def test_bell_mirror_fidelity(ibm_backend):
    """Bell preparation mirror fidelity should be > 0.80."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    fidelity = assert_mirror_fidelity(
        qc,
        ibm_backend,
        shots=4096,
        min_fidelity=0.80,
        timeout=600,
    )
    print(f"\n  Bell mirror fidelity: {fidelity:.3f}")


# ---------------------------------------------------------------------------
# VQE ansatz on real hardware
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_h2_ansatz_on_hardware(ibm_backend):
    """H2 VQE ansatz should execute and return plausible counts.

    At theta0=0.22, theta1=-0.22 (near-optimal for H2), the expected
    measurement distribution (in the computational basis) is dominated
    by |00>. The exact distribution depends on the Hamiltonian encoding
    and qubit mapping.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.ry(0.22, 0)
    qc.ry(-0.22, 1)
    qc.cx(0, 1)
    qc.measure_all()

    # At near-optimal params, the state is mostly |00>
    # Use generous TVD since we're not at exactly the ground state
    counts = assert_real_counts_close(
        qc,
        ibm_backend,
        expected_probs={"00": 0.90, "01": 0.04, "10": 0.04, "11": 0.02},
        shots=4096,
        max_tvd=0.20,
        timeout=600,
        optimization_level=1,
    )
    print(f"\n  H2 ansatz counts: {dict(sorted(counts.items(), key=lambda x: -x[1]))}")


# ---------------------------------------------------------------------------
# Raw execution (no assertion on counts — just verify job completes)
# ---------------------------------------------------------------------------


@pytest.mark.quantum_real
def test_backend_executes_successfully(ibm_backend):
    """Smoke test: verify the backend can execute and return counts."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts = assert_backend_executes(
        qc,
        ibm_backend,
        shots=1024,
        timeout=600,
        transpile=True,
        optimization_level=1,
    )
    assert sum(counts.values()) == 1024, (
        f"Expected 1024 total shots, got {sum(counts.values())}"
    )
    print(f"\n  Smoke test counts: {counts}")
```

Run the full hardware suite:

```bash
export IBM_QUANTUM_TOKEN="<your-token>"
pytest test_ibm_hardware.py --quantum-real -v --tb=short
```

**Expected output (ibm_torino, off-peak):**

```
test_ibm_hardware.py::test_device_calibration           PASSED
  Backend: ibm_torino
  Qubits: 133

test_ibm_hardware.py::test_h_gate_fits_backend          PASSED
  H gate transpiled: depth=1, ops={'sx': 1, 'rz': 2, 'measure': 1}

test_ibm_hardware.py::test_bell_fits_backend            PASSED
  Bell transpiled: depth=7, 2Q gates=2, ops={'rz': 7, 'sx': 4, 'ecr': 2, 'measure': 2}

  [pytest-quantum] Job submitted to ibm_torino: c0a1b2...

test_ibm_hardware.py::test_h_gate_on_hardware           PASSED
  H gate: {'0': 2018, '1': 2078}

test_ibm_hardware.py::test_x_gate_on_hardware           PASSED
  X gate: {'1': 1989, '0': 59}, fidelity=0.971

test_ibm_hardware.py::test_bell_state_on_hardware       PASSED
  Bell state: {'00': 1897, '11': 1994, '01': 97, '10': 108}
  Dominant outcomes: 94.8%

test_ibm_hardware.py::test_cx_mirror_fidelity           PASSED
  CX mirror fidelity: 0.917

test_ibm_hardware.py::test_bell_mirror_fidelity         PASSED
  Bell mirror fidelity: 0.883

test_ibm_hardware.py::test_h2_ansatz_on_hardware        PASSED
  H2 ansatz counts: {'00': 3641, '01': 197, '10': 186, '11': 72}

test_ibm_hardware.py::test_backend_executes_successfully PASSED
  Smoke test counts: {'0': 517, '1': 507}

======= 10 passed in 247.34s (0:04:07) =======
```

The tests ran in ~4 minutes total, using approximately 60 seconds of quantum execution time (well within the 10-minute monthly quota).

---

## Common Issues and Troubleshooting

**`IBM_QUANTUM_TOKEN not set` (test is skipped)**

```bash
export IBM_QUANTUM_TOKEN="<your-64-character-token>"
```

Get your token at [https://quantum.ibm.com](https://quantum.ibm.com) under Account > API Token.

**`IBM Quantum connection failed: (IBM_ERROR_CODE...)`**

The token may be expired or the channel may be wrong. Try:

```bash
# The fixture tries ibm_quantum_platform first, then ibm_cloud
export IBM_QUANTUM_CHANNEL="ibm_quantum_platform"  # for quantum.ibm.com accounts
export IBM_QUANTUM_CHANNEL="ibm_cloud"              # for IBM Cloud accounts
```

**`pytest.skip: IBM Quantum backend selection failed`**

No backends with >= 5 qubits are currently available. Options:
1. Wait a few minutes and retry.
2. Set `IBM_QUANTUM_MIN_QUBITS=2` to accept smaller backends.
3. Set `IBM_QUANTUM_BACKEND=ibm_brisbane` to target a specific backend directly.

**Tests time out (AssertionError: Job ... timed out after 300s)**

The IBM queue is busy. Options:
1. Increase `timeout=1800` (30 minutes) for peak hours.
2. Run at off-peak hours (2-7 AM UTC).
3. Use `least_busy` (the default) rather than a specific backend.

**`assert_backend_calibration` fails**

The device has high error rates today. Options:
1. Re-run in a few hours (IBM recalibrates devices throughout the day).
2. Relax `max_gate_error` or `max_readout_error` thresholds.
3. Try a different backend: `export IBM_QUANTUM_BACKEND=ibm_brisbane`.

**`assert_real_counts_close` fails (TVD exceeded)**

The device has higher noise than `max_tvd` allows. Options:
1. Check the IBM dashboard for current calibration data.
2. Increase `max_tvd` (try 0.20 for 2-qubit circuits on noisy days).
3. Use `optimization_level=3` to reduce gate count via better compilation.
4. If calibration looks bad, skip with `assert_backend_calibration` first.

**Monthly quota exceeded**

You have used all 10 minutes of Open Plan quantum time. Options:
1. Wait until the quota resets (first of the month).
2. Use a Qiskit network account through an institution for more time.
3. Use `aer_noise_simulator` instead for local testing during the month.
