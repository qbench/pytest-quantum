"""Real quantum hardware assertions.

Submit circuits to real backends, wait for results, and assert correctness.
Works with any Qiskit-compatible backend (IBM Quantum, AerSimulator, etc.).

All functions handle job submission, polling with exponential back-off,
timeout, and produce detailed failure messages that include job IDs and
backend names so failures are easy to diagnose on the IBM dashboard.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

# Two-qubit gate names recognised for counting purposes
_TWO_Q_GATES = frozenset(
    {"cx", "cz", "ecr", "cp", "swap", "rzz", "rxx", "ryy", "iswap", "dcx", "cs", "csx"}
)


# ---------------------------------------------------------------------------
# Public assertions
# ---------------------------------------------------------------------------


def assert_backend_executes(
    circuit: Any,
    backend: Any,
    *,
    shots: int = 1024,
    timeout: float = 300.0,
    transpile: bool = True,
    optimization_level: int = 1,
) -> dict[str, int]:
    """Assert a circuit executes successfully on a backend and return counts.

    Submits the circuit (optionally transpiling it first), waits for the job
    to complete, and returns the measurement counts.  Raises AssertionError if
    the job fails, is cancelled, or does not finish within *timeout* seconds.

    Args:
        circuit:           Qiskit QuantumCircuit to execute.
        backend:           Qiskit-compatible backend (IBM, AerSimulator, …).
        shots:             Number of shots (default 1024).
        timeout:           Maximum seconds to wait for job completion (default 300).
        transpile:         Transpile to backend before running (default True).
        optimization_level: Qiskit transpilation level 0–3 (default 1).

    Returns:
        dict[str, int]: Measurement counts keyed by bitstring.

    Raises:
        AssertionError: If the job fails, is cancelled, or times out.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum import assert_backend_executes


        def test_h_gate(ibm_backend):
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            counts = assert_backend_executes(qc, ibm_backend, shots=2048)
            assert counts.get("0", 0) + counts.get("1", 0) == 2048
    """
    try:
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for hardware assertions. "
            "Install with: pip install qiskit"
        ) from exc

    backend_name = _backend_name(backend)

    if transpile:
        circuit = qk_transpile(
            circuit,
            backend,
            optimization_level=optimization_level,
        )

    # IBM Runtime 0.20+ removed backend.run() — use SamplerV2 for IBM backends
    try:
        from qiskit_ibm_runtime import IBMBackend
        from qiskit_ibm_runtime import SamplerV2 as IBMSampler

        if isinstance(backend, IBMBackend):
            sampler = IBMSampler(backend)
            job = sampler.run([circuit], shots=shots)
            print(
                f"\n  [pytest-quantum] Job submitted to {backend_name}: {_job_id(job)}"
            )
            return _wait_for_sampler_job(
                job, timeout=timeout, backend_name=backend_name
            )
    except ImportError:
        pass

    # Fallback: backend.run() for AerSimulator and other non-IBM backends
    job = backend.run(circuit, shots=shots)
    print(f"\n  [pytest-quantum] Job submitted to {backend_name}: {_job_id(job)}")
    return _wait_for_job(job, timeout=timeout, backend_name=backend_name)


def assert_real_counts_close(
    circuit: Any,
    backend: Any,
    expected_probs: dict[str, float],
    *,
    shots: int = 4096,
    max_tvd: float = 0.15,
    timeout: float = 300.0,
    optimization_level: int = 1,
) -> dict[str, int]:
    """Assert real hardware counts match expected probabilities within TVD tolerance.

    Runs the circuit on the backend and compares the empirical distribution to
    *expected_probs* using Total Variation Distance (TVD).  The default TVD
    threshold (0.15) is intentionally more lenient than simulator tests because
    real hardware has noise.

    Args:
        circuit:           QuantumCircuit with measurements.
        backend:           Qiskit-compatible backend.
        expected_probs:    Dict mapping bitstring → ideal probability (must sum to ~1).
        shots:             Number of shots (default 4096; more = less sampling noise).
        max_tvd:           Maximum allowed TVD (default 0.15).
        timeout:           Max seconds to wait for job completion (default 300).
        optimization_level: Transpilation level (default 1).

    Returns:
        dict[str, int]: Actual measurement counts.

    Raises:
        AssertionError: If TVD exceeds max_tvd.

    Example::

        from pytest_quantum import assert_real_counts_close


        def test_bell_state(ibm_backend):
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            assert_real_counts_close(
                qc,
                ibm_backend,
                expected_probs={"00": 0.5, "11": 0.5},
                max_tvd=0.15,
            )
    """
    counts = assert_backend_executes(
        circuit,
        backend,
        shots=shots,
        timeout=timeout,
        transpile=True,
        optimization_level=optimization_level,
    )

    total = sum(counts.values())
    actual_probs = {k: v / total for k, v in counts.items()}

    all_keys = set(expected_probs) | set(actual_probs)
    p = np.array([actual_probs.get(k, 0.0) for k in all_keys])
    q = np.array([expected_probs.get(k, 0.0) for k in all_keys])
    tvd = float(np.sum(np.abs(p - q)) / 2)

    if tvd > max_tvd:
        top_counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:8])
        raise AssertionError(
            f"Real hardware counts differ from expected by TVD={tvd:.4f} > {max_tvd:.4f}.\n"
            f"  Backend          : {_backend_name(backend)}\n"
            f"  Shots            : {shots}\n"
            f"  Top actual counts: {top_counts}\n"
            f"  Expected probs   : {dict(sorted(expected_probs.items()))}\n"
            f"  Hint: real hardware has noise — try raising max_tvd or increasing shots."
        )

    return counts


def assert_mirror_fidelity(
    circuit: Any,
    backend: Any,
    *,
    shots: int = 4096,
    min_fidelity: float = 0.5,
    timeout: float = 300.0,
) -> float:
    """Assert a circuit's mirror achieves sufficient |0…0⟩ return fidelity.

    Appends the circuit's inverse to itself (circuit ∘ circuit†), measures all
    qubits, and checks that the |0…0⟩ outcome fraction meets *min_fidelity*.
    This is a lightweight form of mirror benchmarking that works for any
    invertible circuit without needing a classical simulator reference.

    Args:
        circuit:      QuantumCircuit without measurements (must be invertible).
        backend:      Qiskit-compatible backend.
        shots:        Number of shots (default 4096).
        min_fidelity: Minimum required |0…0⟩ return fraction (default 0.5).
                      Reduce this for deeper circuits or noisier devices.
        timeout:      Max seconds to wait for job completion (default 300).

    Returns:
        float: Measured |0…0⟩ fraction (fidelity proxy).

    Raises:
        AssertionError: If fidelity < min_fidelity.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum import assert_mirror_fidelity


        def test_cx_mirror(ibm_backend):
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            fidelity = assert_mirror_fidelity(qc, ibm_backend, min_fidelity=0.5)
            print(f"Mirror fidelity: {fidelity:.3f}")
    """
    try:
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError("qiskit is required for mirror fidelity assertion.") from exc

    # Compose circuit + inverse BEFORE transpiling so the full mirror is
    # optimised together and only basis-gate instructions reach the backend.
    mirror = circuit.compose(circuit.inverse())
    mirror.measure_all()
    mirror = qk_transpile(mirror, backend, optimization_level=1)

    counts = assert_backend_executes(
        mirror, backend, shots=shots, timeout=timeout, transpile=False
    )

    # Use actual measured bit length (not total backend qubits) to build zero state
    bit_len = len(next(iter(counts))) if counts else mirror.num_qubits
    zero_state = "0" * bit_len
    zero_count = counts.get(zero_state, 0)
    fidelity = zero_count / shots

    if fidelity < min_fidelity:
        top = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])
        raise AssertionError(
            f"Mirror fidelity {fidelity:.4f} < {min_fidelity:.4f}.\n"
            f"  |{'0' * bit_len}⟩ outcomes : {zero_count} / {shots}\n"
            f"  Backend           : {_backend_name(backend)}\n"
            f"  Top outcomes      : {top}\n"
            f"  Hint: lower min_fidelity for noisy devices, or reduce circuit depth."
        )

    return fidelity


def assert_backend_calibration(
    backend: Any,
    *,
    max_gate_error: float = 0.01,
    max_readout_error: float = 0.05,
) -> None:
    """Assert backend calibration data meets quality thresholds.

    Reads the latest calibration data from backend.properties() and checks
    that all readout errors and 2-qubit gate errors are within the specified
    bounds.  Useful as a prerequisite test to skip an entire suite if the
    device is under-performing.

    Args:
        backend:           IBM backend with a .properties() method.
        max_gate_error:    Maximum allowed 2-qubit gate error rate (default 0.01 = 1%).
        max_readout_error: Maximum allowed per-qubit readout error (default 0.05 = 5%).

    Raises:
        AssertionError: If any error rate exceeds the threshold, or if
                        the backend exposes no calibration data.

    Example::

        from pytest_quantum import assert_backend_calibration


        def test_device_quality(ibm_backend):
            assert_backend_calibration(
                ibm_backend,
                max_gate_error=0.005,
                max_readout_error=0.03,
            )
    """
    props = backend.properties() if hasattr(backend, "properties") else None
    if props is None:
        raise AssertionError(
            f"Backend '{_backend_name(backend)}' has no calibration properties.\n"
            f"  Hint: simulator backends don't carry hardware calibration data."
        )

    # --- readout errors ---
    readout_errors: list[float] = []
    num_qubits = backend.num_qubits if hasattr(backend, "num_qubits") else 0
    for qubit in range(num_qubits):
        try:
            err = props.readout_error(qubit)
            if err is not None:
                readout_errors.append(float(err))
        except Exception:
            pass

    if readout_errors:
        worst_ro = max(readout_errors)
        if worst_ro > max_readout_error:
            raise AssertionError(
                f"Backend readout error too high: worst={worst_ro:.4f} > {max_readout_error:.4f}.\n"
                f"  Average readout error : {float(np.mean(readout_errors)):.4f}\n"
                f"  Per-qubit errors      : {[f'{e:.4f}' for e in readout_errors]}\n"
                f"  Backend               : {_backend_name(backend)}\n"
                f"  Hint: choose a lower-noise device or relax max_readout_error."
            )

    # --- 2-qubit gate errors ---
    two_q_errors: list[float] = []
    for gate in getattr(props, "gates", []):
        if len(gate.qubits) == 2:
            for param in gate.parameters:
                if param.name == "gate_error" and param.value is not None:
                    two_q_errors.append(float(param.value))

    # Filter out gate_error = 1.0 — these are disabled/unused qubit links
    two_q_errors = [e for e in two_q_errors if e < 1.0]

    if two_q_errors:
        worst_gate = max(two_q_errors)
        if worst_gate > max_gate_error:
            raise AssertionError(
                f"Backend 2-qubit gate error too high: worst={worst_gate:.4f} > {max_gate_error:.4f}.\n"
                f"  Average 2Q gate error : {float(np.mean(two_q_errors)):.4f}\n"
                f"  Backend               : {_backend_name(backend)}\n"
                f"  Hint: choose a higher-quality device or relax max_gate_error."
            )


def assert_circuit_fits_backend(
    circuit: Any,
    backend: Any,
    *,
    optimization_level: int = 3,
    max_depth: int | None = None,
    max_2q_gates: int | None = None,
) -> dict[str, Any]:
    """Assert a circuit transpiles to the backend and stays within resource limits.

    Transpiles *circuit* for *backend* and checks optional depth and 2-qubit
    gate count constraints.  Even without constraints this is useful to assert
    the circuit is compatible with the backend's basis gates and connectivity.

    Args:
        circuit:           QuantumCircuit to check.
        backend:           Target backend.
        optimization_level: Qiskit transpilation level 0–3 (default 3).
        max_depth:         If set, assert transpiled depth ≤ max_depth.
        max_2q_gates:      If set, assert 2-qubit gate count ≤ max_2q_gates.

    Returns:
        dict with keys 'depth', 'num_2q_gates', 'num_qubits', 'ops'.

    Raises:
        AssertionError: If constraints are violated.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum import assert_circuit_fits_backend


        def test_grover_fits(ibm_backend):
            info = assert_circuit_fits_backend(
                grover_circuit,
                ibm_backend,
                max_depth=200,
                max_2q_gates=50,
            )
            print(
                f"Transpiled depth: {info['depth']}, 2Q gates: {info['num_2q_gates']}"
            )
    """
    try:
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_circuit_fits_backend."
        ) from exc

    transpiled = qk_transpile(
        circuit,
        backend,
        optimization_level=optimization_level,
    )
    depth = transpiled.depth()
    ops = dict(transpiled.count_ops())
    num_2q = sum(v for k, v in ops.items() if k in _TWO_Q_GATES)

    result: dict[str, Any] = {
        "depth": depth,
        "num_2q_gates": num_2q,
        "num_qubits": transpiled.num_qubits,
        "ops": ops,
    }

    failures: list[str] = []
    if max_depth is not None and depth > max_depth:
        failures.append(f"  depth={depth} exceeds max_depth={max_depth}")
    if max_2q_gates is not None and num_2q > max_2q_gates:
        failures.append(f"  2Q gates={num_2q} exceeds max_2q_gates={max_2q_gates}")

    if failures:
        raise AssertionError(
            "Circuit does not fit backend constraints:\n"
            + "\n".join(failures)
            + f"\n  Backend  : {_backend_name(backend)}\n"
            f"  Depth    : {depth}\n"
            f"  2Q gates : {num_2q}\n"
            f"  All ops  : {ops}"
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wait_for_job(
    job: Any,
    *,
    timeout: float = 300.0,
    backend_name: str = "backend",
) -> dict[str, int]:
    """Poll a Qiskit job until done, raise on failure/timeout, return counts."""
    deadline = time.monotonic() + timeout
    poll = 5.0  # initial poll interval seconds

    while True:
        elapsed = time.monotonic()
        remaining = deadline - elapsed

        if remaining <= 0:
            raise AssertionError(
                f"Job {_job_id(job)} on {backend_name} timed out after {timeout:.0f}s.\n"
                f"  Hint: increase timeout or check the IBM Quantum queue status."
            )

        try:
            from qiskit.providers import JobStatus

            status = job.status()
            if status == JobStatus.DONE:
                break
            if status in (JobStatus.ERROR, JobStatus.CANCELLED):
                raise AssertionError(
                    f"Job {_job_id(job)} on {backend_name} ended with status {status.name}.\n"
                    f"  Hint: check the IBM Quantum dashboard for error details."
                )
            print(
                f"  [pytest-quantum] {_job_id(job)}: {status.name}"
                f" ({remaining:.0f}s remaining) …"
            )
        except ImportError:
            # qiskit.providers not available — block directly on result()
            break

        time.sleep(min(poll, remaining))
        poll = min(poll * 1.5, 30.0)  # exponential back-off up to 30 s

    result = job.result()
    counts: dict[str, int] = result.get_counts()
    return counts


def _wait_for_sampler_job(
    job: Any,
    *,
    timeout: float = 300.0,
    backend_name: str = "backend",
) -> dict[str, int]:
    """Poll an IBM SamplerV2 job until done and return counts."""
    deadline = time.monotonic() + timeout
    poll = 5.0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"Job {_job_id(job)} on {backend_name} timed out after {timeout:.0f}s.\n"
                f"  Hint: increase timeout or check the IBM Quantum queue status."
            )

        # SamplerV2 status() returns a string: 'QUEUED', 'RUNNING', 'DONE', 'ERROR', etc.
        status = job.status()
        status_str = (
            status if isinstance(status, str) else getattr(status, "name", str(status))
        )

        if status_str in ("DONE", "COMPLETED"):
            break
        if status_str in ("ERROR", "FAILED", "CANCELLED"):
            raise AssertionError(
                f"Job {_job_id(job)} on {backend_name} ended with status {status_str}.\n"
                f"  Hint: check the IBM Quantum dashboard for error details."
            )

        print(
            f"  [pytest-quantum] {_job_id(job)}: {status_str}"
            f" ({remaining:.0f}s remaining) …"
        )
        time.sleep(min(poll, remaining))
        poll = min(poll * 1.5, 30.0)

    result = job.result()
    return _extract_counts(result[0])


from pytest_quantum._internal import (
    _extract_sampler_counts as _extract_counts,
    _backend_name,
)


def _job_id(job: object) -> str:
    """Return job ID string or '<unknown>'."""
    if hasattr(job, "job_id") and callable(job.job_id):
        return str(job.job_id())
    return "<unknown>"
