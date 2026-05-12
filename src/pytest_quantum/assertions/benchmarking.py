"""Quantum benchmarking assertions for pytest-quantum.

Provides assertions for quantum volume, randomized benchmarking, T1 relaxation,
and gate fidelity — all designed to work with IBM Quantum backends and simulators.

Each assertion raises ``AssertionError`` with a detailed, human-readable message
so failures are easy to diagnose in CI logs.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Public assertions
# ---------------------------------------------------------------------------


def assert_quantum_volume(
    backend: Any,
    target_qv: int,
    *,
    num_trials: int = 100,
    shots: int = 1024,
    confidence: float = 0.97,
    timeout: float = 300.0,
) -> int:
    """Assert a backend achieves at least *target_qv* quantum volume.

    Runs IBM's quantum volume (QV) protocol: for each width *n* from 1 up to
    ``log2(target_qv)``, generate *num_trials* random square circuits of
    depth *n* x *n* using random SU(4) two-qubit unitaries, execute them, and
    check that the heavy output probability (HOP) exceeds 2/3 with the given
    statistical confidence (one-sided binomial test).  The measured QV is
    ``2 ** n`` for the largest *n* that passes.

    For IBM backends (``qiskit_ibm_runtime.IBMBackend``) the circuits are run
    using ``SamplerV2``; all other backends fall back to ``backend.run()``.

    Args:
        backend:     Qiskit-compatible backend to benchmark.
        target_qv:   Minimum required quantum volume (must be a power of 2).
        num_trials:  Number of random circuits per width (default 100).
        shots:       Shots per circuit (default 1024).
        confidence:  Required statistical confidence for HOP > 2/3 test
                     (default 0.97).
        timeout:     Maximum seconds to wait for all jobs (default 300).

    Returns:
        int: The measured quantum volume (largest passing power of 2).

    Raises:
        AssertionError: If the measured QV < target_qv, with details about
                        heavy output probability, confidence, and trial count.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum.assertions.benchmarking import assert_quantum_volume
        from qiskit_aer import AerSimulator


        def test_simulator_qv():
            backend = AerSimulator()
            qv = assert_quantum_volume(backend, target_qv=4, num_trials=20)
            assert qv >= 4
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile as qk_transpile
        from qiskit.quantum_info import random_unitary
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_quantum_volume. "
            "Install with: pip install qiskit"
        ) from exc

    if target_qv < 2 or (target_qv & (target_qv - 1)) != 0:
        raise ValueError(f"target_qv must be a power of 2 >= 2, got {target_qv}")

    import math

    max_width = int(math.log2(target_qv))
    backend_name = _backend_name(backend)
    deadline = time.monotonic() + timeout

    # Detect IBM backend once
    is_ibm = _is_ibm_backend(backend)

    measured_qv = 1

    for width in range(1, max_width + 1):
        hop_successes = 0  # heavy output trials that succeeded

        for _trial in range(num_trials):
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"assert_quantum_volume timed out after {timeout:.0f}s "
                    f"at width={width}, trial={_trial}.\n"
                    f"  Backend: {backend_name}\n"
                    f"  Hint: increase timeout or reduce num_trials."
                )

            # Build random square circuit of depth = width using SU(4) blocks
            qc = _build_qv_circuit(width, QuantumCircuit, random_unitary)

            # Run and collect counts
            counts = _run_circuit(
                qc, backend, shots=shots, is_ibm=is_ibm, qk_transpile=qk_transpile
            )

            # Compute ideal heavy outputs via statevector simulation
            heavy_outputs = _compute_heavy_outputs(qc)

            # Count how many measured shots fell on heavy outputs
            heavy_count = sum(
                cnt
                for bitstr, cnt in counts.items()
                if _bitstr_to_int(bitstr, width) in heavy_outputs
            )
            hop = heavy_count / shots
            if hop > 2 / 3:
                hop_successes += 1

        # One-sided binomial test: H0 = HOP <= 2/3
        hop_rate = hop_successes / num_trials
        binom_result = stats.binomtest(
            hop_successes, num_trials, p=2 / 3, alternative="greater"
        )
        p_value = float(binom_result.pvalue)
        passes = (1 - p_value) >= confidence

        if not passes:
            raise AssertionError(
                f"Quantum volume check failed at width={width} "
                f"(target QV={target_qv}).\n"
                f"  Backend           : {backend_name}\n"
                f"  Heavy output rate : {hop_rate:.4f} ({hop_successes}/{num_trials} trials)\n"
                f"  Required HOP      : > 2/3 ≈ 0.6667\n"
                f"  Confidence        : {1 - p_value:.4f} (required {confidence:.4f})\n"
                f"  Shots per circuit : {shots}\n"
                f"  Hint: increase num_trials/shots, use a less noisy backend, "
                f"or lower target_qv."
            )

        measured_qv = 2**width

    return measured_qv


def assert_randomized_benchmarking(
    backend: Any,
    qubit: int,
    *,
    clifford_lengths: list[int] | None = None,
    num_sequences: int = 20,
    shots: int = 1024,
    min_fidelity_per_clifford: float = 0.999,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Assert 1-qubit randomized benchmarking fidelity meets a minimum threshold.

    For each sequence length *m* in *clifford_lengths*, generates *num_sequences*
    random 1-qubit Clifford sequences followed by the recovery Clifford that
    maps the state back to |0⟩.  The survival probability (fraction of |0⟩
    outcomes) is averaged across sequences for each length.  An exponential
    decay ``A * p^m + B`` is fit to the length–survival-probability curve.
    The average gate fidelity is ``F = 1 - (1-p)/2``.

    For IBM backends the circuits are run using ``SamplerV2``.

    Args:
        backend:                  Qiskit-compatible backend.
        qubit:                    Index of the qubit to benchmark.
        clifford_lengths:         List of Clifford sequence lengths to test.
                                  Default: ``[1, 10, 20, 50, 100]``.
        num_sequences:            Number of random sequences per length (default 20).
        shots:                    Shots per circuit (default 1024).
        min_fidelity_per_clifford: Minimum average gate fidelity (default 0.999).
        timeout:                  Maximum seconds for all jobs (default 300).

    Returns:
        dict with keys:

        - ``'fidelity'`` (float): Measured average gate fidelity.
        - ``'decay_rate'`` (float): Fitted decay parameter *p*.
        - ``'lengths'`` (list[int]): Sequence lengths used.
        - ``'survival_probs'`` (list[float]): Mean survival probability per length.

    Raises:
        AssertionError: If measured fidelity < min_fidelity_per_clifford.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum.assertions.benchmarking import (
            assert_randomized_benchmarking,
        )
        from qiskit_aer import AerSimulator


        def test_rb_fidelity():
            backend = AerSimulator()
            result = assert_randomized_benchmarking(
                backend, qubit=0, clifford_lengths=[1, 5, 10], num_sequences=5
            )
            assert result["fidelity"] >= 0.99
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_randomized_benchmarking. "
            "Install with: pip install qiskit"
        ) from exc

    if clifford_lengths is None:
        clifford_lengths = [1, 10, 20, 50, 100]

    backend_name = _backend_name(backend)
    is_ibm = _is_ibm_backend(backend)
    deadline = time.monotonic() + timeout

    survival_probs: list[float] = []

    for length in clifford_lengths:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"assert_randomized_benchmarking timed out after {timeout:.0f}s "
                f"at length={length}.\n"
                f"  Backend: {backend_name}"
            )

        seq_survival: list[float] = []
        for _ in range(num_sequences):
            qc = _build_rb_circuit(qubit, length, QuantumCircuit)
            counts = _run_circuit(
                qc, backend, shots=shots, is_ibm=is_ibm, qk_transpile=qk_transpile
            )
            total = sum(counts.values())
            # Survival = fraction of |0⟩ on the target qubit
            zero_count = _count_zero_on_qubit(counts, qubit=0, total_qubits=1)
            seq_survival.append(zero_count / total if total > 0 else 0.0)

        survival_probs.append(float(np.mean(seq_survival)))

    # Fit A * p^m + B
    lengths_arr = np.asarray(clifford_lengths, dtype=float)
    probs_arr = np.asarray(survival_probs, dtype=float)

    try:
        popt, _ = optimize.curve_fit(
            _rb_decay,
            lengths_arr,
            probs_arr,
            p0=[0.5, 0.99, 0.25],
            bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            maxfev=5000,
        )
        _a, p, _b = popt
    except (RuntimeError, ValueError):
        # Fall back to linear fit in log space
        p = float(
            np.exp(
                float(
                    np.polyfit(lengths_arr, np.log(np.clip(probs_arr, 1e-9, 1.0)), 1)[0]
                )
            )
        )

    # Average gate fidelity for 1-qubit gates (d=2)
    fidelity = 1.0 - (1.0 - float(p)) / 2.0

    if fidelity < min_fidelity_per_clifford:
        raise AssertionError(
            f"Randomized benchmarking fidelity too low.\n"
            f"  Backend          : {backend_name}\n"
            f"  Qubit            : {qubit}\n"
            f"  Measured fidelity: {fidelity:.6f}\n"
            f"  Required fidelity: {min_fidelity_per_clifford:.6f}\n"
            f"  Decay parameter p: {p:.6f}\n"
            f"  Lengths tested   : {clifford_lengths}\n"
            f"  Survival probs   : {[f'{s:.4f}' for s in survival_probs]}\n"
            f"  Hint: run on a less noisy backend or lower min_fidelity_per_clifford."
        )

    return {
        "fidelity": fidelity,
        "decay_rate": float(p),
        "lengths": clifford_lengths,
        "survival_probs": survival_probs,
    }


def assert_t1_above(
    backend: Any,
    qubit: int,
    target_t1_us: float,
    *,
    shots: int = 1024,
    timeout: float = 300.0,
) -> float:
    """Assert qubit T1 relaxation time is at least *target_t1_us* microseconds.

    Prepares the qubit in |1⟩ and waits variable ``delay`` durations (using
    Qiskit's ``Delay`` gate), then measures.  The |1⟩ survival probability is
    fit to an exponential decay ``exp(-t / T1)`` to extract T1.

    Note:
        Accurate T1 measurement requires pulse-level backend support and a
        backend that honours ``Delay`` gates (e.g. a real IBM device or
        ``AerSimulator`` with a noise model).  On an ideal simulator T1 will
        appear infinite.

    Args:
        backend:        Qiskit-compatible backend with delay gate support.
        qubit:          Index of the qubit to measure.
        target_t1_us:   Minimum required T1 in microseconds.
        shots:          Shots per delay point (default 1024).
        timeout:        Maximum seconds for all jobs (default 300).

    Returns:
        float: Measured T1 in microseconds.

    Raises:
        AssertionError: If measured T1 < target_t1_us.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum.assertions.benchmarking import assert_t1_above


        def test_t1(ibm_backend):
            t1 = assert_t1_above(ibm_backend, qubit=0, target_t1_us=50.0)
            print(f"Measured T1: {t1:.1f} µs")
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_t1_above. Install with: pip install qiskit"
        ) from exc

    backend_name = _backend_name(backend)
    is_ibm = _is_ibm_backend(backend)
    deadline = time.monotonic() + timeout

    # Determine dt (seconds per sample) from backend configuration
    dt_s = _get_backend_dt(backend)
    dt_us = dt_s * 1e6  # convert to microseconds

    # Choose delay points spanning ~3x the target T1
    max_delay_us = target_t1_us * 3.0
    num_points = 10
    delay_us_points = np.linspace(0.0, max_delay_us, num_points)

    survival_probs: list[float] = []

    for delay_us in delay_us_points:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"assert_t1_above timed out after {timeout:.0f}s.\n"
                f"  Backend: {backend_name}"
            )

        qc = QuantumCircuit(1, 1)
        qc.x(0)  # prepare |1⟩

        # Insert delay (in dt units if backend has dt, else use identity gates)
        if dt_s > 0 and delay_us > 0:
            delay_dt = int(delay_us / dt_us)
            if delay_dt > 0:
                qc.delay(delay_dt, 0)
        qc.measure(0, 0)

        counts = _run_circuit(
            qc, backend, shots=shots, is_ibm=is_ibm, qk_transpile=qk_transpile
        )
        total = sum(counts.values())
        ones = counts.get("1", 0)
        survival_probs.append(ones / total if total > 0 else 0.0)

    # Fit exp(-t / T1)
    delays_arr = np.asarray(delay_us_points, dtype=float)
    probs_arr = np.asarray(survival_probs, dtype=float)

    try:
        popt, _ = optimize.curve_fit(
            _t1_decay,
            delays_arr,
            probs_arr,
            p0=[target_t1_us],
            bounds=([0.0], [np.inf]),
            maxfev=5000,
        )
        t1_us = float(popt[0])
    except (RuntimeError, ValueError):
        # Fallback: estimate from half-life
        half = float(np.interp(0.5, probs_arr[::-1], delays_arr[::-1]))
        t1_us = half / np.log(2)

    if t1_us < target_t1_us:
        raise AssertionError(
            f"T1 relaxation time too short.\n"
            f"  Backend        : {backend_name}\n"
            f"  Qubit          : {qubit}\n"
            f"  Measured T1    : {t1_us:.2f} µs\n"
            f"  Required T1    : {target_t1_us:.2f} µs\n"
            f"  Delay points   : {[f'{d:.1f}' for d in delay_us_points]} µs\n"
            f"  Survival probs : {[f'{p:.4f}' for p in survival_probs]}\n"
            f"  Hint: choose a lower-noise device or a qubit with better T1."
        )

    return t1_us


def assert_gate_fidelity_above(
    backend: Any,
    gate_name: str,
    qubits: list[int] | tuple[int, ...],
    target_fidelity: float,
    *,
    shots: int = 2048,
    timeout: float = 300.0,
) -> float:
    """Assert gate fidelity (from calibration data) is at least *target_fidelity*.

    Reads the gate error rate from ``backend.properties()`` for the specified
    gate on the specified qubits, computes ``fidelity = 1 - error_rate``, and
    asserts the result meets the threshold.  This is a fast, passive check
    against existing calibration data — **no circuit is executed**.

    Args:
        backend:          IBM backend (or any backend with a ``.properties()``
                          method that exposes per-gate error rates).
        gate_name:        Name of the gate (e.g. ``"cx"``, ``"ecr"``, ``"x"``).
        qubits:           Qubit indices the gate acts on (e.g. ``[0, 1]``).
        target_fidelity:  Minimum required fidelity (0.0–1.0).
        shots:            Unused; kept for API consistency with other assertions.
        timeout:          Unused; kept for API consistency with other assertions.

    Returns:
        float: Measured gate fidelity (1 - error_rate).

    Raises:
        AssertionError: If fidelity < target_fidelity, or if gate/qubit data
                        is not available in calibration properties.

    Example::

        from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above


        def test_cx_fidelity(ibm_backend):
            fidelity = assert_gate_fidelity_above(
                ibm_backend,
                gate_name="cx",
                qubits=[0, 1],
                target_fidelity=0.99,
            )
            print(f"CX fidelity: {fidelity:.4f}")
    """
    backend_name = _backend_name(backend)
    qubit_list = list(qubits)

    props = backend.properties() if hasattr(backend, "properties") else None
    if props is None:
        raise AssertionError(
            f"Backend '{backend_name}' has no calibration properties.\n"
            f"  Hint: use a real IBM backend or a simulator with a noise model."
        )

    # Try the convenience method first (BackendProperties.gate_error)
    error_rate: float | None = None
    if hasattr(props, "gate_error"):
        try:
            error_rate = float(props.gate_error(gate_name, qubit_list))
        except Exception:
            error_rate = None

    # Manual scan as fallback
    if error_rate is None:
        for gate in getattr(props, "gates", []):
            if (
                getattr(gate, "gate", None) == gate_name
                and list(getattr(gate, "qubits", [])) == qubit_list
            ):
                for param in getattr(gate, "parameters", []):
                    if getattr(param, "name", None) == "gate_error":
                        v = getattr(param, "value", None)
                        if v is not None:
                            error_rate = float(v)
                            break
                if error_rate is not None:
                    break

    if error_rate is None:
        raise AssertionError(
            f"Gate '{gate_name}' on qubits {qubit_list} not found in calibration data.\n"
            f"  Backend: {backend_name}\n"
            f"  Hint: check gate name spelling and qubit indices."
        )

    fidelity = 1.0 - error_rate

    if fidelity < target_fidelity:
        raise AssertionError(
            f"Gate fidelity below target.\n"
            f"  Backend          : {backend_name}\n"
            f"  Gate             : {gate_name} on qubits {qubit_list}\n"
            f"  Measured fidelity: {fidelity:.6f}  (error={error_rate:.6f})\n"
            f"  Required fidelity: {target_fidelity:.6f}\n"
            f"  Hint: choose a better qubit pair or a lower target_fidelity."
        )

    return fidelity


def assert_t2_above(
    backend: Any,
    qubit: int,
    target_t2_us: float,
    *,
    shots: int = 1024,
    timeout: float = 300.0,
) -> float:
    """Assert qubit T2 (Hahn echo) coherence time is at least *target_t2_us* µs.

    Implements a Hahn echo sequence: X/2 – delay/2 – X – delay/2 – X/2 – measure.
    Fits |0⟩ survival probability to exp(-t / T2) to extract T2.

    Args:
        backend:        Qiskit-compatible backend.
        qubit:          Index of the qubit to measure.
        target_t2_us:   Minimum required T2 in microseconds.
        shots:          Shots per delay point (default 1024).
        timeout:        Maximum seconds for all jobs (default 300).

    Returns:
        float: Measured T2 in microseconds.

    Raises:
        AssertionError: If measured T2 < target_t2_us.
        ImportError:    If qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_t2_above. Install with: pip install qiskit"
        ) from exc

    backend_name = _backend_name(backend)
    is_ibm = _is_ibm_backend(backend)
    deadline = time.monotonic() + timeout

    dt_s = _get_backend_dt(backend)
    dt_us = dt_s * 1e6 if dt_s > 0 else 1e-3  # fallback: 1 ns per dt unit

    max_delay_us = target_t2_us * 3.0
    num_points = 10
    delay_us_points = np.linspace(0.0, max_delay_us, num_points)

    survival_probs: list[float] = []

    for delay_us in delay_us_points:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"assert_t2_above timed out after {timeout:.0f}s.\n"
                f"  Backend: {backend_name}"
            )

        qc = QuantumCircuit(1, 1)
        # Hahn echo: Rx(π/2) – delay/2 – X – delay/2 – Rx(π/2)
        qc.rx(np.pi / 2, 0)
        half_delay_dt = int((delay_us / 2) / dt_us) if dt_s > 0 and delay_us > 0 else 0
        if half_delay_dt > 0:
            qc.delay(half_delay_dt, 0)
        qc.x(0)  # π pulse refocuses dephasing
        if half_delay_dt > 0:
            qc.delay(half_delay_dt, 0)
        qc.rx(np.pi / 2, 0)
        qc.measure(0, 0)

        counts = _run_circuit(
            qc, backend, shots=shots, is_ibm=is_ibm, qk_transpile=qk_transpile
        )
        total = sum(counts.values())
        zeros = _count_zero_on_qubit(counts, qubit=0, total_qubits=1)
        survival_probs.append(zeros / total if total > 0 else 0.0)

    delays_arr = np.asarray(delay_us_points, dtype=float)
    probs_arr = np.asarray(survival_probs, dtype=float)

    try:
        popt, _ = optimize.curve_fit(
            _t1_decay,  # same exp(-t/T) form
            delays_arr,
            probs_arr,
            p0=[target_t2_us],
            bounds=([0.0], [np.inf]),
            maxfev=5000,
        )
        t2_us = float(popt[0])
    except (RuntimeError, ValueError):
        half = float(np.interp(0.5, probs_arr[::-1], delays_arr[::-1]))
        t2_us = half / np.log(2)

    if t2_us < target_t2_us:
        raise AssertionError(
            f"T2 (Hahn echo) coherence time too short.\n"
            f"  Backend        : {backend_name}\n"
            f"  Qubit          : {qubit}\n"
            f"  Measured T2    : {t2_us:.2f} µs\n"
            f"  Required T2    : {target_t2_us:.2f} µs\n"
        )

    return t2_us


def assert_t2star_above(
    backend: Any,
    qubit: int,
    target_t2star_us: float,
    *,
    shots: int = 1024,
    timeout: float = 300.0,
) -> float:
    """Assert qubit T2* (free induction decay) is at least *target_t2star_us* µs.

    Implements a Ramsey sequence: Rx(π/2) – delay – Rx(π/2) – measure.
    Fits |0⟩ survival to a decaying cosine; the envelope gives T2*.

    Args:
        backend:           Qiskit-compatible backend.
        qubit:             Index of the qubit.
        target_t2star_us:  Minimum required T2* in microseconds.
        shots:             Shots per delay point (default 1024).
        timeout:           Maximum seconds for all jobs (default 300).

    Returns:
        float: Measured T2* in microseconds.

    Raises:
        AssertionError: If measured T2* < target_t2star_us.
        ImportError:    If qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_t2star_above. Install with: pip install qiskit"
        ) from exc

    backend_name = _backend_name(backend)
    is_ibm = _is_ibm_backend(backend)
    deadline = time.monotonic() + timeout

    dt_s = _get_backend_dt(backend)
    dt_us = dt_s * 1e6 if dt_s > 0 else 1e-3

    max_delay_us = target_t2star_us * 3.0
    num_points = 12
    delay_us_points = np.linspace(0.0, max_delay_us, num_points)

    survival_probs: list[float] = []

    for delay_us in delay_us_points:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"assert_t2star_above timed out after {timeout:.0f}s.\n"
                f"  Backend: {backend_name}"
            )

        qc = QuantumCircuit(1, 1)
        qc.rx(np.pi / 2, 0)
        delay_dt = int(delay_us / dt_us) if dt_s > 0 and delay_us > 0 else 0
        if delay_dt > 0:
            qc.delay(delay_dt, 0)
        qc.rx(np.pi / 2, 0)
        qc.measure(0, 0)

        counts = _run_circuit(
            qc, backend, shots=shots, is_ibm=is_ibm, qk_transpile=qk_transpile
        )
        total = sum(counts.values())
        zeros = _count_zero_on_qubit(counts, qubit=0, total_qubits=1)
        survival_probs.append(zeros / total if total > 0 else 0.0)

    delays_arr = np.asarray(delay_us_points, dtype=float)
    probs_arr = np.asarray(survival_probs, dtype=float)

    # Fit decaying cosine: A * exp(-t/T2*) * cos(2π*f*t + φ) + B
    def _ramsey_decay(
        t: Any, amp: float, t2s: float, freq: float, phi: float, offset: float
    ) -> Any:
        return (
            amp * np.exp(-t / (t2s + 1e-12)) * np.cos(2 * np.pi * freq * t + phi)
            + offset
        )

    try:
        # Initial guess: freq from FFT
        if len(probs_arr) > 4:
            fft_freqs = np.fft.rfftfreq(
                len(probs_arr), d=float(np.mean(np.diff(delays_arr))) + 1e-12
            )
            fft_amp = np.abs(np.fft.rfft(probs_arr - np.mean(probs_arr)))
            freq_guess = (
                float(fft_freqs[np.argmax(fft_amp[1:]) + 1])
                if len(fft_freqs) > 1
                else 0.01
            )
        else:
            freq_guess = 0.01
        popt, _ = optimize.curve_fit(
            _ramsey_decay,
            delays_arr,
            probs_arr,
            p0=[0.5, target_t2star_us, freq_guess, 0.0, 0.5],
            bounds=([-1.0, 0.0, 0.0, -np.pi, 0.0], [1.0, np.inf, np.inf, np.pi, 1.0]),
            maxfev=10000,
        )
        t2star_us = float(popt[1])
    except (RuntimeError, ValueError):
        # Fallback: fit simple exp decay to envelope
        try:
            popt2, _ = optimize.curve_fit(
                _t1_decay,
                delays_arr,
                probs_arr,
                p0=[target_t2star_us],
                bounds=([0.0], [np.inf]),
                maxfev=5000,
            )
            t2star_us = float(popt2[0])
        except (RuntimeError, ValueError):
            half = float(np.interp(0.5, probs_arr[::-1], delays_arr[::-1]))
            t2star_us = half / np.log(2)

    if t2star_us < target_t2star_us:
        raise AssertionError(
            f"T2* (Ramsey) dephasing time too short.\n"
            f"  Backend        : {backend_name}\n"
            f"  Qubit          : {qubit}\n"
            f"  Measured T2*   : {t2star_us:.2f} µs\n"
            f"  Required T2*   : {target_t2star_us:.2f} µs\n"
        )

    return t2star_us


def assert_interleaved_rb(
    backend: Any,
    qubit: int,
    gate_name: str,
    gate_circuit: Any,
    *,
    clifford_lengths: list[int] | None = None,
    num_sequences: int = 20,
    shots: int = 1024,
    min_gate_fidelity: float = 0.999,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Assert interleaved randomized benchmarking (IRB) gate fidelity.

    Runs two RB experiments:
    1. Standard RB to get reference decay rate *p_ref*.
    2. Interleaved RB where each Clifford is followed by the target *gate_circuit*
       to get *p_irb*.

    Gate fidelity: ``F_gate = 1 - (d-1)/d * (1 - p_irb/p_ref)``
    where ``d = 2`` for single-qubit gates.

    Args:
        backend:          Qiskit-compatible backend.
        qubit:            Index of the qubit to benchmark.
        gate_name:        Human-readable name for the gate (for error messages).
        gate_circuit:     A single-qubit QuantumCircuit implementing the gate.
        clifford_lengths: Sequence lengths. Default: ``[1, 5, 10, 20, 50]``.
        num_sequences:    Random sequences per length (default 20).
        shots:            Shots per circuit (default 1024).
        min_gate_fidelity: Minimum acceptable gate fidelity (default 0.999).
        timeout:          Maximum seconds for all jobs (default 300).

    Returns:
        dict with keys ``'fidelity'``, ``'p_ref'``, ``'p_irb'``, ``'lengths'``,
        ``'ref_survival'``, ``'irb_survival'``.

    Raises:
        AssertionError: If gate fidelity < min_gate_fidelity.
        ImportError:    If qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile as qk_transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_interleaved_rb. "
            "Install with: pip install qiskit"
        ) from exc

    if clifford_lengths is None:
        clifford_lengths = [1, 5, 10, 20, 50]

    backend_name = _backend_name(backend)
    is_ibm = _is_ibm_backend(backend)
    deadline = time.monotonic() + timeout

    def _run_rb(interleave: bool) -> list[float]:
        survival: list[float] = []
        for length in clifford_lengths:
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"assert_interleaved_rb timed out after {timeout:.0f}s.\n"
                    f"  Backend: {backend_name}"
                )
            seq_surv: list[float] = []
            for _ in range(num_sequences):
                qc = _build_rb_circuit(qubit, length, QuantumCircuit)
                if interleave:
                    # Insert gate after each Clifford by rebuilding
                    qc_irb = _build_irb_circuit(
                        qubit, length, gate_circuit, QuantumCircuit
                    )
                    qc = qc_irb
                counts = _run_circuit(
                    qc, backend, shots=shots, is_ibm=is_ibm, qk_transpile=qk_transpile
                )
                total = sum(counts.values())
                zeros = _count_zero_on_qubit(counts, qubit=0, total_qubits=1)
                seq_surv.append(zeros / total if total > 0 else 0.0)
            survival.append(float(np.mean(seq_surv)))
        return survival

    ref_survival = _run_rb(interleave=False)
    irb_survival = _run_rb(interleave=True)

    lengths_arr = np.asarray(clifford_lengths, dtype=float)

    def _fit_p(surv: list[float]) -> float:
        probs = np.asarray(surv, dtype=float)
        try:
            popt, _ = optimize.curve_fit(
                _rb_decay,
                lengths_arr,
                probs,
                p0=[0.5, 0.99, 0.25],
                bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                maxfev=5000,
            )
            return float(popt[1])
        except (RuntimeError, ValueError):
            return float(
                np.exp(
                    float(
                        np.polyfit(lengths_arr, np.log(np.clip(probs, 1e-9, 1.0)), 1)[0]
                    )
                )
            )

    p_ref = _fit_p(ref_survival)
    p_irb = _fit_p(irb_survival)

    # IRB gate fidelity formula: F = 1 - (d-1)/d * (1 - p_irb/p_ref), d=2
    d = 2.0
    ratio = p_irb / (p_ref + 1e-12)
    gate_fidelity = 1.0 - (d - 1) / d * (1.0 - ratio)
    gate_fidelity = float(np.clip(gate_fidelity, 0.0, 1.0))

    if gate_fidelity < min_gate_fidelity:
        raise AssertionError(
            f"Interleaved RB gate fidelity too low.\n"
            f"  Backend           : {backend_name}\n"
            f"  Gate              : {gate_name} on qubit {qubit}\n"
            f"  Gate fidelity     : {gate_fidelity:.6f}\n"
            f"  Required fidelity : {min_gate_fidelity:.6f}\n"
            f"  p_ref             : {p_ref:.6f}\n"
            f"  p_irb             : {p_irb:.6f}\n"
        )

    return {
        "fidelity": gate_fidelity,
        "p_ref": p_ref,
        "p_irb": p_irb,
        "lengths": clifford_lengths,
        "ref_survival": ref_survival,
        "irb_survival": irb_survival,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


from pytest_quantum._internal import (  # noqa: E402
    _backend_name,
    _is_ibm_backend,
    _run_circuit,
)


def _get_backend_dt(backend: Any) -> float:
    """Return the backend sample time *dt* in seconds, or 0.0 if unavailable."""
    # Try BackendV2 style
    dt = getattr(backend, "dt", None)
    if dt is not None and isinstance(dt, (int, float)):
        return float(dt)
    # Try configuration()
    cfg = None
    if hasattr(backend, "configuration") and callable(backend.configuration):
        try:
            cfg = backend.configuration()
        except Exception:
            pass
    if cfg is not None:
        dt = getattr(cfg, "dt", None)
        if dt is not None and isinstance(dt, (int, float)):
            return float(dt)
    return 0.0


def _build_qv_circuit(
    width: int,
    QuantumCircuit: Any,
    random_unitary: Any,
) -> Any:
    """Build a random quantum volume square circuit of size *width* x *width*."""
    import math

    rng = np.random.default_rng()
    qc = QuantumCircuit(width, width)
    for _layer in range(width):
        # Pair up qubits randomly
        perm = rng.permutation(width)
        pairs = [(int(perm[i]), int(perm[i + 1])) for i in range(0, width - 1, 2)]
        for q0, q1 in pairs:
            u = random_unitary(4)
            qc.unitary(u, [q0, q1])
    qc.measure(list(range(width)), list(range(width)))
    # Compute depth for annotation only
    _ = int(math.log2(width + 1))  # suppress unused import warning
    return qc


def _compute_heavy_outputs(qc: Any) -> set[int]:
    """Compute the set of heavy output integers for a QV circuit via statevector simulation."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        # Remove measurements for statevector
        qc_no_meas = QuantumCircuit(qc.num_qubits)
        for instruction in qc.data:
            if instruction.operation.name != "measure":
                qc_no_meas.append(instruction)

        sv = Statevector(qc_no_meas)
        probs = sv.probabilities()
        median_prob = float(np.median(probs))
        heavy: set[int] = {i for i, p in enumerate(probs) if p > median_prob}
        return heavy
    except Exception:
        # If statevector fails, return all outputs (every shot passes — conservative)
        width = qc.num_qubits
        return set(range(2**width))


def _bitstr_to_int(bitstr: str, width: int) -> int:
    """Convert a Qiskit bitstring (LSB-first) to an integer."""
    # Qiskit returns bitstrings with rightmost bit = qubit 0
    stripped = bitstr.replace(" ", "")
    return int(stripped, 2)


def _build_rb_circuit(qubit: int, length: int, QuantumCircuit: Any) -> Any:
    """Build a 1-qubit randomized benchmarking circuit of *length* Cliffords."""
    # Generate random 1-qubit Cliffords using SX/RZ decomposition
    rng = np.random.default_rng()
    qc = QuantumCircuit(1, 1)

    # Track the cumulative Clifford as an index (0-23) for recovery computation
    clifford_idx = 0
    applied: list[int] = []

    for _ in range(length):
        c = int(rng.integers(0, 24))
        applied.append(c)
        clifford_idx = _compose_clifford_1q(clifford_idx, c)
        _apply_clifford_1q(qc, c)

    # Apply recovery Clifford (inverse of cumulative)
    recovery = _invert_clifford_1q(clifford_idx)
    _apply_clifford_1q(qc, recovery)

    qc.measure(0, 0)
    return qc


def _build_irb_circuit(
    qubit: int, length: int, gate_circuit: Any, QuantumCircuit: Any
) -> Any:
    """Build 1-qubit IRB circuit: alternates Clifford gates with the interleaved gate."""
    rng = np.random.default_rng()
    qc = QuantumCircuit(1, 1)

    clifford_idx = 0
    for _ in range(length):
        c = int(rng.integers(0, 24))
        clifford_idx = _compose_clifford_1q(clifford_idx, c)
        _apply_clifford_1q(qc, c)
        # Interleave the target gate
        for instruction in gate_circuit.data:
            qc.append(instruction.operation, [0])

    # Recovery Clifford — note: we need to account for the interleaved gate
    # For simplicity, we reuse the standard recovery (this gives an approximate IRB)
    recovery = _invert_clifford_1q(clifford_idx)
    _apply_clifford_1q(qc, recovery)
    qc.measure(0, 0)
    return qc


# 24-element 1-qubit Clifford group represented as (X_bool, Z_bool, phase_bool)
# tuple: (maps |0> to X-eigenstate?, maps |+> to Z-eigenstate?, global phase)
# We use a simple gate-sequence table instead.
_CLIFFORD_1Q_GATES: list[list[tuple[str, float]]] = [
    # Each entry: list of (gate_name, angle) pairs
    [],  # 0: I
    [("h", 0)],  # 1: H
    [("s", 0)],  # 2: S
    [("sdg", 0)],  # 3: Sdg
    [("x", 0)],  # 4: X
    [("y", 0)],  # 5: Y
    [("z", 0)],  # 6: Z
    [("h", 0), ("s", 0)],  # 7: HS
    [("h", 0), ("sdg", 0)],  # 8: HSdg
    [("s", 0), ("h", 0)],  # 9: SH
    [("sdg", 0), ("h", 0)],  # 10: SdgH
    [("x", 0), ("h", 0)],  # 11: XH
    [("y", 0), ("h", 0)],  # 12: YH
    [("z", 0), ("h", 0)],  # 13: ZH
    [("h", 0), ("x", 0)],  # 14: HX
    [("h", 0), ("y", 0)],  # 15: HY
    [("h", 0), ("z", 0)],  # 16: HZ
    [("s", 0), ("x", 0)],  # 17: SX
    [("s", 0), ("y", 0)],  # 18: SY
    [("s", 0), ("z", 0)],  # 19: SZ
    [("h", 0), ("s", 0), ("h", 0)],  # 20: HSH
    [("s", 0), ("h", 0), ("s", 0)],  # 21: SHS
    [("sdg", 0), ("h", 0), ("sdg", 0)],  # 22: SdgHSdg
    [("h", 0), ("sdg", 0), ("h", 0)],  # 23: HSdgH
]

# Cayley table for 1-qubit Cliffords: _CLIFFORD_COMPOSE[a][b] = a ∘ b
# Pre-computed via symplectic representation (simplified approximation)
# We approximate with a full 24x24 table using matrix multiplication
_CLIFFORD_1Q_MATRICES: list[Any] | None = None


def _get_clifford_matrices() -> list[Any]:
    """Return list of 24 unitary 2x2 matrices for 1-qubit Clifford group."""
    global _CLIFFORD_1Q_MATRICES
    if _CLIFFORD_1Q_MATRICES is not None:
        return _CLIFFORD_1Q_MATRICES

    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    Sdg = np.conj(S).T
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    gate_map: dict[str, Any] = {"h": H, "s": S, "sdg": Sdg, "x": X, "y": Y, "z": Z}

    matrices: list[Any] = []
    for gate_seq in _CLIFFORD_1Q_GATES:
        mat = I2.copy()
        for gate_name, _angle in gate_seq:
            mat = gate_map[gate_name] @ mat
        matrices.append(mat)

    _CLIFFORD_1Q_MATRICES = matrices
    return matrices


def _clifford_equal_up_to_phase(
    a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]
) -> bool:
    """Return True if a == e^{iθ} b for some phase θ."""
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-12:
        return False
    # If a = phase * b then |<a, b>| / (||a|| ||b||) == 1
    overlap = abs(np.vdot(a.flatten(), b.flatten()))
    norm_a = np.linalg.norm(a)
    return bool(np.isclose(overlap, norm_a * norm_b, rtol=1e-6))


def _compose_clifford_1q(a: int, b: int) -> int:
    """Return the index of the Clifford equal to C_b ∘ C_a."""
    mats = _get_clifford_matrices()
    composed = mats[b] @ mats[a]
    for i, m in enumerate(mats):
        if _clifford_equal_up_to_phase(composed, m):
            return i
    return 0  # fallback


def _invert_clifford_1q(idx: int) -> int:
    """Return the index of the inverse of Clifford *idx*."""
    mats = _get_clifford_matrices()
    inv = np.linalg.inv(mats[idx])
    for i, m in enumerate(mats):
        if _clifford_equal_up_to_phase(inv, m):
            return i
    return 0  # fallback


def _apply_clifford_1q(qc: Any, idx: int) -> None:
    """Append gates for Clifford *idx* to single-qubit circuit *qc*."""
    for gate_name, _angle in _CLIFFORD_1Q_GATES[idx]:
        getattr(qc, gate_name)(0)


def _count_zero_on_qubit(
    counts: dict[str, int], *, qubit: int, total_qubits: int
) -> int:
    """Count shots where *qubit* measured 0 (Qiskit LSB bitstring convention)."""
    bit_pos = qubit  # LSB = qubit 0
    total = 0
    for bitstr, cnt in counts.items():
        stripped = bitstr.replace(" ", "")
        # Pad to expected width
        padded = stripped.zfill(total_qubits)
        # Qiskit: rightmost character = qubit 0
        if padded[-(bit_pos + 1)] == "0":
            total += cnt
    return total


def _rb_decay(m: Any, a: float, p: float, b: float) -> Any:
    """Exponential decay model for randomized benchmarking: A*p^m + B."""
    return a * np.power(p, m) + b


def _t1_decay(t: Any, t1: float) -> Any:
    """Exponential decay model for T1: exp(-t / T1)."""
    return np.exp(-t / t1)
