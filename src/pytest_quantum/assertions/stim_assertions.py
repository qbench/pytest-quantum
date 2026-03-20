"""Assertions for Stim stabilizer circuits and quantum error correction."""

from __future__ import annotations

from typing import Any

import numpy as np


def assert_stim_logical_error_rate_below(
    circuit: Any,
    max_error_rate: float,
    *,
    shots: int = 10_000,
    seed: int | None = None,
) -> None:
    """Assert that a Stim QEC circuit has logical error rate below threshold.

    The circuit must contain DETECTOR and OBSERVABLE_INCLUDE instructions.

    Args:
        circuit:        ``stim.Circuit`` with detectors and observables.
        max_error_rate: Maximum allowed logical error rate.
        shots:          Number of shots (default 10,000).
        seed:           Optional random seed.

    Raises:
        AssertionError: If logical error rate exceeds *max_error_rate*.
        ImportError:    If stim is not installed.
        ValueError:     If the circuit has no observables.

    Example::

        import stim
        from pytest_quantum import assert_stim_logical_error_rate_below

        c = stim.Circuit.generated(
            "repetition_code:memory",
            rounds=3,
            distance=3,
            after_clifford_depolarization=0.001,
        )
        assert_stim_logical_error_rate_below(c, max_error_rate=0.05, shots=1000)
    """
    try:
        import stim  # noqa: F401
    except ImportError as exc:
        raise ImportError("stim is required: pip install stim") from exc

    if circuit.num_observables == 0:
        raise ValueError(
            "Circuit has no observables (OBSERVABLE_INCLUDE instructions). "
            "Add OBSERVABLE_INCLUDE to track logical qubit errors."
        )

    sampler = circuit.compile_detector_sampler(seed=seed)
    _det_data, obs_data = sampler.sample(shots, separate_observables=True)

    logical_errors = np.any(obs_data, axis=1)
    error_rate = float(np.mean(logical_errors))

    if error_rate > max_error_rate:
        raise AssertionError(
            f"Logical error rate {error_rate:.4f} exceeds threshold "
            f"{max_error_rate:.4f}\n"
            f"  Shots: {shots}\n"
            f"  Logical errors: {int(np.sum(logical_errors))}/{shots}\n"
            f"  Observables tracked: {circuit.num_observables}"
        )


def assert_stim_detector_error_rate_below(
    circuit: Any,
    max_error_rate: float,
    *,
    shots: int = 10_000,
    seed: int | None = None,
) -> None:
    """Assert that the mean detector error rate is below threshold.

    Useful for verifying that a noise model produces errors at the expected
    rate.

    Args:
        circuit:        ``stim.Circuit`` with DETECTOR instructions and noise.
        max_error_rate: Maximum allowed mean detector error rate.
        shots:          Number of shots (default 10,000).
        seed:           Optional random seed.

    Raises:
        AssertionError: If mean detector error rate exceeds *max_error_rate*.
        ImportError:    If stim is not installed.
        ValueError:     If the circuit has no detectors.
    """
    try:
        import stim  # noqa: F401
    except ImportError as exc:
        raise ImportError("stim is required: pip install stim") from exc

    if circuit.num_detectors == 0:
        raise ValueError("Circuit has no detectors (DETECTOR instructions).")

    sampler = circuit.compile_detector_sampler(seed=seed)
    det_data = sampler.sample(shots)

    per_detector_rates = np.mean(det_data, axis=0)
    mean_rate = float(np.mean(per_detector_rates))

    if mean_rate > max_error_rate:
        raise AssertionError(
            f"Mean detector error rate {mean_rate:.4f} exceeds threshold "
            f"{max_error_rate:.4f}\n"
            f"  Detectors: {circuit.num_detectors}\n"
            f"  Max single-detector rate: "
            f"{float(np.max(per_detector_rates)):.4f}"
        )


def assert_stabilizer_state(
    tableau_simulator: Any,
    expected_stabilizers: list[str],
) -> None:
    """Assert a Stim TableauSimulator is in the expected stabilizer state.

    Args:
        tableau_simulator:   ``stim.TableauSimulator`` after running a circuit.
        expected_stabilizers: List of Pauli strings, e.g. ``["+XX", "+ZZ"]``.

    Raises:
        AssertionError: If any stabilizer is not satisfied (expectation ≠ +1).
        ImportError:    If stim is not installed.

    Example::

        import stim
        from pytest_quantum import assert_stabilizer_state

        sim = stim.TableauSimulator()
        sim.h(0)
        sim.cnot(0, 1)
        assert_stabilizer_state(sim, ["+XX", "+ZZ"])
    """
    try:
        import stim
    except ImportError as exc:
        raise ImportError("stim is required: pip install stim") from exc

    failing = []
    for stab_str in expected_stabilizers:
        p = stim.PauliString(stab_str)
        expectation = tableau_simulator.peek_observable_expectation(p)
        if expectation != 1:
            failing.append(f"  {stab_str!r}: expectation = {expectation} (expected +1)")

    if failing:
        raise AssertionError(
            f"Stabilizer state check failed for "
            f"{len(failing)}/{len(expected_stabilizers)} stabilizers:\n"
            + "\n".join(failing)
        )
