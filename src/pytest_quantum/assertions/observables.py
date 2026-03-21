"""Expectation value and observable assertions for VQE/QAOA testing."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "assert_cost_decreases",
    "assert_expectation_value_close",
    "assert_ground_state_energy_close",
    "assert_vqe_converges",
]


def assert_expectation_value_close(
    result_or_value: Any,
    expected: float,
    *,
    atol: float = 0.1,
) -> None:
    """Assert a measured expectation value is close to expected.

    Accepts: plain float/int, numpy scalar, Qiskit EstimatorResult/PrimitiveResult,
    or PennyLane measurement result.

    Args:
        result_or_value: Measured expectation value or result object.
        expected:        Expected value.
        atol:            Absolute tolerance (default 0.1).

    Raises:
        AssertionError: If |actual - expected| > atol.
        TypeError:      If result type is not recognised.

    Example::

        from pytest_quantum import assert_expectation_value_close

        assert_expectation_value_close(0.95, expected=1.0, atol=0.1)
    """
    actual = _extract_expectation_value(result_or_value)
    diff = abs(actual - expected)
    if diff <= atol:
        return
    raise AssertionError(
        f"Expectation value mismatch.\n"
        f"  Expected : {expected}\n"
        f"  Actual   : {actual:.6f}\n"
        f"  |diff|   : {diff:.6f}   (tolerance: {atol})\n"
        f"  Hint: increase atol for shot-based estimators, or increase shots."
    )


def assert_ground_state_energy_close(
    result_or_value: Any,
    expected_energy: float,
    *,
    atol: float = 0.1,
) -> None:
    """Assert VQE/QAOA result is close to the known ground state energy.

    Args:
        result_or_value: Measured energy or EstimatorResult.
        expected_energy: Known ground state energy (e.g. from exact diagonalisation).
        atol:            Absolute tolerance in energy units (default 0.1).

    Raises:
        AssertionError: If measured energy differs from expected by more than atol.

    Example::

        from pytest_quantum import assert_ground_state_energy_close

        # H2 ground state energy (Hartree)
        assert_ground_state_energy_close(-1.85, expected_energy=-1.8572, atol=0.05)
    """
    actual = _extract_expectation_value(result_or_value)
    diff = abs(actual - expected_energy)
    if diff <= atol:
        return
    raise AssertionError(
        f"Ground state energy mismatch.\n"
        f"  Expected energy : {expected_energy:.6f}\n"
        f"  Measured energy : {actual:.6f}\n"
        f"  |diff|          : {diff:.6f}   (tolerance: {atol})\n"
        f"  Hint: check VQE convergence — more iterations or a deeper ansatz."
    )


def assert_vqe_converges(
    cost_function: Any,
    initial_params: Any,
    *,
    method: str = "COBYLA",
    max_iterations: int = 200,
    expected_minimum: float | None = None,
    atol: float = 0.1,
    rtol: float = 0.0,
) -> None:
    """Assert a VQE / variational optimization converges.

    Runs a full optimization loop and checks:
    1. The final cost is lower than the initial cost (energy decreased).
    2. If ``expected_minimum`` is given, the result is within ``atol`` of it.

    Args:
        cost_function: Callable mapping parameter array -> float (e.g. a QNode
            returning ``qml.expval(H)``).
        initial_params: Starting parameter vector (list or numpy array).
        method: SciPy optimizer (default ``"COBYLA"``; gradient-free, good for
            noisy quantum hardware).
        max_iterations: Maximum optimizer iterations (default 200).
        expected_minimum: Known ground-state energy to compare against.
        atol: Absolute tolerance when comparing against expected_minimum.
        rtol: Relative tolerance (default 0, i.e. absolute only).

    Raises:
        AssertionError: If energy did not decrease or result misses expected_minimum.

    Example::

        import pennylane as qml
        import numpy as np
        from pytest_quantum import assert_vqe_converges

        dev = qml.device("default.qubit", wires=1)


        @qml.qnode(dev)
        def circuit(theta):
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))


        # Ground state of Z is |1>, energy = -1
        assert_vqe_converges(circuit, [0.5], expected_minimum=-1.0, atol=0.05)
    """
    from scipy.optimize import minimize

    params0 = np.asarray(initial_params, dtype=float)
    history: list[float] = []

    def _tracked(p: np.ndarray) -> float:
        val = float(np.real(cost_function(p)))
        history.append(val)
        return val

    result = minimize(
        _tracked,
        params0,
        method=method,
        options={"maxiter": max_iterations, "rhobeg": 0.5},
    )

    if len(history) < 2:
        raise AssertionError(
            "VQE ran fewer than 2 iterations — cost function may not be callable."
        )

    initial_cost = history[0]
    final_cost = result.fun

    if final_cost >= initial_cost - atol:
        raise AssertionError(
            f"VQE did not converge: energy did not decrease.\n"
            f"  Initial cost : {initial_cost:.6f}\n"
            f"  Final cost   : {final_cost:.6f}\n"
            f"  Iterations   : {len(history)}\n"
            f"  Optimizer    : {method}\n"
            f"  Hint: try more iterations (max_iterations=500) or a "
            f"different optimizer."
        )

    if expected_minimum is not None:
        tol = atol + rtol * abs(expected_minimum)
        deviation = abs(final_cost - expected_minimum)
        if deviation > tol:
            raise AssertionError(
                f"VQE converged but missed expected minimum.\n"
                f"  Expected minimum : {expected_minimum:.6f}\n"
                f"  Got              : {final_cost:.6f}\n"
                f"  Deviation        : {deviation:.6f}  (atol={atol})\n"
                f"  Convergence path : {history[0]:.4f} -> ... -> {history[-1]:.4f} "
                f"({len(history)} steps)"
            )


def assert_cost_decreases(
    cost_history: list[float],
    *,
    min_decrease: float = 0.0,
    atol: float = 1e-6,
) -> None:
    """Assert an optimization cost history shows overall decrease.

    Checks that ``cost_history[-1] < cost_history[0] - min_decrease``.

    Args:
        cost_history: List of cost values recorded during optimization.
        min_decrease: Minimum required decrease from first to last value.
        atol: Tolerance for floating-point comparison (default 1e-6).

    Raises:
        AssertionError: If cost did not decrease sufficiently.
        ValueError: If cost_history has fewer than 2 entries.

    Example::

        history = []
        for step in range(50):
            cost = run_vqe_step(params)
            history.append(cost)

        assert_cost_decreases(history, min_decrease=0.1)
    """
    if len(cost_history) < 2:
        raise ValueError(
            f"cost_history must have at least 2 entries; got {len(cost_history)}."
        )
    initial = cost_history[0]
    final = cost_history[-1]
    actual_decrease = initial - final
    if actual_decrease < min_decrease - atol:
        raise AssertionError(
            f"Cost did not decrease sufficiently.\n"
            f"  Initial cost : {initial:.6f}\n"
            f"  Final cost   : {final:.6f}\n"
            f"  Decrease     : {actual_decrease:.6f}  "
            f"(required >= {min_decrease:.6f})\n"
            f"  History length: {len(cost_history)} steps"
        )


def _extract_expectation_value(result: Any) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, np.ndarray):
        if result.ndim == 0:
            return float(result)
        if result.size == 1:
            return float(result.flat[0])
        raise TypeError(f"Expected scalar, got numpy array of shape {result.shape}.")
    module = type(result).__module__
    if module.startswith("qiskit"):
        if hasattr(result, "values"):
            vals = np.asarray(result.values)
            if vals.size == 1:
                return float(vals.flat[0])
        if hasattr(result, "data") and hasattr(result.data, "evs"):
            return float(np.asarray(result.data.evs).flat[0])
    if hasattr(result, "numpy"):
        return float(result.numpy())
    try:
        return float(result)
    except (TypeError, ValueError):
        pass
    raise TypeError(
        f"Cannot extract expectation value from {type(result).__qualname__!r}. "
        "Pass a plain float, numpy scalar, Qiskit EstimatorResult, "
        "or PennyLane result."
    )
