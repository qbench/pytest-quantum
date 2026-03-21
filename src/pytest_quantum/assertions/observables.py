"""Expectation value and observable assertions for VQE/QAOA testing."""

from __future__ import annotations

from typing import Any

import numpy as np


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
        return float(result)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    raise TypeError(
        f"Cannot extract expectation value from {type(result).__qualname__!r}. "
        "Pass a plain float, numpy scalar, Qiskit EstimatorResult, "
        "or PennyLane result."
    )
