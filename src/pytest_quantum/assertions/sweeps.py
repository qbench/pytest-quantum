"""Parametrized circuit sweep assertions."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def assert_circuit_sweep(
    circuit_fn: Callable[..., Any],
    param_values: dict[str, list[float] | NDArray[np.float64]],
    expected_fn: Callable[..., NDArray[np.complex128]],
    *,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert a parametrized circuit matches expected unitary for all parameter values.

    Calls circuit_fn(**params) for every combination of param_values, computes
    the unitary, and compares to expected_fn(**params).

    Args:
        circuit_fn:         Callable that accepts keyword parameter arguments and
                            returns a quantum circuit.
        param_values:       Dict mapping parameter name to list/array of values.
                            All combinations are tested (cartesian product).
        expected_fn:        Callable that accepts the same keyword arguments and
                            returns the expected unitary as a numpy array.
        atol:               Absolute tolerance for unitary comparison (default 1e-6).
        allow_global_phase: If True, ignore global phase differences (default True).

    Raises:
        AssertionError: If any parameter combination fails.

    Example::

        import numpy as np
        from pytest_quantum import assert_circuit_sweep
        from qiskit import QuantumCircuit


        def rx_circuit(theta):
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            return qc


        def rx_expected(theta):
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -1j * s], [-1j * s, c]])


        assert_circuit_sweep(
            rx_circuit,
            {"theta": np.linspace(0, 2 * np.pi, 8)},
            rx_expected,
        )
    """
    from pytest_quantum.converters.to_unitary import to_unitary

    keys = list(param_values.keys())
    value_lists = [list(param_values[k]) for k in keys]
    failures: list[str] = []

    for combo in itertools.product(*value_lists):
        kwargs = dict(zip(keys, combo, strict=False))
        circuit = circuit_fn(**kwargs)
        actual = to_unitary(circuit)
        expected = np.asarray(expected_fn(**kwargs), dtype=np.complex128)

        if actual.shape != expected.shape:
            failures.append(
                f"  params={kwargs}: shape mismatch {actual.shape} vs {expected.shape}"
            )
            continue

        from pytest_quantum._internal import _unitaries_equivalent

        if _unitaries_equivalent(
            actual, expected, atol=atol, allow_global_phase=allow_global_phase
        ):
            continue

        max_diff = float(np.max(np.abs(actual - expected)))
        failures.append(
            f"  params={_fmt_params(kwargs)}: max|diff|={max_diff:.2e} (tol={atol:.2e})"
        )

    if failures:
        raise AssertionError(
            f"assert_circuit_sweep failed for {len(failures)} / "
            f"{_count_combos(param_values)} parameter combinations:\n"
            + "\n".join(failures)
        )


def assert_circuit_sweep_states(
    circuit_fn: Callable[..., Any],
    initial_state: NDArray[np.complex128],
    param_values: dict[str, list[float] | NDArray[np.float64]],
    expected_fn: Callable[..., NDArray[np.complex128]],
    *,
    min_fidelity: float = 0.99,
) -> None:
    """Assert a parametrized circuit produces expected output states for all params.

    Applies circuit_fn(**params) to initial_state and compares the output
    statevector fidelity to expected_fn(**params).

    Args:
        circuit_fn:    Callable(**params) -> circuit.
        initial_state: Input statevector, shape (2**n,).
        param_values:  Dict of parameter name -> list of values.
        expected_fn:   Callable(**params) -> expected output statevector.
        min_fidelity:  Minimum fidelity |<actual|expected>|^2 (default 0.99).

    Raises:
        AssertionError: If any combination fails the fidelity check.

    Example::

        import numpy as np
        from pytest_quantum import assert_circuit_sweep_states

        psi0 = np.array([1, 0], dtype=complex)


        def rz_circuit(phi):
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(1)
            qc.rz(phi, 0)
            return qc


        def expected_state(phi):
            return np.array([np.exp(-1j * phi / 2), 0])


        assert_circuit_sweep_states(
            rz_circuit, psi0, {"phi": [0, np.pi / 2, np.pi]}, expected_state
        )
    """
    from pytest_quantum.converters.to_unitary import to_unitary

    keys = list(param_values.keys())
    value_lists = [list(param_values[k]) for k in keys]
    failures: list[str] = []

    for combo in itertools.product(*value_lists):
        kwargs = dict(zip(keys, combo, strict=False))
        circuit = circuit_fn(**kwargs)
        U = to_unitary(circuit)
        psi_in = np.asarray(initial_state, dtype=np.complex128)
        actual = U @ psi_in
        actual /= np.linalg.norm(actual)

        expected = np.asarray(expected_fn(**kwargs), dtype=np.complex128)
        expected /= np.linalg.norm(expected)

        fidelity = float(abs(np.vdot(actual, expected)) ** 2)
        if fidelity < min_fidelity:
            failures.append(
                f"  params={_fmt_params(kwargs)}: fidelity={fidelity:.4f} < {min_fidelity}"
            )

    if failures:
        raise AssertionError(
            f"assert_circuit_sweep_states failed for {len(failures)} / "
            f"{_count_combos(param_values)} parameter combinations:\n"
            + "\n".join(failures)
        )


def assert_parametrized_unitary_continuous(
    circuit_fn: Callable[..., Any],
    param_name: str,
    param_range: tuple[float, float],
    *,
    n_samples: int = 20,
    max_jump: float = 0.5,
) -> None:
    """Assert that a parametrized circuit's unitary varies continuously.

    Detects discontinuities (e.g. branch cut issues, phase jumps) by checking
    that consecutive unitary matrices are close in Frobenius norm.

    Args:
        circuit_fn:   Callable(param_name=value) -> circuit.
        param_name:   Name of the parameter to sweep.
        param_range:  (start, end) range for the parameter.
        n_samples:    Number of evenly-spaced sample points (default 20).
        max_jump:     Maximum allowed Frobenius norm change between adjacent
                      samples (default 0.5).

    Raises:
        AssertionError: If any discontinuity exceeds max_jump.

    Example::

        assert_parametrized_unitary_continuous(
            lambda theta: rx_circuit(theta), "theta", (0, 2 * np.pi)
        )
    """
    from pytest_quantum.converters.to_unitary import to_unitary

    values = np.linspace(param_range[0], param_range[1], n_samples)
    unitaries = []
    for v in values:
        circuit = circuit_fn(**{param_name: float(v)})
        unitaries.append(to_unitary(circuit))

    jumps: list[str] = []
    for i in range(1, len(unitaries)):
        diff = float(np.linalg.norm(unitaries[i] - unitaries[i - 1], "fro"))
        if diff > max_jump:
            jumps.append(
                f"  {param_name}={values[i - 1]:.4f}->{values[i]:.4f}: ||DeltaU||_F={diff:.4f}"
            )

    if jumps:
        raise AssertionError(
            f"Unitary not continuous at {len(jumps)} transition(s) "
            f"(max_jump={max_jump}):\n" + "\n".join(jumps)
        )


def _fmt_params(kwargs: dict[str, float]) -> str:
    """Format parameter dict for error messages."""
    return "{" + ", ".join(f"{k}={v:.4g}" for k, v in kwargs.items()) + "}"


def _count_combos(
    param_values: dict[str, list[float] | NDArray[np.float64]],
) -> int:
    """Count total number of parameter combinations."""
    result = 1
    for v in param_values.values():
        result *= len(v)
    return result
