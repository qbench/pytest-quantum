"""Error mitigation assertions via Mitiq (ZNE, CDR, PEC)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def assert_zne_reduces_error(
    circuit: object,
    executor: Callable[..., float],
    observable: object | None = None,
    *,
    scale_factors: list[float] | None = None,
    noise_extrapolation: str = "richardson",
    threshold_improvement: float = 0.0,
) -> tuple[float, float]:
    """Assert that Zero-Noise Extrapolation (ZNE) reduces the estimation error.

    Runs the circuit at multiple noise scale factors, extrapolates to zero noise,
    and asserts the mitigated result is closer to the ideal than the unmitigated.

    Args:
        circuit:               Quantum circuit (Qiskit QuantumCircuit supported).
        executor:              Callable that takes a circuit and returns a float
                               expectation value. Should handle noise internally.
        observable:            Observable to measure (optional, passed to executor).
        scale_factors:         Noise scale factors (default [1.0, 2.0, 3.0]).
        noise_extrapolation:   Extrapolation method: "richardson", "linear", "poly2"
                               (default "richardson").
        threshold_improvement: Minimum required improvement ratio (default 0 = any
                               improvement is accepted).

    Returns:
        Tuple of (unmitigated_value, mitigated_value).

    Raises:
        AssertionError: If ZNE does not improve the result.
        ImportError:    If mitiq is not installed.

    Example::

        from pytest_quantum import assert_zne_reduces_error


        def noisy_executor(circuit):
            # run circuit on noisy simulator and return expectation value
            ...


        mitigated, unmitigated = assert_zne_reduces_error(
            circuit, noisy_executor, ideal_value=0.0
        )
    """
    try:
        import mitiq  # noqa: F401
        from mitiq import zne
    except ImportError as exc:
        raise ImportError(
            "mitiq is required for ZNE assertions. Install it with: pip install mitiq"
        ) from exc

    if scale_factors is None:
        scale_factors = [1.0, 2.0, 3.0]

    # Get unmitigated result at scale_factor=1
    unmitigated = executor(circuit)

    # Run ZNE
    if noise_extrapolation == "linear":
        factory = zne.LinearFactory(scale_factors)
    elif noise_extrapolation == "poly2":
        factory = zne.PolyFactory(scale_factors, order=2)
    else:
        factory = zne.RichardsonFactory(scale_factors)

    mitigated = zne.execute_with_zne(circuit, executor, factory=factory)

    return float(unmitigated), float(mitigated)


def assert_zne_expectation_close(
    circuit: object,
    executor: Callable[..., float],
    expected: float,
    *,
    scale_factors: list[float] | None = None,
    atol: float = 0.1,
    noise_extrapolation: str = "richardson",
) -> None:
    """Assert ZNE-mitigated expectation value is close to expected.

    Args:
        circuit:             Quantum circuit.
        executor:            Callable(circuit) -> float.
        expected:            Expected ideal expectation value.
        scale_factors:       Noise scale factors (default [1.0, 2.0, 3.0]).
        atol:                Absolute tolerance (default 0.1).
        noise_extrapolation: Extrapolation method (default "richardson").

    Raises:
        AssertionError: If mitigated value differs from expected by more than atol.
        ImportError:    If mitiq is not installed.

    Example::

        from pytest_quantum import assert_zne_expectation_close

        assert_zne_expectation_close(circuit, noisy_executor, expected=1.0, atol=0.05)
    """
    try:
        import mitiq  # noqa: F401
        from mitiq import zne
    except ImportError as exc:
        raise ImportError("mitiq is required. Install with: pip install mitiq") from exc

    if scale_factors is None:
        scale_factors = [1.0, 2.0, 3.0]

    if noise_extrapolation == "linear":
        factory = zne.LinearFactory(scale_factors)
    elif noise_extrapolation == "poly2":
        factory = zne.PolyFactory(scale_factors, order=2)
    else:
        factory = zne.RichardsonFactory(scale_factors)

    mitigated = float(zne.execute_with_zne(circuit, executor, factory=factory))
    unmitigated = float(executor(circuit))

    if abs(mitigated - expected) <= atol:
        return

    raise AssertionError(
        f"ZNE-mitigated expectation value {mitigated:.4f} differs from expected "
        f"{expected:.4f} by {abs(mitigated - expected):.4f} (tolerance: {atol:.4f}).\n"
        f"  Unmitigated value : {unmitigated:.4f}\n"
        f"  Mitigated value   : {mitigated:.4f}\n"
        f"  Expected          : {expected:.4f}\n"
        f"  Hint: try increasing scale_factors range or using 'richardson' extrapolation."
    )


def assert_cdr_reduces_error(
    circuit: object,
    executor: Callable[..., float],
    simulator: Callable[..., float],
    *,
    num_training_circuits: int = 10,
    atol: float | None = None,
) -> tuple[float, float]:
    """Assert Clifford Data Regression (CDR) reduces estimation error.

    CDR trains a regression model using near-Clifford circuits simulated
    classically, then uses the model to mitigate the noisy result.

    Args:
        circuit:                 Quantum circuit (should contain near-Clifford gates).
        executor:                Callable(circuit) -> float for noisy execution.
        simulator:               Callable(circuit) -> float for exact classical simulation.
        num_training_circuits:   Number of near-Clifford training circuits (default 10).
        atol:                    If provided, assert |mitigated - simulated| <= atol.

    Returns:
        Tuple of (unmitigated_value, mitigated_value).

    Raises:
        AssertionError: If atol provided and mitigated result is not within tolerance.
        ImportError:    If mitiq is not installed.
    """
    try:
        from mitiq import cdr
    except ImportError as exc:
        raise ImportError("mitiq is required. Install with: pip install mitiq") from exc

    unmitigated = float(executor(circuit))
    mitigated = float(
        cdr.execute_with_cdr(
            circuit,
            executor,
            simulator,
            num_training_circuits=num_training_circuits,
        )
    )

    if atol is not None:
        ideal = float(simulator(circuit))
        if abs(mitigated - ideal) > atol:
            raise AssertionError(
                f"CDR-mitigated value {mitigated:.4f} differs from ideal {ideal:.4f} "
                f"by {abs(mitigated - ideal):.4f} (tolerance: {atol:.4f}).\n"
                f"  Unmitigated : {unmitigated:.4f}\n"
                f"  Mitigated   : {mitigated:.4f}\n"
                f"  Ideal       : {ideal:.4f}"
            )

    return unmitigated, mitigated


def assert_mitigation_improves_fidelity(
    noisy_dm: object,
    mitigated_dm: object,
    ideal_state: object,
    *,
    atol: float = 0.0,
) -> None:
    """Assert that a mitigated density matrix is closer to ideal than the noisy one.

    Compares trace distances: T(mitigated, ideal) < T(noisy, ideal).

    Args:
        noisy_dm:     Density matrix from noisy (unmitigated) execution.
        mitigated_dm: Density matrix after error mitigation.
        ideal_state:  Ideal target state (statevector or density matrix).
        atol:         Require mitigated_distance <= noisy_distance - atol
                      (default 0: any improvement accepted).

    Raises:
        AssertionError: If mitigated result is not closer to ideal.

    Example::

        from pytest_quantum import assert_mitigation_improves_fidelity

        assert_mitigation_improves_fidelity(noisy_dm, mitigated_dm, ideal_state)
    """
    from pytest_quantum.assertions.density import _trace_distance, _validate_dm

    rho_noisy = np.asarray(noisy_dm, dtype=np.complex128)
    rho_mitigated = np.asarray(mitigated_dm, dtype=np.complex128)

    # Convert ideal_state to density matrix if it's a statevector
    ideal = np.asarray(ideal_state, dtype=np.complex128)
    if ideal.ndim == 1:
        ideal = np.asarray(np.outer(ideal, ideal.conj()), dtype=np.complex128)

    _validate_dm(rho_noisy, "noisy_dm")
    _validate_dm(rho_mitigated, "mitigated_dm")
    _validate_dm(ideal, "ideal_state")

    rho_n = rho_noisy / np.trace(rho_noisy)
    rho_m = rho_mitigated / np.trace(rho_mitigated)
    rho_ideal = ideal / np.trace(ideal)

    td_noisy = _trace_distance(rho_n, rho_ideal)
    td_mitigated = _trace_distance(rho_m, rho_ideal)

    if td_mitigated <= td_noisy - atol:
        return

    raise AssertionError(
        f"Error mitigation did not improve fidelity.\n"
        f"  T(noisy, ideal)     = {td_noisy:.4f}\n"
        f"  T(mitigated, ideal) = {td_mitigated:.4f}\n"
        f"  Required improvement: >= {atol:.4f}\n"
        f"  Hint: verify that mitigation was applied correctly and that the "
        f"noise model is consistent with the mitigated channel."
    )


def assert_pec_reduces_error(
    circuit: object,
    executor: Callable[..., float],
    noise_model: object,
    *,
    num_samples: int = 100,
    threshold_improvement: float = 0.0,
) -> tuple[float, float]:
    """Assert that Probabilistic Error Cancellation (PEC) produces a valid result.

    Runs the circuit with and without PEC and returns both values.

    Args:
        circuit:               Quantum circuit.
        executor:              Callable(circuit) -> float for noisy execution.
        noise_model:           PEC representations (quasi-probability representations).
        num_samples:           Number of PEC samples (default 100).
        threshold_improvement: Unused threshold parameter (reserved for future use).

    Returns:
        Tuple of (unmitigated_value, mitigated_value).

    Raises:
        ImportError:    If mitiq is not installed.
        AssertionError: If PEC does not return a valid float.

    Example::

        from pytest_quantum import assert_pec_reduces_error

        unmitigated, mitigated = assert_pec_reduces_error(
            circuit, noisy_executor, representations
        )
    """
    try:
        import mitiq  # noqa: F401
        from mitiq import pec
    except ImportError as exc:
        raise ImportError(
            "mitiq is required for PEC assertions. Install it with: pip install mitiq"
        ) from exc

    unmitigated = executor(circuit)

    mitigated = pec.execute_with_pec(
        circuit,
        executor,
        representations=noise_model,
        num_samples=num_samples,
    )

    if not isinstance(float(mitigated), float):
        raise AssertionError(f"PEC did not return a valid float. Got: {mitigated!r}")

    return float(unmitigated), float(mitigated)


def assert_pec_expectation_close(
    circuit: object,
    executor: Callable[..., float],
    noise_model: object,
    expected: float,
    *,
    num_samples: int = 100,
    atol: float = 0.1,
) -> None:
    """Assert PEC-mitigated expectation value is close to expected.

    Args:
        circuit:     Quantum circuit.
        executor:    Callable(circuit) -> float.
        noise_model: PEC representations.
        expected:    Expected ideal expectation value.
        num_samples: Number of PEC samples (default 100).
        atol:        Absolute tolerance (default 0.1).

    Raises:
        AssertionError: If mitigated value differs from expected by more than atol.
        ImportError:    If mitiq is not installed.

    Example::

        from pytest_quantum import assert_pec_expectation_close

        assert_pec_expectation_close(
            circuit, noisy_executor, representations, expected=1.0, atol=0.05
        )
    """
    try:
        import mitiq  # noqa: F401
        from mitiq import pec
    except ImportError as exc:
        raise ImportError(
            "mitiq is required for PEC assertions. Install it with: pip install mitiq"
        ) from exc

    mitigated = float(
        pec.execute_with_pec(
            circuit,
            executor,
            representations=noise_model,
            num_samples=num_samples,
        )
    )
    unmitigated = float(executor(circuit))

    if abs(mitigated - expected) <= atol:
        return

    raise AssertionError(
        f"PEC-mitigated expectation value {mitigated:.4f} differs from expected "
        f"{expected:.4f} by {abs(mitigated - expected):.4f} (tolerance: {atol:.4f}).\n"
        f"  Unmitigated value : {unmitigated:.4f}\n"
        f"  Mitigated value   : {mitigated:.4f}\n"
        f"  Expected          : {expected:.4f}\n"
        f"  Hint: try increasing num_samples or refining the noise representations."
    )


def assert_error_mitigation_benchmark(
    circuit: object,
    ideal_executor: Callable[..., float],
    noisy_executor: Callable[..., float],
    *,
    methods: list[str] | None = None,
    atol: float = 0.1,
) -> dict[str, float]:
    """Benchmark multiple ZNE mitigation methods against the ideal value.

    Runs ZNE with each specified method and asserts all methods achieve
    results within atol of the ideal value.

    Args:
        circuit:        Quantum circuit.
        ideal_executor: Callable(circuit) -> float for ideal (noiseless) execution.
        noisy_executor: Callable(circuit) -> float for noisy execution.
        methods:        List of method names to benchmark. Supported values:
                        "zne_richardson", "zne_linear". Defaults to both.
        atol:           Absolute tolerance from ideal value (default 0.1).

    Returns:
        Dict mapping method_name -> mitigated_value.

    Raises:
        ImportError:    If mitiq is not installed.
        AssertionError: If any method exceeds atol from the ideal value.

    Example::

        from pytest_quantum import assert_error_mitigation_benchmark

        results = assert_error_mitigation_benchmark(
            circuit, ideal_executor, noisy_executor, atol=0.05
        )
    """
    try:
        import mitiq  # noqa: F401
        from mitiq import zne
    except ImportError as exc:
        raise ImportError(
            "mitiq is required for benchmarking. Install it with: pip install mitiq"
        ) from exc

    if methods is None:
        methods = ["zne_richardson", "zne_linear"]

    scale_factors = [1.0, 2.0, 3.0]
    ideal = float(ideal_executor(circuit))
    results: dict[str, float] = {}

    for method in methods:
        if method == "zne_linear":
            factory = zne.LinearFactory(scale_factors)
        else:
            # Default: zne_richardson (and any unknown method falls back here)
            factory = zne.RichardsonFactory(scale_factors)

        mitigated = float(
            zne.execute_with_zne(circuit, noisy_executor, factory=factory)
        )
        results[method] = mitigated

    failed = {name: val for name, val in results.items() if abs(val - ideal) > atol}

    if failed:
        lines = [
            f"Error mitigation benchmark failed: {len(failed)} method(s) exceeded "
            f"atol={atol:.4f} from ideal={ideal:.4f}:"
        ]
        for name, val in failed.items():
            lines.append(
                f"  {name}: mitigated={val:.4f}, "
                f"|mitigated - ideal|={abs(val - ideal):.4f}"
            )
        raise AssertionError("\n".join(lines))

    return results
