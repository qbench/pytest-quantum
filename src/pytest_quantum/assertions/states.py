"""Statevector-level assertions for quantum tests.

Use these when you have the full output statevector from a simulator (Aer
statevector mode, Cirq simulator, or the Graphix backend) and want to verify
the quantum state directly — more informative than comparing shot distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytest_quantum.stats.tests import fidelity as _fidelity

if TYPE_CHECKING:
    from numpy.typing import NDArray


def assert_state_fidelity_above(
    actual: NDArray[np.complex128],
    target: NDArray[np.complex128],
    threshold: float = 0.99,
) -> None:
    """Assert that two pure quantum states have fidelity at or above *threshold*.

    Fidelity :math:`F = |\\langle\\text{actual}|\\text{target}\\rangle|^2`
    equals 1.0 for identical states (up to global phase) and 0.0 for
    orthogonal states.

    This is the primary assertion for MBQC / Graphix tests where the circuit
    does not have a fixed unitary representation.

    Args:
        actual:    Simulated output statevector, any shape (will be flattened).
        target:    Ideal target statevector, same number of elements.
        threshold: Minimum acceptable fidelity (default ``0.99``).

    Raises:
        AssertionError: If ``fidelity(actual, target) < threshold``.
        ValueError:     If the arrays have incompatible sizes.

    Example::

        import numpy as np
        from pytest_quantum import assert_state_fidelity_above

        BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


        def test_bell_graphix(graphix_backend):
            from graphix.transpiler import Circuit

            circuit = Circuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)
            pattern = circuit.transpile().pattern
            output = graphix_backend.run_pattern(pattern)
            assert_state_fidelity_above(output, BELL, threshold=0.999)
    """
    f = _fidelity(actual, target)
    if f < threshold:
        raise AssertionError(
            f"State fidelity too low.\n"
            f"  |⟨actual|target⟩|² = {f:.6f}\n"
            f"  Required            ≥ {threshold}\n"
            f"  Shortfall           = {threshold - f:.2e}"
        )


def assert_normalized(statevector: object, *, atol: float = 1e-6) -> None:
    """Assert statevector has unit norm: ||ψ||₂ = 1.

    A common bug in manual statevector construction is forgetting to normalize.

    Args:
        statevector: Complex array-like of any shape (flattened internally).
        atol: Absolute tolerance from 1.0 (default 1e-6).

    Raises:
        AssertionError: If ||sv||₂ is not within atol of 1.0, showing the actual norm.

    Example::

        >>> import numpy as np
        >>> sv = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        >>> assert_normalized(sv)  # passes

        >>> sv_bad = np.array([1, 1], dtype=complex)  # NOT normalized
        >>> assert_normalized(sv_bad)  # fails: norm = 1.4142
    """
    sv = np.asarray(statevector, dtype=np.complex128).flatten()
    norm = float(np.linalg.norm(sv))
    if abs(norm - 1.0) > atol:
        raise AssertionError(
            f"Statevector is not normalized.\n"
            f"  Norm: {norm:.6f}  (expected 1.0, tolerance: {atol:.2e})\n"
            f"  Deviation: {abs(norm - 1.0):.6f}\n"
            f"  Hint: divide by np.linalg.norm(sv) to normalize."
        )


def assert_states_close(
    actual: NDArray[np.complex128],
    target: NDArray[np.complex128],
    *,
    atol: float = 1e-6,
) -> None:
    """Assert that two statevectors are element-wise close, up to global phase.

    Stricter than :func:`assert_state_fidelity_above` — use for exact
    simulator-to-simulator comparisons where you want bit-for-bit agreement.

    Args:
        actual: Simulated statevector (will be flattened and normalised).
        target: Ideal statevector (will be flattened and normalised).
        atol:   Absolute tolerance per element (default ``1e-6``).

    Raises:
        AssertionError: If any element differs by more than *atol* after
            removing the global phase.

    Example::

        def test_plus_state(aer_statevector_simulator):
            from qiskit import QuantumCircuit, transpile

            qc = QuantumCircuit(1)
            qc.h(0)
            qc.save_statevector()
            qc_t = transpile(qc, aer_statevector_simulator)
            sv = aer_statevector_simulator.run(qc_t).result().get_statevector()
            PLUS = np.array([1, 1]) / np.sqrt(2)
            assert_states_close(sv.data, PLUS)
    """
    a = np.asarray(actual, dtype=np.complex128).flatten()
    t = np.asarray(target, dtype=np.complex128).flatten()

    if a.size != t.size:
        raise AssertionError(
            f"Statevector size mismatch: actual has {a.size} elements, "
            f"target has {t.size}."
        )

    # Normalise
    a = a / np.linalg.norm(a)
    t = t / np.linalg.norm(t)

    # Remove global phase: align the largest-magnitude element of `a` to `t`
    idx = int(np.argmax(np.abs(t)))
    if abs(t[idx]) > 1e-10 and abs(a[idx]) > 1e-10:
        phase = a[idx] / t[idx]
        a = a / phase

    if not np.allclose(a, t, atol=atol):
        max_diff = float(np.max(np.abs(a - t)))
        raise AssertionError(
            f"Statevectors are not close (after global-phase alignment).\n"
            f"  Max |difference|: {max_diff:.2e}   (tolerance: {atol:.2e})"
        )
