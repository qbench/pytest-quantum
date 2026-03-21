"""Cross-platform circuit equivalence assertions.

Assert that circuits expressed in different frameworks (Qiskit, Cirq, pytket)
implement the same unitary up to global phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytest_quantum.converters.to_unitary import (
    _is_cirq,
    _is_pytket,
    _is_qiskit,
    _reverse_qubit_order,
    to_unitary,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _framework_label(circuit: object, hint: str | None) -> str:
    """Return a human-readable framework name for error messages."""
    if hint is not None:
        return hint
    mod = type(circuit).__module__
    if mod.startswith("qiskit"):
        return "qiskit"
    if mod.startswith("cirq"):
        return "cirq"
    if mod.startswith("pytket"):
        return "pytket"
    if mod.startswith("braket"):
        return "braket"
    if mod.startswith("pennylane"):
        return "pennylane"
    return mod.split(".")[0]


def _unitaries_equivalent(
    u_a: NDArray[np.complex128],
    u_b: NDArray[np.complex128],
    *,
    atol: float,
    allow_global_phase: bool,
) -> bool:
    """Return True if u_a and u_b are close (optionally up to global phase)."""
    if np.allclose(u_a, u_b, atol=atol):
        return True
    if not allow_global_phase:
        return False
    flat_idx = int(np.argmax(np.abs(u_a)))
    a_val = u_a.flat[flat_idx]
    b_val = u_b.flat[flat_idx]
    if abs(a_val) > 1e-10 and abs(b_val) > 1e-10:
        phase = a_val / b_val
        if np.allclose(u_a, phase * u_b, atol=atol):
            return True
    return False


def _normalise_endianness(
    u_a: NDArray[np.complex128],
    circuit_a: object,
    u_b: NDArray[np.complex128],
    circuit_b: object,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Align qubit ordering conventions between frameworks.

    Qiskit uses little-endian; Cirq and pytket use big-endian.
    When comparing across these conventions, reverse one of the unitaries.
    """
    big_endian_a = _is_cirq(circuit_a) or _is_pytket(circuit_a)
    big_endian_b = _is_cirq(circuit_b) or _is_pytket(circuit_b)

    if _is_qiskit(circuit_a) and big_endian_b:
        u_a = _reverse_qubit_order(u_a)
    elif big_endian_a and _is_qiskit(circuit_b):
        u_b = _reverse_qubit_order(u_b)
    return u_a, u_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assert_cross_platform_equivalent(
    circuit_a: object,
    circuit_b: object,
    *,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
    framework_a: str | None = None,
    framework_b: str | None = None,
) -> None:
    """Assert that two circuits implement the same unitary, up to global phase.

    Converts both circuits to numpy unitary matrices using
    :func:`pytest_quantum.converters.to_unitary.to_unitary` and compares them
    numerically, optionally ignoring a global phase factor ``e^{iθ}``.

    Qubit-ordering conventions are normalised automatically: Qiskit uses
    little-endian while Cirq and pytket use big-endian, so comparing a Qiskit
    circuit against a Cirq or pytket circuit applies the appropriate permutation
    before the element-wise check.

    Args:
        circuit_a:          First circuit (any supported framework).
        circuit_b:          Second circuit (any supported framework).
        atol:               Absolute tolerance for element-wise comparison
                            (default ``1e-6``).
        allow_global_phase: If ``True`` (default), circuits that differ only
                            by a global phase ``e^{iθ}`` are considered
                            equivalent.
        framework_a:        Optional hint like ``"qiskit"`` or ``"cirq"`` used
                            only to improve error messages.
        framework_b:        Optional hint for the second circuit.

    Raises:
        AssertionError: If the circuits implement different unitaries, with a
                        message showing the max element-wise difference and the
                        framework names.
        ValueError:     If either circuit cannot be converted to a unitary
                        (e.g. it contains measurements or is not a pure-unitary
                        circuit).

    Example::

        from qiskit import QuantumCircuit
        from pytest_quantum.assertions.cross_platform import (
            assert_cross_platform_equivalent,
        )

        qc1 = QuantumCircuit(1)
        qc1.h(0)

        qc2 = QuantumCircuit(1)
        qc2.h(0)

        assert_cross_platform_equivalent(qc1, qc2)
    """
    label_a = _framework_label(circuit_a, framework_a)
    label_b = _framework_label(circuit_b, framework_b)

    try:
        u_a: NDArray[np.complex128] = to_unitary(circuit_a)
    except (TypeError, ImportError) as exc:
        raise ValueError(
            f"Could not convert circuit_a ({label_a}) to a unitary matrix.\n"
            f"  Reason: {exc}"
        ) from exc

    try:
        u_b: NDArray[np.complex128] = to_unitary(circuit_b)
    except (TypeError, ImportError) as exc:
        raise ValueError(
            f"Could not convert circuit_b ({label_b}) to a unitary matrix.\n"
            f"  Reason: {exc}"
        ) from exc

    u_a, u_b = _normalise_endianness(u_a, circuit_a, u_b, circuit_b)

    if u_a.shape != u_b.shape:
        raise AssertionError(
            f"Circuits act on different-sized Hilbert spaces.\n"
            f"  {label_a}: unitary shape {u_a.shape}\n"
            f"  {label_b}: unitary shape {u_b.shape}\n"
            f"  Hint: check that both circuits have the same number of qubits."
        )

    if _unitaries_equivalent(u_a, u_b, atol=atol, allow_global_phase=allow_global_phase):
        return

    max_diff = float(np.max(np.abs(u_a - u_b)))
    raise AssertionError(
        f"Circuits are NOT equivalent across platforms.\n"
        f"  Framework A   : {label_a}\n"
        f"  Framework B   : {label_b}\n"
        f"  Max |U_a - U_b|: {max_diff:.2e}   (tolerance: {atol:.2e})\n"
        f"  allow_global_phase = {allow_global_phase}\n"
        f"  Hint: if the circuits differ only in global phase, set "
        f"allow_global_phase=True."
    )


def assert_qiskit_cirq_equivalent(
    qiskit_circuit: object,
    cirq_circuit: object,
    *,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert that a Qiskit circuit and a Cirq circuit implement the same unitary.

    Convenience wrapper around :func:`assert_cross_platform_equivalent` with
    the framework labels pre-set to ``"qiskit"`` and ``"cirq"``.

    Qubit-ordering is normalised automatically (Qiskit is little-endian, Cirq
    is big-endian), so you can compare circuits using their natural qubit
    indices in each framework without manual reversal.

    Args:
        qiskit_circuit:     A ``qiskit.QuantumCircuit`` instance.
        cirq_circuit:       A ``cirq.Circuit`` instance.
        atol:               Absolute tolerance for element-wise comparison
                            (default ``1e-6``).
        allow_global_phase: If ``True`` (default), circuits that differ only
                            by a global phase are considered equivalent.

    Raises:
        AssertionError: If the circuits implement different unitaries.
        ValueError:     If either circuit cannot be converted to a unitary.

    Example::

        import cirq
        from qiskit import QuantumCircuit
        from pytest_quantum.assertions.cross_platform import (
            assert_qiskit_cirq_equivalent,
        )

        qc = QuantumCircuit(1)
        qc.h(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.H(q[0]))

        assert_qiskit_cirq_equivalent(qc, cc)
    """
    assert_cross_platform_equivalent(
        qiskit_circuit,
        cirq_circuit,
        atol=atol,
        allow_global_phase=allow_global_phase,
        framework_a="qiskit",
        framework_b="cirq",
    )


def assert_qiskit_pytket_equivalent(
    qiskit_circuit: object,
    pytket_circuit: object,
    *,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert that a Qiskit circuit and a pytket circuit implement the same unitary.

    Convenience wrapper around :func:`assert_cross_platform_equivalent` with
    the framework labels pre-set to ``"qiskit"`` and ``"pytket"``.

    The pytket unitary is obtained via ``circuit.get_unitary()`` (exposed by
    :func:`pytest_quantum.converters.to_unitary.to_unitary`).  Qubit-ordering
    is normalised automatically (Qiskit is little-endian, pytket is big-endian).

    Args:
        qiskit_circuit:     A ``qiskit.QuantumCircuit`` instance.
        pytket_circuit:     A ``pytket.Circuit`` instance.
        atol:               Absolute tolerance for element-wise comparison
                            (default ``1e-6``).
        allow_global_phase: If ``True`` (default), circuits that differ only
                            by a global phase are considered equivalent.

    Raises:
        AssertionError: If the circuits implement different unitaries.
        ValueError:     If either circuit cannot be converted to a unitary.

    Example::

        from pytket import Circuit as TketCircuit
        from qiskit import QuantumCircuit
        from pytest_quantum.assertions.cross_platform import (
            assert_qiskit_pytket_equivalent,
        )

        qc = QuantumCircuit(1)
        qc.h(0)

        tk = TketCircuit(1)
        tk.H(0)

        assert_qiskit_pytket_equivalent(qc, tk)
    """
    assert_cross_platform_equivalent(
        qiskit_circuit,
        pytket_circuit,
        atol=atol,
        allow_global_phase=allow_global_phase,
        framework_a="qiskit",
        framework_b="pytket",
    )
