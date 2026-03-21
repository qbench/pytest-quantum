"""Framework-agnostic circuit → numpy unitary converter.

This is the core of cross-framework support.  Every assertion that needs to
compare circuits ultimately calls :func:`to_unitary`.

Supported circuit types
-----------------------
* ``qiskit.QuantumCircuit`` — via ``qiskit.quantum_info.Operator``
* ``cirq.Circuit``           — via ``cirq.unitary()``
* ``braket.circuits.Circuit``— via ``circuit.to_unitary()``
* PennyLane ``QNode``        — via ``qml.matrix()``

Graphix ``Pattern`` objects are **not** handled here because MBQC circuits
do not have a fixed unitary representation (measurement outcomes are random).
Use :func:`pytest_quantum.assertions.states.assert_state_fidelity_above`
with the ``graphix_backend`` fixture instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _reverse_qubit_order(U: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Reverse qubit ordering convention (little-endian ↔ big-endian)."""
    n = round(np.log2(U.shape[0]))
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)]
    return U[np.ix_(perm, perm)]


def _is_qiskit(circuit: object) -> bool:
    return type(circuit).__module__.startswith("qiskit")


def _is_cirq(circuit: object) -> bool:
    return type(circuit).__module__.startswith("cirq")


def _is_braket(circuit: object) -> bool:
    return type(circuit).__module__.startswith("braket")


def _is_pennylane(circuit: object) -> bool:
    mod = type(circuit).__module__
    return mod.startswith("pennylane") or hasattr(circuit, "device")


def _is_pytket(circuit: object) -> bool:
    return type(circuit).__module__.startswith("pytket")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_unitary(circuit: object) -> NDArray[np.complex128]:
    """Convert any supported quantum circuit to a numpy unitary matrix.

    The returned matrix has shape ``(2**n, 2**n)`` and dtype ``complex128``,
    where *n* is the number of qubits.

    Args:
        circuit: A quantum circuit from one of the supported frameworks:
            ``qiskit.QuantumCircuit``, ``cirq.Circuit``,
            ``braket.circuits.Circuit``, or a PennyLane ``QNode``.

    Returns:
        A unitary matrix as a ``numpy.ndarray`` of shape ``(2**n, 2**n)``.

    Raises:
        TypeError: If *circuit* is not a recognised circuit type.
        ImportError: If the required SDK for the circuit type is not installed,
            with a message telling the user which extra to install.

    Example::

        from qiskit import QuantumCircuit
        from pytest_quantum.converters import to_unitary
        import numpy as np

        qc = QuantumCircuit(1)
        qc.h(0)
        U = to_unitary(qc)
        assert U.shape == (2, 2)
    """
    if _is_qiskit(circuit):
        return _from_qiskit(circuit)
    if _is_cirq(circuit):
        return _from_cirq(circuit)
    if _is_braket(circuit):
        return _from_braket(circuit)
    if _is_pennylane(circuit):
        return _from_pennylane(circuit)
    if _is_pytket(circuit):
        return _from_pytket(circuit)

    raise TypeError(
        f"Unrecognised circuit type: {type(circuit).__qualname__!r}.\n"
        "pytest-quantum supports: qiskit.QuantumCircuit, cirq.Circuit, "
        "braket.circuits.Circuit, pennylane QNode, pytket Circuit.\n"
        "For graphix patterns use assert_state_fidelity_above() instead."
    )


# ---------------------------------------------------------------------------
# Per-framework helpers (private)
# ---------------------------------------------------------------------------


def _from_qiskit(circuit: object) -> NDArray[np.complex128]:
    try:
        from qiskit.quantum_info import Operator
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for Qiskit circuit support. "
            "Install it with: pip install pytest-quantum[qiskit]"
        ) from exc

    return np.asarray(Operator(circuit).data, dtype=np.complex128)


def _from_cirq(circuit: object) -> NDArray[np.complex128]:
    try:
        import cirq
    except ImportError as exc:
        raise ImportError(
            "cirq is required for Cirq circuit support. "
            "Install it with: pip install pytest-quantum[cirq]"
        ) from exc

    return np.asarray(cirq.unitary(circuit), dtype=np.complex128)


def _from_braket(circuit: object) -> NDArray[np.complex128]:
    try:
        result = cast("Any", circuit).to_unitary()
    except AttributeError as exc:
        raise TypeError(
            "The Braket circuit does not expose a to_unitary() method. "
            "Ensure amazon-braket-sdk is installed: "
            "pip install pytest-quantum[braket]"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "amazon-braket-sdk is required for Braket circuit support. "
            "Install it with: pip install pytest-quantum[braket]"
        ) from exc

    return np.asarray(result, dtype=np.complex128)


def _from_pytket(circuit: object) -> NDArray[np.complex128]:
    try:
        import numpy as np  # already imported at top level, but keep for clarity

        U = np.asarray(cast("Any", circuit).get_unitary())
        # pytket uses ILO-BE (big-endian) like Cirq — no reversal needed vs Cirq,
        # but qubit order reversal is handled in assert_circuits_equivalent when
        # comparing against Qiskit (little-endian).
        return U.astype(np.complex128)
    except AttributeError as exc:
        raise ImportError(
            "pytket is required for Pytket circuit support. "
            "Install it with: pip install pytket"
        ) from exc
    except ImportError as exc:
        raise ImportError("pytket is required: pip install pytket") from exc


def _from_pennylane(circuit: object) -> NDArray[np.complex128]:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "pennylane is required for PennyLane circuit support. "
            "Install it with: pip install pytest-quantum[pennylane]"
        ) from exc

    # Determine wire order from the QNode's device
    if not hasattr(circuit, "device"):
        raise TypeError(
            "Cannot auto-detect wire_order for this PennyLane object. "
            "Pass a QNode (decorated with @qml.qnode) or use "
            "qml.matrix(your_fn, wire_order=[0, 1, ...])(params) directly."
        )
    wire_order = list(cast("Any", circuit).device.wires)
    matrix_fn = qml.matrix(circuit, wire_order=wire_order)
    return np.asarray(matrix_fn(), dtype=np.complex128)
