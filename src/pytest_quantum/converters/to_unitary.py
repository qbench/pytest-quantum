"""Framework-agnostic circuit → numpy unitary converter.

This is the core of cross-framework support.  Every assertion that needs to
compare circuits ultimately calls :func:`to_unitary`.

Internally delegates to the adapter registry
(:mod:`pytest_quantum.adapters`) for framework detection and conversion.

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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _reverse_qubit_order(U: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Reverse qubit ordering convention (little-endian ↔ big-endian)."""
    d = U.shape[0]
    n = round(np.log2(d))
    if 2**n != d:
        raise ValueError(f"Matrix dimension {d} is not a power of 2")
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(d)]
    return U[np.ix_(perm, perm)]


def _is_qiskit(circuit: object) -> bool:
    from pytest_quantum.adapters.qiskit import QiskitAdapter
    return QiskitAdapter.detect(circuit)


def _is_cirq(circuit: object) -> bool:
    from pytest_quantum.adapters.cirq import CirqAdapter
    return CirqAdapter.detect(circuit)


def _is_braket(circuit: object) -> bool:
    from pytest_quantum.adapters.braket import BraketAdapter
    return BraketAdapter.detect(circuit)


def _is_pennylane(circuit: object) -> bool:
    from pytest_quantum.adapters.pennylane import PennyLaneAdapter
    return PennyLaneAdapter.detect(circuit)


def _is_pytket(circuit: object) -> bool:
    from pytest_quantum.adapters.pytket import PytketAdapter
    return PytketAdapter.detect(circuit)


def _is_qutip(circuit: object) -> bool:
    from pytest_quantum.adapters.qutip import QutipAdapter
    return QutipAdapter.detect(circuit)


def _is_tequila(circuit: object) -> bool:
    from pytest_quantum.adapters.tequila import TequilaAdapter
    return TequilaAdapter.detect(circuit)


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
    from pytest_quantum.adapters import get_adapter

    try:
        return get_adapter(circuit).to_unitary(circuit)
    except TypeError:
        raise TypeError(
            f"Unrecognised circuit type: {type(circuit).__qualname__!r}.\n"
            "pytest-quantum supports: qiskit.QuantumCircuit, cirq.Circuit, "
            "braket.circuits.Circuit, pennylane QNode, pytket Circuit, "
            "qutip.Qobj, tequila QCircuit, CUDA Quantum kernel, "
            "Qibo Circuit.\n"
            "For graphix patterns use assert_state_fidelity_above() instead."
        )
