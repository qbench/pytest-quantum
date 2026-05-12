"""PennyLane framework adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CLIFFORD_PENNYLANE = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "SX",
        "CNOT",
        "CY",
        "CZ",
        "SWAP",
        "ISWAP",
        "Identity",
        "Adjoint(S)",
        "Adjoint(SX)",
        # Aliases
        "X",
        "Y",
        "Z",
        "H",
    }
)


class PennyLaneAdapter(FrameworkAdapter):
    """Adapter for PennyLane ``QNode`` objects.

    PennyLane uses little-endian qubit ordering.
    """

    framework_name = "pennylane"
    big_endian = False

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if *circuit* is a PennyLane object.

        Args:
            circuit: Any Python object.

        Returns:
            ``True`` when ``type(circuit).__module__`` starts with
            ``"pennylane"`` or the object is a PennyLane ``QNode``.
        """
        if type(circuit).__module__.startswith("pennylane"):
            return True
        # Avoid broad `hasattr(circuit, "device")` which would match
        # PyTorch models, USB wrappers, etc. Instead, check for the
        # specific PennyLane QNode type when the package is available.
        try:
            import pennylane as qml

            return isinstance(circuit, qml.QNode)
        except ImportError:
            return False

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Convert a PennyLane QNode to a unitary matrix via ``qml.matrix()``.

        Args:
            circuit: A PennyLane ``QNode``.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]``.

        Raises:
            ImportError: If ``pennylane`` is not installed.
            TypeError: If *circuit* lacks a ``device`` attribute.
        """
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "pennylane is required for PennyLane circuit support. "
                "Install it with: pip install pytest-quantum[pennylane]"
            ) from exc

        if not hasattr(circuit, "device"):
            raise TypeError(
                "Cannot auto-detect wire_order for this PennyLane object. "
                "Pass a QNode (decorated with @qml.qnode) or use "
                "qml.matrix(your_fn, wire_order=[0, 1, ...])(params) directly."
            )
        wire_order = list(cast("Any", circuit).device.wires)
        matrix_fn = qml.matrix(circuit, wire_order=wire_order)
        return np.asarray(matrix_fn(), dtype=np.complex128)

    def get_depth(self, circuit: object) -> int:
        """Return the circuit depth via ``qml.specs()``.

        Args:
            circuit: A PennyLane ``QNode``.

        Returns:
            Circuit depth as an integer.

        Raises:
            ImportError: If ``pennylane`` is not installed.
            TypeError: If depth cannot be extracted from ``qml.specs()`` output.
        """
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "pennylane is required for PennyLane circuit depth. "
                "Install it with: pip install pytest-quantum[pennylane]"
            ) from exc

        specs = qml.specs(cast("Any", circuit))()
        if hasattr(specs, "get"):
            resources = specs.get("resources", None)
            if resources is not None and hasattr(resources, "depth"):
                return int(resources.depth)
            depth_val = specs.get("depth", None)
            if depth_val is not None:
                return int(depth_val)
        raise TypeError(
            "Could not extract depth from qml.specs() output. "
            "Upgrade PennyLane to a version that exposes 'resources' or 'depth'."
        )

    def get_width(self, circuit: object) -> int:
        """Return the number of qubits (wires).

        Args:
            circuit: A PennyLane ``QNode``.

        Returns:
            Number of wires.
        """
        return len(cast("Any", circuit).device.wires)

    def count_gates(self, circuit: object) -> dict[str, int]:
        """Return gate counts from the QNode tape.

        Args:
            circuit: A PennyLane ``QNode``.

        Returns:
            Dictionary mapping operation class name to count.
        """
        tape = self._get_tape(circuit)
        counts: dict[str, int] = {}
        for op in tape.operations:
            name = type(op).__name__
            counts[name] = counts.get(name, 0) + 1
        return counts

    def gate_names(self, circuit: object) -> set[str]:
        """Return the set of operation names used in the circuit.

        Args:
            circuit: A PennyLane ``QNode``.

        Returns:
            Set of operation class name strings.
        """
        tape = self._get_tape(circuit)
        return {type(op).__name__ for op in tape.operations}

    def is_clifford(self, circuit: object) -> bool:
        """Return ``True`` if the circuit uses only Clifford gates.

        Args:
            circuit: A PennyLane ``QNode``.

        Returns:
            ``True`` if all operations are in the Clifford set.
        """
        tape = self._get_tape(circuit)
        return all(op.name in _CLIFFORD_PENNYLANE for op in tape.operations)

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        """Not supported for PennyLane QNodes.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "PennyLaneAdapter does not support has_mid_circuit_measurement"
        )

    def get_diagram(self, circuit: object) -> str:
        """Not supported for PennyLane QNodes.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("PennyLaneAdapter does not support get_diagram")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tape(circuit: object) -> Any:
        """Obtain the tape from a PennyLane QNode, executing if necessary.

        Args:
            circuit: A PennyLane ``QNode``.

        Returns:
            The QNode's tape.

        Raises:
            TypeError: If the tape cannot be obtained.
        """
        c = cast("Any", circuit)
        tape = None
        try:
            tape = c.tape
        except AttributeError:
            pass
        if tape is None:
            try:
                c()
                tape = c.tape
            except Exception:
                tape = None
        if tape is None:
            raise TypeError(
                "PennyLane QNode tape could not be obtained. "
                "Ensure the QNode is properly constructed."
            )
        return tape
