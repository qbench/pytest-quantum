"""Cirq framework adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CLIFFORD_CIRQ = frozenset(
    {"h", "x", "y", "z", "s", "cnot", "cz", "swap", "i", "measure"}
)


class CirqAdapter(FrameworkAdapter):
    """Adapter for Cirq ``Circuit`` objects.

    Cirq uses big-endian qubit ordering (MSB = q0).
    """

    framework_name = "cirq"
    big_endian = True

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if *circuit* is a Cirq object.

        Args:
            circuit: Any Python object.

        Returns:
            ``True`` when ``type(circuit).__module__`` starts with ``"cirq"``.
        """
        return type(circuit).__module__.startswith("cirq")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Convert a Cirq circuit to a unitary matrix via ``cirq.unitary()``.

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]``.

        Raises:
            ImportError: If ``cirq`` is not installed.
        """
        try:
            import cirq
        except ImportError as exc:
            raise ImportError(
                "cirq is required for Cirq circuit support. "
                "Install it with: pip install pytest-quantum[cirq]"
            ) from exc
        return np.asarray(cirq.unitary(circuit), dtype=np.complex128)

    def get_depth(self, circuit: object) -> int:
        """Return the circuit depth (number of moments).

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            Number of moments.
        """
        return len(cast("Any", circuit))

    def get_width(self, circuit: object) -> int:
        """Return the number of qubits.

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            Number of qubits.
        """
        return len(cast("Any", circuit).all_qubits())

    def count_gates(self, circuit: object) -> dict[str, int]:
        """Return gate counts by lowercased gate name.

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            Dictionary mapping lowercased gate name to count.
        """
        counts: dict[str, int] = {}
        for op in cast("Any", circuit).all_operations():
            name = str(op.gate).lower()
            counts[name] = counts.get(name, 0) + 1
        return counts

    def gate_names(self, circuit: object) -> set[str]:
        """Return the set of gate names used in the circuit.

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            Set of gate name strings (original case).
        """
        return {str(op.gate) for op in cast("Any", circuit).all_operations()}

    def is_clifford(self, circuit: object) -> bool:
        """Return ``True`` if the circuit uses only Clifford gates.

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            ``True`` if all gates are in the Clifford set.
        """
        for op in cast("Any", circuit).all_operations():
            name = str(op.gate).lower()
            if name not in _CLIFFORD_CIRQ:
                return False
        return True

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        """Return ``True`` if a measurement moment appears before the last gate moment.

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            ``True`` if mid-circuit measurements are present.
        """
        try:
            import cirq
        except ImportError as exc:
            raise ImportError(
                "cirq is required for Cirq circuit support. "
                "Install it with: pip install pytest-quantum[cirq]"
            ) from exc

        c = cast("Any", circuit)
        last_gate_moment_idx = -1
        first_measure_moment_idx = -1

        for idx, moment in enumerate(c):
            has_gate = False
            has_measure = False
            for op in moment.operations:
                if isinstance(op.gate, cirq.MeasurementGate):
                    has_measure = True
                else:
                    has_gate = True
            if has_gate:
                last_gate_moment_idx = idx
            if has_measure and first_measure_moment_idx == -1:
                first_measure_moment_idx = idx

        if first_measure_moment_idx == -1:
            return False
        return first_measure_moment_idx < last_gate_moment_idx

    def get_diagram(self, circuit: object) -> str:
        """Return a text diagram of the circuit.

        Args:
            circuit: A ``cirq.Circuit``.

        Returns:
            Text-based circuit diagram.
        """
        return str(cast("Any", circuit))
