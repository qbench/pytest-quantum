"""Qiskit framework adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CLIFFORD_QISKIT = frozenset(
    {
        "h",
        "s",
        "sdg",
        "x",
        "y",
        "z",
        "cx",
        "cy",
        "cz",
        "swap",
        "id",
        "sx",
        "sxdg",
        "measure",
        "barrier",
        "reset",
    }
)


class QiskitAdapter(FrameworkAdapter):
    """Adapter for Qiskit ``QuantumCircuit`` objects.

    Qiskit uses little-endian qubit ordering (LSB = q0).
    """

    framework_name = "qiskit"
    big_endian = False  # Qiskit uses little-endian

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if *circuit* is a Qiskit object.

        Args:
            circuit: Any Python object.

        Returns:
            ``True`` when ``type(circuit).__module__`` starts with ``"qiskit"``.
        """
        return type(circuit).__module__.startswith("qiskit")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Convert a Qiskit circuit to a unitary matrix via ``Operator``.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]``.

        Raises:
            ImportError: If ``qiskit`` is not installed.
        """
        try:
            from qiskit.quantum_info import Operator
        except ImportError as exc:
            raise ImportError(
                "qiskit is required for Qiskit circuit support. "
                "Install it with: pip install pytest-quantum[qiskit]"
            ) from exc
        return np.asarray(Operator(circuit).data, dtype=np.complex128)

    def get_depth(self, circuit: object) -> int:
        """Return the circuit depth.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            Circuit depth as an integer.
        """
        return int(cast("Any", circuit).depth())

    def get_width(self, circuit: object) -> int:
        """Return the number of qubits.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            Number of qubits.
        """
        return int(cast("Any", circuit).num_qubits)

    def count_gates(self, circuit: object) -> dict[str, int]:
        """Return gate counts from ``circuit.count_ops()``.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            Dictionary mapping gate name to count.
        """
        return dict(cast("Any", circuit).count_ops())

    def gate_names(self, circuit: object) -> set[str]:
        """Return the set of gate names used in the circuit.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            Set of gate name strings.
        """
        return set(cast("Any", circuit).count_ops().keys())

    def is_clifford(self, circuit: object) -> bool:
        """Return ``True`` if the circuit uses only Clifford gates.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            ``True`` if all gates are in the Clifford set.
        """
        ops = cast("Any", circuit).count_ops()
        non_clifford = [g for g in ops if g not in _CLIFFORD_QISKIT]
        return len(non_clifford) == 0

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        """Return ``True`` if a measurement is followed by a gate on the same qubit.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            ``True`` if mid-circuit measurements are present.
        """
        from qiskit.circuit import Measure

        c = cast("Any", circuit)
        qubit_measure_idx: dict[int, int] = {}
        for idx, instr in enumerate(c.data):
            qubits = [c.find_bit(q).index for q in instr.qubits]
            if isinstance(instr.operation, Measure):
                for q in qubits:
                    qubit_measure_idx[q] = idx
            else:
                if instr.operation.name in ("barrier",):
                    continue
                for q in qubits:
                    if q in qubit_measure_idx:
                        return True
        return False

    def get_diagram(self, circuit: object) -> str:
        """Return a text diagram of the circuit.

        Args:
            circuit: A ``qiskit.QuantumCircuit``.

        Returns:
            Text-based circuit diagram.
        """
        return str(cast("Any", circuit).draw("text"))
