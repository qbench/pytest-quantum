"""Amazon Braket framework adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CLIFFORD_BRAKET = frozenset(
    {"H", "X", "Y", "Z", "S", "Si", "CNot", "CZ", "Swap", "CY", "I", "V", "Vi"}
)


class BraketAdapter(FrameworkAdapter):
    """Adapter for Amazon Braket ``Circuit`` objects.

    Braket uses little-endian qubit ordering.
    """

    framework_name = "braket"
    big_endian = False

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if *circuit* is a Braket object.

        Args:
            circuit: Any Python object.

        Returns:
            ``True`` when ``type(circuit).__module__`` starts with ``"braket"``.
        """
        return type(circuit).__module__.startswith("braket")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Convert a Braket circuit to a unitary matrix.

        Args:
            circuit: A ``braket.circuits.Circuit``.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]``.

        Raises:
            TypeError: If the circuit lacks ``to_unitary()``.
            ImportError: If ``amazon-braket-sdk`` is not installed.
        """
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

    def get_depth(self, circuit: object) -> int:
        """Return the circuit depth.

        Args:
            circuit: A ``braket.circuits.Circuit``.

        Returns:
            Circuit depth as an integer.
        """
        return int(cast("Any", circuit).depth)

    def get_width(self, circuit: object) -> int:
        """Return the number of qubits.

        Args:
            circuit: A ``braket.circuits.Circuit``.

        Returns:
            Number of qubits.
        """
        return int(cast("Any", circuit).qubit_count)

    def count_gates(self, circuit: object) -> dict[str, int]:
        """Return gate counts by operator name.

        Args:
            circuit: A ``braket.circuits.Circuit``.

        Returns:
            Dictionary mapping operator name to count.
        """
        counts: dict[str, int] = {}
        for instr in cast("Any", circuit).instructions:
            name = instr.operator.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def gate_names(self, circuit: object) -> set[str]:
        """Return the set of operator names used in the circuit.

        Args:
            circuit: A ``braket.circuits.Circuit``.

        Returns:
            Set of operator name strings.
        """
        return {instr.operator.name for instr in cast("Any", circuit).instructions}

    def is_clifford(self, circuit: object) -> bool:
        """Return ``True`` if the circuit uses only Clifford gates.

        Args:
            circuit: A ``braket.circuits.Circuit``.

        Returns:
            ``True`` if all operators are in the Clifford set.
        """
        for instr in cast("Any", circuit).instructions:
            if type(instr.operator).__name__ not in _CLIFFORD_BRAKET:
                return False
        return True

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        """Not supported for Braket circuits.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "BraketAdapter does not support has_mid_circuit_measurement"
        )

    def get_diagram(self, circuit: object) -> str:
        """Not supported for Braket circuits.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("BraketAdapter does not support get_diagram")
