"""Pytket framework adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PytketAdapter(FrameworkAdapter):
    """Adapter for Pytket ``Circuit`` objects.

    Pytket uses big-endian (ILO-BE) qubit ordering.
    """

    framework_name = "pytket"
    big_endian = True

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if *circuit* is a Pytket object.

        Args:
            circuit: Any Python object.

        Returns:
            ``True`` when ``type(circuit).__module__`` starts with ``"pytket"``.
        """
        return type(circuit).__module__.startswith("pytket")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Convert a Pytket circuit to a unitary matrix.

        Args:
            circuit: A ``pytket.Circuit``.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]``.

        Raises:
            ImportError: If ``pytket`` is not installed.
        """
        try:
            U = np.asarray(cast("Any", circuit).get_unitary())
            return U.astype(np.complex128)
        except AttributeError as exc:
            raise ImportError(
                "pytket is required for Pytket circuit support. "
                "Install it with: pip install pytket"
            ) from exc
        except ImportError as exc:
            raise ImportError("pytket is required: pip install pytket") from exc

    def get_depth(self, circuit: object) -> int:
        """Return the circuit depth.

        Args:
            circuit: A ``pytket.Circuit``.

        Returns:
            Circuit depth as an integer.
        """
        return int(cast("Any", circuit).depth())

    def get_width(self, circuit: object) -> int:
        """Return the number of qubits.

        Args:
            circuit: A ``pytket.Circuit``.

        Returns:
            Number of qubits.
        """
        return int(cast("Any", circuit).n_qubits)

    def count_gates(self, circuit: object) -> dict[str, int]:
        """Return gate counts by ``OpType`` name.

        Args:
            circuit: A ``pytket.Circuit``.

        Returns:
            Dictionary mapping ``OpType`` name to count.
        """
        counts: dict[str, int] = {}
        for cmd in cast("Any", circuit).get_commands():
            name = cmd.op.type.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def gate_names(self, circuit: object) -> set[str]:
        """Return the set of ``OpType`` names used in the circuit.

        Args:
            circuit: A ``pytket.Circuit``.

        Returns:
            Set of ``OpType`` name strings.
        """
        return {cmd.op.type.name for cmd in cast("Any", circuit).get_commands()}

    def is_clifford(self, circuit: object) -> bool:
        """Return ``True`` if the circuit uses only Clifford gates.

        Uses ``pytket.tableau.UnitaryTableau`` to check — if the tableau can be
        constructed without error, the circuit is Clifford.

        Args:
            circuit: A ``pytket.Circuit``.

        Returns:
            ``True`` if the circuit is a Clifford circuit.

        Raises:
            ImportError: If ``pytket`` is not installed.
        """
        try:
            from pytket.tableau import UnitaryTableau

            UnitaryTableau(circuit)
            return True
        except ImportError as exc:
            raise ImportError("pytket is required: pip install pytket") from exc
        except Exception:
            return False

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        """Not supported for Pytket circuits.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "PytketAdapter does not support has_mid_circuit_measurement"
        )

    def get_diagram(self, circuit: object) -> str:
        """Return the string representation of the circuit.

        Args:
            circuit: A ``pytket.Circuit``.

        Returns:
            Text-based circuit representation.
        """
        return str(cast("Any", circuit))
