"""Base class for quantum framework adapters.

Every supported framework (Qiskit, Cirq, Braket, PennyLane, Pytket, QuTiP,
Tequila) provides a concrete subclass of :class:`FrameworkAdapter`.  The base
class defines the interface with default implementations that raise
:exc:`NotImplementedError`, so each adapter only needs to override the methods
it can actually support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class FrameworkAdapter:
    """Base class for quantum framework adapters.

    Subclasses must set :attr:`framework_name` and override :meth:`detect` at
    minimum.  All other methods raise :exc:`NotImplementedError` by default.

    Attributes:
        framework_name: Short, lowercase identifier for the framework
            (e.g. ``"qiskit"``, ``"cirq"``).
        big_endian: ``True`` if the framework uses big-endian qubit ordering
            (MSB = q0).  Cirq and Pytket are big-endian; Qiskit is
            little-endian.
    """

    framework_name: str = ""
    big_endian: bool = False  # True for Cirq, Pytket (big-endian qubit ordering)

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if this adapter handles the given circuit type.

        Args:
            circuit: An arbitrary Python object that may or may not be a
                quantum circuit from this adapter's framework.

        Returns:
            ``True`` if this adapter can handle *circuit*.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Convert *circuit* to a unitary matrix.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]`` of shape ``(2**n, 2**n)``.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support to_unitary")

    def get_depth(self, circuit: object) -> int:
        """Return the depth (number of time-steps / moments) of *circuit*.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            Circuit depth as an integer.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support get_depth")

    def get_width(self, circuit: object) -> int:
        """Return the number of qubits in *circuit*.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            Qubit count as an integer.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support get_width")

    def count_gates(self, circuit: object) -> dict[str, int]:
        """Return a mapping of gate name → occurrence count.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            Dictionary mapping gate names to their counts.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support count_gates")

    def gate_names(self, circuit: object) -> set[str]:
        """Return the set of distinct gate names used in *circuit*.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            Set of gate name strings.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support gate_names")

    def is_clifford(self, circuit: object) -> bool:
        """Return ``True`` if *circuit* uses only Clifford gates.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            ``True`` if the circuit is a Clifford circuit.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support is_clifford")

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        """Return ``True`` if *circuit* contains a mid-circuit measurement.

        A mid-circuit measurement is one followed by further quantum gates on
        any qubit that was measured.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            ``True`` if mid-circuit measurements are present.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support has_mid_circuit_measurement"
        )

    def get_diagram(self, circuit: object) -> str:
        """Return a text-based diagram of *circuit*.

        Args:
            circuit: A quantum circuit from this adapter's framework.

        Returns:
            Human-readable circuit diagram as a string.

        Raises:
            NotImplementedError: If the adapter does not support this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support get_diagram")
