"""Tequila framework adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TequilaAdapter(FrameworkAdapter):
    """Adapter for Tequila ``QCircuit`` objects.

    Only :meth:`to_unitary` is supported — it uses column-by-column simulation
    to build the unitary matrix.
    """

    framework_name = "tequila"
    big_endian = False

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if *circuit* is a Tequila object.

        Args:
            circuit: Any Python object.

        Returns:
            ``True`` when ``type(circuit).__module__`` starts with ``"tequila"``.
        """
        return type(circuit).__module__.startswith("tequila")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Extract a unitary matrix via column-by-column simulation.

        Args:
            circuit: A Tequila ``QCircuit``.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]``.

        Raises:
            ImportError: If ``tequila`` is not installed.
        """
        try:
            import tequila as tq
        except ImportError as exc:
            raise ImportError(
                "tequila is required for Tequila support. "
                "Install it with: pip install tequila-basic"
            ) from exc

        c = cast("Any", circuit)
        n_qubits = len(c.qubits)
        dim = 2**n_qubits
        U = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            basis_state = tq.QubitWaveFunction.from_int(i, n_qubits=n_qubits)
            result = tq.simulate(c, initial_state=basis_state)
            for key, val in result.items():
                U[key.integer, i] = val
        return U
