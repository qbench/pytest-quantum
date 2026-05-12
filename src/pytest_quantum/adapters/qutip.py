"""QuTiP framework adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class QutipAdapter(FrameworkAdapter):
    """Adapter for QuTiP ``Qobj`` objects.

    Only :meth:`to_unitary` is supported — QuTiP ``Qobj`` objects represent
    operators (not circuits) so structural queries are not meaningful.
    """

    framework_name = "qutip"
    big_endian = False

    @classmethod
    def detect(cls, circuit: object) -> bool:
        """Return ``True`` if *circuit* is a QuTiP object.

        Args:
            circuit: Any Python object.

        Returns:
            ``True`` when ``type(circuit).__module__`` starts with ``"qutip"``.
        """
        return type(circuit).__module__.startswith("qutip")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Extract a unitary matrix from a QuTiP ``Qobj``.

        The ``Qobj`` must be a unitary operator (``obj.type == 'oper'``)
        with a square matrix representation.

        Args:
            circuit: A ``qutip.Qobj``.

        Returns:
            Unitary matrix as ``NDArray[np.complex128]``.

        Raises:
            ImportError: If ``qutip`` is not installed.
            TypeError: If the ``Qobj`` cannot be converted to a unitary.
        """
        try:
            import qutip  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "qutip is required for QuTiP support. "
                "Install it with: pip install qutip"
            ) from exc

        obj = cast("Any", circuit)
        if hasattr(obj, "full"):
            U = np.asarray(obj.full(), dtype=np.complex128)
            if U.ndim == 2 and U.shape[0] == U.shape[1]:
                return U
        raise TypeError(
            f"QuTiP Qobj of type {obj.type!r} cannot be converted to a unitary matrix. "
            "Pass a unitary operator Qobj (obj.type == 'oper')."
        )
