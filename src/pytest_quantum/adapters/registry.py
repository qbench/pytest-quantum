"""Adapter registry — automatic framework detection and adapter lookup.

The module-level :func:`get_adapter` and :func:`register_adapter` functions
delegate to a private :class:`AdapterRegistry` singleton.  Built-in adapters
are auto-registered at import time via the block at the bottom of this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    pass


class AdapterRegistry:
    """Registry that maps circuit objects to the correct :class:`FrameworkAdapter`.

    Adapters are tried in registration order; the first whose :meth:`detect`
    returns ``True`` wins.  Results are cached by the top-level module prefix
    of ``type(circuit).__module__`` (e.g. ``"qiskit"``, ``"cirq"``).

    Attributes:
        _adapters: Registered adapter *classes* in priority order.
        _cache: Instantiated adapters keyed by module prefix.
    """

    def __init__(self) -> None:
        self._adapters: list[type[FrameworkAdapter]] = []
        self._cache: dict[str, FrameworkAdapter] = {}

    def register(self, adapter_cls: type[FrameworkAdapter]) -> None:
        """Register an adapter class.

        Args:
            adapter_cls: A :class:`FrameworkAdapter` subclass.

        Raises:
            TypeError: If *adapter_cls* is not a subclass of
                :class:`FrameworkAdapter`.
        """
        if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, FrameworkAdapter)):
            raise TypeError(
                f"Expected a FrameworkAdapter subclass, got {adapter_cls!r}"
            )
        if adapter_cls not in self._adapters:
            self._adapters.append(adapter_cls)

    def get(self, circuit: object) -> FrameworkAdapter:
        """Return the adapter instance for *circuit*.

        Args:
            circuit: A quantum circuit from any supported framework.

        Returns:
            A :class:`FrameworkAdapter` instance that can handle *circuit*.

        Raises:
            TypeError: If no registered adapter recognises *circuit*.
        """
        module = type(circuit).__module__
        prefix = module.split(".")[0]

        if prefix in self._cache:
            return self._cache[prefix]

        for adapter_cls in self._adapters:
            if adapter_cls.detect(circuit):
                instance = adapter_cls()
                self._cache[prefix] = instance
                return instance

        raise TypeError(
            f"Unrecognised circuit type: {type(circuit).__qualname__!r} "
            f"(module={module!r}).\n"
            "pytest-quantum supports: qiskit.QuantumCircuit, cirq.Circuit, "
            "braket.circuits.Circuit, pennylane QNode, pytket Circuit, "
            "qutip.Qobj, tequila QCircuit, CUDA Quantum kernel, "
            "Qibo Circuit.\n"
            "Register a custom adapter with register_adapter()."
        )

    def detect_framework(self, circuit: object) -> str:
        """Return the framework name for *circuit*.

        Args:
            circuit: A quantum circuit from any supported framework.

        Returns:
            The :attr:`FrameworkAdapter.framework_name` string
            (e.g. ``"qiskit"``).

        Raises:
            TypeError: If no registered adapter recognises *circuit*.
        """
        adapter = self.get(circuit)
        return adapter.framework_name


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_registry = AdapterRegistry()


def get_adapter(circuit: object) -> FrameworkAdapter:
    """Return the :class:`FrameworkAdapter` for *circuit*.

    Args:
        circuit: A quantum circuit from any supported framework.

    Returns:
        A :class:`FrameworkAdapter` instance.

    Raises:
        TypeError: If no registered adapter recognises *circuit*.
    """
    return _registry.get(circuit)


def register_adapter(adapter_cls: type[FrameworkAdapter]) -> None:
    """Register a custom adapter class.

    Args:
        adapter_cls: A :class:`FrameworkAdapter` subclass.
    """
    _registry.register(adapter_cls)


def detect_framework(circuit: object) -> str:
    """Return the framework name for *circuit*.

    Args:
        circuit: A quantum circuit from any supported framework.

    Returns:
        Framework name string (e.g. ``"qiskit"``).

    Raises:
        TypeError: If no registered adapter recognises *circuit*.
    """
    return _registry.detect_framework(circuit)


# ---------------------------------------------------------------------------
# Auto-register built-in adapters
# ---------------------------------------------------------------------------

try:
    from pytest_quantum.adapters.qiskit import QiskitAdapter

    _registry.register(QiskitAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.cirq import CirqAdapter

    _registry.register(CirqAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.braket import BraketAdapter

    _registry.register(BraketAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.pennylane import PennyLaneAdapter

    _registry.register(PennyLaneAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.pytket import PytketAdapter

    _registry.register(PytketAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.qutip import QutipAdapter

    _registry.register(QutipAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.tequila import TequilaAdapter

    _registry.register(TequilaAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.cuda_quantum import CudaQuantumAdapter

    _registry.register(CudaQuantumAdapter)
except ImportError:  # pragma: no cover
    pass

try:
    from pytest_quantum.adapters.qibo import QiboAdapter

    _registry.register(QiboAdapter)
except ImportError:  # pragma: no cover
    pass
