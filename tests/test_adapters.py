"""Tests for the adapter abstraction layer.

Verifies that ``AdapterRegistry``, ``get_adapter``, ``register_adapter``,
and ``detect_framework`` work correctly with mock circuit objects.
"""

from __future__ import annotations

import types

import pytest

from pytest_quantum.adapters import (
    AdapterRegistry,
    FrameworkAdapter,
    get_adapter,
    register_adapter,
)
from pytest_quantum.adapters.registry import detect_framework


# ---------------------------------------------------------------------------
# Helpers — mock circuit objects with controllable __module__
# ---------------------------------------------------------------------------


def _make_mock_circuit(module_prefix: str) -> object:
    """Create a minimal object whose type.__module__ starts with *module_prefix*."""
    # Dynamically create a class whose __module__ is set to the desired prefix
    cls = type("MockCircuit", (), {})
    cls.__module__ = module_prefix
    return cls()


# ---------------------------------------------------------------------------
# AdapterRegistry.register() and get()
# ---------------------------------------------------------------------------


class _TestAdapter(FrameworkAdapter):
    framework_name = "test_framework"

    @classmethod
    def detect(cls, circuit: object) -> bool:
        return type(circuit).__module__.startswith("test_mod")


class TestAdapterRegistry:
    """Unit tests for :class:`AdapterRegistry`."""

    def test_register_and_get(self) -> None:
        """Registering an adapter allows ``get()`` to find it."""
        registry = AdapterRegistry()
        registry.register(_TestAdapter)

        circuit = _make_mock_circuit("test_mod.submod")
        adapter = registry.get(circuit)
        assert isinstance(adapter, _TestAdapter)
        assert adapter.framework_name == "test_framework"

    def test_get_caches_by_module_prefix(self) -> None:
        """The same adapter instance is returned for the same module prefix."""
        registry = AdapterRegistry()
        registry.register(_TestAdapter)

        c1 = _make_mock_circuit("test_mod.a")
        c2 = _make_mock_circuit("test_mod.b")
        assert registry.get(c1) is registry.get(c2)

    def test_get_raises_type_error_for_unknown(self) -> None:
        """``get()`` raises ``TypeError`` for unrecognised circuit types."""
        registry = AdapterRegistry()
        registry.register(_TestAdapter)

        unknown = _make_mock_circuit("unknown.module")
        with pytest.raises(TypeError, match="Unrecognised circuit type"):
            registry.get(unknown)

    def test_register_rejects_non_adapter(self) -> None:
        """``register()`` raises ``TypeError`` for non-adapter classes."""
        registry = AdapterRegistry()
        with pytest.raises(TypeError, match="FrameworkAdapter subclass"):
            registry.register(object)  # type: ignore[arg-type]

    def test_register_idempotent(self) -> None:
        """Registering the same adapter twice is a no-op."""
        registry = AdapterRegistry()
        registry.register(_TestAdapter)
        registry.register(_TestAdapter)
        assert registry._adapters.count(_TestAdapter) == 1


# ---------------------------------------------------------------------------
# detect_framework()
# ---------------------------------------------------------------------------


class TestDetectFramework:
    """Tests for :func:`detect_framework`."""

    @pytest.mark.parametrize(
        "module_prefix,expected_name",
        [
            ("qiskit.test", "qiskit"),
            ("cirq.test", "cirq"),
            ("braket.test", "braket"),
            ("pennylane.test", "pennylane"),
            ("pytket.test", "pytket"),
            ("qutip.test", "qutip"),
            ("tequila.test", "tequila"),
        ],
    )
    def test_detect_builtin_frameworks(
        self, module_prefix: str, expected_name: str
    ) -> None:
        """Built-in adapters detect objects by module prefix."""
        circuit = _make_mock_circuit(module_prefix)
        assert detect_framework(circuit) == expected_name


# ---------------------------------------------------------------------------
# get_adapter() module-level function
# ---------------------------------------------------------------------------


class TestGetAdapter:
    """Tests for the module-level :func:`get_adapter`."""

    def test_raises_for_unknown(self) -> None:
        """``get_adapter()`` raises ``TypeError`` for unknown types."""
        unknown = _make_mock_circuit("unknown.module")
        with pytest.raises(TypeError):
            get_adapter(unknown)

    def test_returns_adapter_for_known(self) -> None:
        """``get_adapter()`` returns an adapter for known circuit types."""
        circuit = _make_mock_circuit("qiskit.circuit")
        adapter = get_adapter(circuit)
        assert adapter.framework_name == "qiskit"


# ---------------------------------------------------------------------------
# register_adapter() module-level function
# ---------------------------------------------------------------------------


class _CustomAdapter(FrameworkAdapter):
    framework_name = "custom"

    @classmethod
    def detect(cls, circuit: object) -> bool:
        return type(circuit).__module__.startswith("my_custom_sdk")


class TestRegisterAdapter:
    """Tests for the module-level :func:`register_adapter`."""

    def test_register_custom_adapter(self) -> None:
        """A custom adapter can be registered and found."""
        register_adapter(_CustomAdapter)
        circuit = _make_mock_circuit("my_custom_sdk.circuits")
        adapter = get_adapter(circuit)
        assert adapter.framework_name == "custom"


# ---------------------------------------------------------------------------
# FrameworkAdapter base class
# ---------------------------------------------------------------------------


class TestFrameworkAdapterBase:
    """Ensure base class methods raise NotImplementedError."""

    def test_detect_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            FrameworkAdapter.detect(object())

    def test_to_unitary_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support to_unitary"):
            FrameworkAdapter().to_unitary(object())

    def test_get_depth_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support get_depth"):
            FrameworkAdapter().get_depth(object())

    def test_get_width_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support get_width"):
            FrameworkAdapter().get_width(object())

    def test_count_gates_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support count_gates"):
            FrameworkAdapter().count_gates(object())

    def test_gate_names_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support gate_names"):
            FrameworkAdapter().gate_names(object())

    def test_is_clifford_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support is_clifford"):
            FrameworkAdapter().is_clifford(object())

    def test_has_mid_circuit_measurement_raises(self) -> None:
        with pytest.raises(
            NotImplementedError,
            match="does not support has_mid_circuit_measurement",
        ):
            FrameworkAdapter().has_mid_circuit_measurement(object())

    def test_get_diagram_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support get_diagram"):
            FrameworkAdapter().get_diagram(object())
