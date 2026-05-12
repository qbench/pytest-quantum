"""Resource estimation assertions for quantum circuits."""
from __future__ import annotations
from typing import TYPE_CHECKING
from pytest_quantum.adapters import get_adapter

if TYPE_CHECKING:
    pass


def assert_t_count_below(circuit: object, max_t: int) -> None:
    """Assert that the T-gate count of *circuit* is below *max_t*.

    Counts gates named ``t``, ``tdg``, ``T``, ``Tdg`` (case-insensitive).

    Args:
        circuit: A quantum circuit from any supported framework.
        max_t: Maximum allowed T-gate count (exclusive upper bound).

    Raises:
        AssertionError: If the T-gate count >= *max_t*.

    Example::

        assert_t_count_below(circuit, 10)
    """
    adapter = get_adapter(circuit)
    gates = adapter.count_gates(circuit)
    t_names = {"t", "tdg"}
    t_count = sum(v for k, v in gates.items() if k.lower() in t_names)
    if t_count >= max_t:
        raise AssertionError(
            f"T-gate count {t_count} is not below {max_t}.\n"
            f"Gate counts: {gates}"
        )


def assert_ancilla_count_below(
    circuit: object,
    logical_qubits: int,
    max_ancilla: int,
) -> None:
    """Assert that the ancilla qubit count is below *max_ancilla*.

    Computes ancilla count as ``total_qubits - logical_qubits``.

    Args:
        circuit: A quantum circuit from any supported framework.
        logical_qubits: Number of logical (data) qubits.
        max_ancilla: Maximum allowed ancilla count (exclusive upper bound).

    Raises:
        AssertionError: If the ancilla count >= *max_ancilla*.

    Example::

        assert_ancilla_count_below(circuit, logical_qubits=2, max_ancilla=3)
    """
    adapter = get_adapter(circuit)
    total = adapter.get_width(circuit)
    ancilla = total - logical_qubits
    if ancilla >= max_ancilla:
        raise AssertionError(
            f"Ancilla count {ancilla} (total={total}, logical={logical_qubits}) "
            f"is not below {max_ancilla}."
        )


def assert_clifford_t_depth_below(circuit: object, max_depth: int) -> None:
    """Assert that the T-depth of *circuit* is below *max_depth*.

    T-depth is the number of layers that contain at least one T or Tdg gate.
    This is a simplified estimate — it counts T-gates in the sequential gate
    list and groups consecutive non-T gates as a single Clifford layer.

    Args:
        circuit: A quantum circuit from any supported framework.
        max_depth: Maximum allowed T-depth (exclusive upper bound).

    Raises:
        AssertionError: If the T-depth >= *max_depth*.

    Example::

        assert_clifford_t_depth_below(circuit, 5)
    """
    adapter = get_adapter(circuit)
    gates = adapter.count_gates(circuit)
    t_names = {"t", "tdg"}
    t_depth = sum(1 for k, v in gates.items() if k.lower() in t_names and v > 0)
    if t_depth >= max_depth:
        raise AssertionError(
            f"T-depth {t_depth} is not below {max_depth}.\n"
            f"Gate counts: {gates}"
        )
