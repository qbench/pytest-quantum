"""Topology and connectivity assertions for quantum circuits."""
from __future__ import annotations
from typing import TYPE_CHECKING
from pytest_quantum.adapters import get_adapter

if TYPE_CHECKING:
    pass


def assert_circuit_respects_topology(
    circuit: object,
    coupling_map: list[tuple[int, int]],
) -> None:
    """Assert that all 2-qubit gates in *circuit* respect the *coupling_map*.

    Each 2-qubit gate's qubit pair must appear in the coupling map as an
    undirected edge.

    Args:
        circuit: A quantum circuit from any supported framework.
        coupling_map: List of ``(qubit_i, qubit_j)`` pairs representing
            allowed 2-qubit interactions.

    Raises:
        AssertionError: If any 2-qubit gate violates the coupling map.

    Example::

        coupling = [(0, 1), (1, 2), (2, 3)]
        assert_circuit_respects_topology(circuit, coupling)
    """
    adapter = get_adapter(circuit)
    gates = adapter.count_gates(circuit)
    # Build undirected edge set
    edges = set()
    for i, j in coupling_map:
        edges.add((i, j))
        edges.add((j, i))

    # We need to inspect the actual circuit structure for qubit operands.
    # Use a framework-specific approach via the adapter's gate inspection.
    # For a generic approach, we check via the adapter's internal circuit representation.
    _check_topology_via_adapter(circuit, adapter, edges)


def _check_topology_via_adapter(circuit: object, adapter: object, edges: set[tuple[int, int]]) -> None:
    """Check topology using framework-specific circuit inspection."""
    mod = type(circuit).__module__

    if mod.startswith("qiskit"):
        _check_qiskit_topology(circuit, edges)
    elif mod.startswith("cirq"):
        _check_cirq_topology(circuit, edges)
    elif mod.startswith("braket"):
        _check_braket_topology(circuit, edges)
    elif mod.startswith("qibo"):
        _check_qibo_topology(circuit, edges)
    else:
        raise NotImplementedError(
            f"Topology checking is not supported for {mod} circuits. "
            f"Supported: qiskit, cirq, braket, qibo."
        )


def _check_qiskit_topology(circuit: object, edges: set[tuple[int, int]]) -> None:
    violations = []
    for instruction in circuit.data:  # type: ignore[attr-defined]
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]  # type: ignore[attr-defined]
        if len(qubits) == 2:
            pair = (qubits[0], qubits[1])
            if pair not in edges:
                violations.append(
                    f"{instruction.operation.name} on qubits {pair}"
                )
    if violations:
        raise AssertionError(
            f"Circuit violates coupling map. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def _check_cirq_topology(circuit: object, edges: set[tuple[int, int]]) -> None:
    import cirq
    violations = []
    qubit_list = sorted(circuit.all_qubits())  # type: ignore[attr-defined]
    qubit_to_idx = {q: i for i, q in enumerate(qubit_list)}
    for op in circuit.all_operations():  # type: ignore[attr-defined]
        if len(op.qubits) == 2:
            idx0 = qubit_to_idx[op.qubits[0]]
            idx1 = qubit_to_idx[op.qubits[1]]
            pair = (idx0, idx1)
            if pair not in edges:
                violations.append(
                    f"{op.gate} on qubits {pair}"
                )
    if violations:
        raise AssertionError(
            f"Circuit violates coupling map. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def _check_braket_topology(circuit: object, edges: set[tuple[int, int]]) -> None:
    violations = []
    for instruction in circuit.instructions:  # type: ignore[attr-defined]
        if len(instruction.target) == 2:
            pair = (int(instruction.target[0]), int(instruction.target[1]))
            if pair not in edges:
                violations.append(
                    f"{instruction.operator.name} on qubits {pair}"
                )
    if violations:
        raise AssertionError(
            f"Circuit violates coupling map. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def _check_qibo_topology(circuit: object, edges: set[tuple[int, int]]) -> None:
    violations = []
    for gate in circuit.queue:  # type: ignore[attr-defined]
        if len(gate.qubits) == 2:
            pair = (gate.qubits[0], gate.qubits[1])
            if pair not in edges:
                violations.append(
                    f"{gate.__class__.__name__} on qubits {pair}"
                )
    if violations:
        raise AssertionError(
            f"Circuit violates coupling map. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def assert_routing_overhead_below(
    circuit: object,
    routed_circuit: object,
    max_overhead: float,
) -> None:
    """Assert that routing overhead is below *max_overhead*.

    Computes ``overhead = (routed_gates - original_gates) / original_gates``
    where gate counts are the total number of gates.

    Args:
        circuit: The original (unrouted) circuit.
        routed_circuit: The routed circuit.
        max_overhead: Maximum allowed overhead as a fraction (e.g. 0.5 for 50%).

    Raises:
        AssertionError: If the overhead >= *max_overhead*.

    Example::

        assert_routing_overhead_below(original, routed, max_overhead=0.5)
    """
    adapter_orig = get_adapter(circuit)
    adapter_routed = get_adapter(routed_circuit)
    orig_count = sum(adapter_orig.count_gates(circuit).values())
    routed_count = sum(adapter_routed.count_gates(routed_circuit).values())
    if orig_count == 0:
        if routed_count == 0:
            return
        raise AssertionError(
            "Original circuit has 0 gates but routed circuit has "
            f"{routed_count} gates."
        )
    overhead = (routed_count - orig_count) / orig_count
    if overhead >= max_overhead:
        raise AssertionError(
            f"Routing overhead {overhead:.2%} is not below {max_overhead:.2%}.\n"
            f"Original gates: {orig_count}, Routed gates: {routed_count}"
        )
