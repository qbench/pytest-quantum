"""Topology and connectivity assertions for quantum circuits."""

from __future__ import annotations

import itertools

from pytest_quantum.adapters import get_adapter


def _check_qubit_pairs(
    gate_name: str,
    qubits: list[int],
    edges: set[tuple[int, int]],
    violations: list[str],
) -> None:
    """Validate that all pairwise qubit combinations for a gate are in *edges*.

    For 2-qubit gates, checks the single pair. For 3+ qubit gates, checks
    every pairwise combination.
    """
    if len(qubits) < 2:
        return
    for qi, qj in itertools.combinations(qubits, 2):
        pair = (qi, qj)
        if pair not in edges and (qj, qi) not in edges:
            violations.append(
                f"{gate_name} on qubits {tuple(qubits)} — missing edge {pair}"
            )


def assert_circuit_respects_topology(
    circuit: object,
    coupling_map: list[tuple[int, int]],
) -> None:
    """Assert that all multi-qubit gates in *circuit* respect the *coupling_map*.

    Each 2-qubit gate's qubit pair must appear in the coupling map as an
    undirected edge. Gates acting on 3 or more qubits are validated by
    checking that every pairwise combination of their qubit operands
    appears in the coupling map.

    Args:
        circuit: A quantum circuit from any supported framework.
        coupling_map: List of ``(qubit_i, qubit_j)`` pairs representing
            allowed 2-qubit interactions.

    Raises:
        AssertionError: If any multi-qubit gate violates the coupling map.

    Example::

        coupling = [(0, 1), (1, 2), (2, 3)]
        assert_circuit_respects_topology(circuit, coupling)
    """
    adapter = get_adapter(circuit)
    adapter.count_gates(circuit)
    # Build undirected edge set
    edges = set()
    for i, j in coupling_map:
        edges.add((i, j))
        edges.add((j, i))

    # We need to inspect the actual circuit structure for qubit operands.
    # Use a framework-specific approach via the adapter's gate inspection.
    # For a generic approach, we check via the adapter's internal circuit representation.
    _check_topology_via_adapter(circuit, adapter, edges)


def _check_topology_via_adapter(
    circuit: object, adapter: object, edges: set[tuple[int, int]]
) -> None:
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
    violations: list[str] = []
    for instruction in circuit.data:  # type: ignore[attr-defined]
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]  # type: ignore[attr-defined]
        if len(qubits) >= 2:
            _check_qubit_pairs(instruction.operation.name, qubits, edges, violations)
    if violations:
        raise AssertionError(
            "Circuit violates coupling map. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def _check_cirq_topology(circuit: object, edges: set[tuple[int, int]]) -> None:
    violations: list[str] = []
    qubit_list = sorted(circuit.all_qubits())  # type: ignore[attr-defined]
    qubit_to_idx = {q: i for i, q in enumerate(qubit_list)}
    for op in circuit.all_operations():  # type: ignore[attr-defined]
        if len(op.qubits) >= 2:
            indices = [qubit_to_idx[q] for q in op.qubits]
            gate_name = str(op.gate) if hasattr(op, "gate") else str(op)
            _check_qubit_pairs(gate_name, indices, edges, violations)
    if violations:
        raise AssertionError(
            "Circuit violates coupling map. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def _check_braket_topology(circuit: object, edges: set[tuple[int, int]]) -> None:
    violations: list[str] = []
    for instruction in circuit.instructions:  # type: ignore[attr-defined]
        if len(instruction.target) >= 2:
            qubits = [int(q) for q in instruction.target]
            gate_name = (
                instruction.operator.name
                if hasattr(instruction.operator, "name")
                else type(instruction.operator).__name__
            )
            _check_qubit_pairs(gate_name, qubits, edges, violations)
    if violations:
        raise AssertionError(
            "Circuit violates coupling map. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def _check_qibo_topology(circuit: object, edges: set[tuple[int, int]]) -> None:
    violations: list[str] = []
    for gate in circuit.queue:  # type: ignore[attr-defined]
        if len(gate.qubits) >= 2:
            qubits = list(gate.qubits)
            _check_qubit_pairs(gate.__class__.__name__, qubits, edges, violations)
    if violations:
        raise AssertionError(
            "Circuit violates coupling map. Violations:\n"
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
            f"Original circuit has 0 gates but routed circuit has {routed_count} gates."
        )
    overhead = (routed_count - orig_count) / orig_count
    if overhead >= max_overhead:
        raise AssertionError(
            f"Routing overhead {overhead:.2%} is not below {max_overhead:.2%}.\n"
            f"Original gates: {orig_count}, Routed gates: {routed_count}"
        )
