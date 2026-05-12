"""Resource estimation assertions for quantum circuits."""

from __future__ import annotations

from typing import Any, cast

from pytest_quantum.adapters import get_adapter

_T_GATE_NAMES = frozenset({"t", "tdg"})


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
    t_count = sum(v for k, v in gates.items() if k.lower() in _T_GATE_NAMES)
    if t_count >= max_t:
        raise AssertionError(
            f"T-gate count {t_count} is not below {max_t}.\nGate counts: {gates}"
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


def _compute_t_depth_qiskit(circuit: object) -> int:
    """Compute T-depth for a Qiskit circuit using DAGCircuit layers."""
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(cast("Any", circuit))
    t_depth = 0
    for layer in dag.layers():
        layer_has_t = False
        for node in layer["graph"].op_nodes():
            if node.op.name.lower() in _T_GATE_NAMES:
                layer_has_t = True
                break
        if layer_has_t:
            t_depth += 1
    return t_depth


def _compute_t_depth_cirq(circuit: object) -> int:
    """Compute T-depth for a Cirq circuit using moments."""
    t_depth = 0
    for moment in cast("Any", circuit):
        moment_has_t = False
        for op in moment.operations:
            gate_name = str(op.gate).lower() if hasattr(op, "gate") else ""
            if gate_name in _T_GATE_NAMES or gate_name in ("t**1", "t**-1"):
                moment_has_t = True
                break
        if moment_has_t:
            t_depth += 1
    return t_depth


def _compute_t_depth_generic(circuit: object) -> int:
    """Compute T-depth using a generic ASAP scheduling simulation.

    Walks the gate list sequentially, tracking the time-step at which each
    qubit becomes free. Each T/Tdg gate is placed at the earliest available
    time-step for its qubit(s). The T-depth is the number of distinct
    time-steps that contain at least one T/Tdg gate.
    """
    adapter = get_adapter(circuit)
    mod = type(circuit).__module__

    # Attempt to get ordered operations with qubit info
    ops: list[tuple[str, list[int]]] = []

    if mod.startswith("qiskit"):
        c = cast("Any", circuit)
        for instr in c.data:
            name = instr.operation.name
            qubits = [c.find_bit(q).index for q in instr.qubits]
            ops.append((name, qubits))
    elif mod.startswith("braket"):
        c = cast("Any", circuit)
        for instr in c.instructions:
            name = (
                instr.operator.name
                if hasattr(instr.operator, "name")
                else type(instr.operator).__name__
            )
            qubits = [int(q) for q in instr.target]
            ops.append((name, qubits))
    else:
        # Fallback: just count distinct T-gate types present (original behaviour)
        gates = adapter.count_gates(circuit)
        return sum(v for k, v in gates.items() if k.lower() in _T_GATE_NAMES)

    if not ops:
        return 0

    # ASAP scheduling
    qubit_time: dict[int, int] = {}  # qubit -> next available time-step
    t_layers: set[int] = set()

    for name, qubits in ops:
        # Determine the earliest time-step this gate can run
        t = max((qubit_time.get(q, 0) for q in qubits), default=0)
        # Advance all involved qubits past this time-step
        for q in qubits:
            qubit_time[q] = t + 1
        if name.lower() in _T_GATE_NAMES:
            t_layers.add(t)

    return len(t_layers)


def assert_clifford_t_depth_below(circuit: object, max_depth: int) -> None:
    """Assert that the T-depth of *circuit* is below *max_depth*.

    T-depth is the number of circuit layers that contain at least one T or
    Tdg gate, respecting qubit dependencies. For Qiskit circuits the
    DAGCircuit layer decomposition is used; for Cirq circuits the moment
    structure is used; other frameworks fall back to ASAP scheduling
    simulation.

    Args:
        circuit: A quantum circuit from any supported framework.
        max_depth: Maximum allowed T-depth (exclusive upper bound).

    Raises:
        AssertionError: If the T-depth >= *max_depth*.

    Example::

        assert_clifford_t_depth_below(circuit, 5)
    """
    mod = type(circuit).__module__

    if mod.startswith("qiskit"):
        t_depth = _compute_t_depth_qiskit(circuit)
    elif mod.startswith("cirq"):
        t_depth = _compute_t_depth_cirq(circuit)
    else:
        t_depth = _compute_t_depth_generic(circuit)

    if t_depth >= max_depth:
        adapter = get_adapter(circuit)
        gates = adapter.count_gates(circuit)
        raise AssertionError(
            f"T-depth {t_depth} is not below {max_depth}.\nGate counts: {gates}"
        )
