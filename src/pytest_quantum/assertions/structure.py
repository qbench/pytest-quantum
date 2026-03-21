"""Circuit structure assertions.

These assertions check static properties of a circuit — depth, gate counts,
qubit width — without executing it.  Useful for catching regressions in
compiler output or ensuring a circuit meets hardware constraints.
"""

from __future__ import annotations

from typing import Any


def assert_circuit_depth(
    circuit: object,
    *,
    max_depth: int | None = None,
    min_depth: int | None = None,
) -> None:
    """Assert that a circuit's depth is within the specified bounds.

    At least one of *max_depth* or *min_depth* must be provided.

    Supported frameworks: Qiskit, Cirq, Amazon Braket.

    Args:
        circuit:   A quantum circuit from a supported framework.
        max_depth: If given, the circuit depth must be ≤ this value.
        min_depth: If given, the circuit depth must be ≥ this value.

    Raises:
        AssertionError: If the depth is outside the specified bounds.
        TypeError:      If the circuit type is not supported.
        ValueError:     If neither bound is provided.

    Example::

        def test_circuit_depth():
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            assert_circuit_depth(qc, max_depth=3)
    """
    if max_depth is None and min_depth is None:
        raise ValueError("Provide at least one of max_depth or min_depth.")

    depth = _get_depth(circuit)

    if max_depth is not None and depth > max_depth:
        raise AssertionError(f"Circuit depth {depth} exceeds max_depth {max_depth}.")
    if min_depth is not None and depth < min_depth:
        raise AssertionError(f"Circuit depth {depth} is below min_depth {min_depth}.")


def assert_circuit_width(
    circuit: object,
    expected_qubits: int,
) -> None:
    """Assert that a circuit acts on exactly *expected_qubits* qubits.

    Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane.

    Args:
        circuit:         A quantum circuit from a supported framework.
        expected_qubits: Expected number of qubits.

    Raises:
        AssertionError:  If the qubit count does not match.
        TypeError:       If the circuit type is not supported.

    Example::

        def test_circuit_width():
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            assert_circuit_width(qc, expected_qubits=3)
    """
    actual = _get_width(circuit)
    if actual != expected_qubits:
        raise AssertionError(
            f"Circuit qubit count mismatch.\n"
            f"  Expected : {expected_qubits}\n"
            f"  Actual   : {actual}"
        )


def assert_gate_count(
    circuit: object,
    gate_name: str,
    expected: int,
) -> None:
    """Assert that a circuit contains exactly *expected* occurrences of *gate_name*.

    Supported frameworks: Qiskit, Cirq, PennyLane.

    Args:
        circuit:    A quantum circuit from a supported framework.
        gate_name:  Gate name as a string, e.g. ``"cx"``, ``"h"``, ``"t"``,
                    ``"CNOT"``, ``"Hadamard"``.  Case-insensitive for Qiskit;
                    Cirq and PennyLane match case-insensitively by gate class name.
        expected:   Expected count.

    Raises:
        AssertionError:      If the actual count differs from *expected*.
        NotImplementedError: If the framework is not yet supported.

    Example::

        def test_t_count():
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2)
            qc.t(0)
            qc.t(1)
            qc.cx(0, 1)
            assert_gate_count(qc, "t", 2)
            assert_gate_count(qc, "cx", 1)
    """
    module = type(circuit).__module__
    c: Any = circuit

    if module.startswith("qiskit"):
        ops = c.count_ops()
        actual = ops.get(gate_name.lower(), 0)

    elif module.startswith("cirq"):
        # Count operations by matching str(op.gate) which gives human-readable names
        # e.g. cirq.H -> "H", cirq.CNOT -> "CNOT", cirq.CZ -> "CZ"
        name_lower = gate_name.lower()
        actual = sum(
            1
            for moment in c
            for op in moment.operations
            if str(op.gate).lower() == name_lower
        )

    elif module.startswith("pennylane"):
        # QNode: inspect the tape after a dry run, or count via string matching
        # For PennyLane, gate names match operation class names (e.g. "CNOT", "Hadamard")
        name_lower = gate_name.lower()
        try:
            tape = c.tape  # works if QNode has been called at least once
            actual = sum(
                1 for op in tape.operations if type(op).__name__.lower() == name_lower
            )
        except AttributeError as exc:
            raise TypeError(
                "PennyLane QNode must be called at least once before gate count "
                "can be inspected via assert_gate_count. Call circuit() first."
            ) from exc

    else:
        raise NotImplementedError(
            f"assert_gate_count supports Qiskit, Cirq, and PennyLane. "
            f"Got circuit type: {type(circuit).__qualname__!r}."
        )

    if actual != expected:
        raise AssertionError(
            f"Gate count mismatch for {gate_name!r}.\n"
            f"  Expected : {expected}\n"
            f"  Actual   : {actual}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_depth(circuit: object) -> int:
    """Extract depth from any supported circuit type."""
    module = type(circuit).__module__
    c: Any = circuit

    if module.startswith("qiskit"):
        return int(c.depth())

    if module.startswith("cirq"):
        # cirq.Circuit depth = number of non-empty moments
        return len(c)

    if module.startswith("braket"):
        return int(c.depth)

    raise TypeError(
        f"assert_circuit_depth does not support circuit type "
        f"{type(circuit).__qualname__!r}.\n"
        "Supported frameworks: Qiskit, Cirq, Amazon Braket."
    )


def _get_width(circuit: object) -> int:
    """Extract qubit count from any supported circuit type."""
    module = type(circuit).__module__
    c: Any = circuit

    if module.startswith("qiskit"):
        return int(c.num_qubits)

    if module.startswith("cirq"):
        return len(c.all_qubits())

    if module.startswith("braket"):
        return int(c.qubit_count)

    if module.startswith("pennylane") or hasattr(circuit, "device"):
        return len(c.device.wires)

    raise TypeError(
        f"assert_circuit_width does not support circuit type "
        f"{type(circuit).__qualname__!r}.\n"
        "Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane."
    )
