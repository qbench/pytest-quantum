"""Circuit structure assertions.

These assertions check static properties of a circuit — depth, gate counts,
qubit width — without executing it.  Useful for catching regressions in
compiler output or ensuring a circuit meets hardware constraints.
"""

from __future__ import annotations

from typing import Any, cast

# ---------------------------------------------------------------------------
# Clifford gate sets per framework
# ---------------------------------------------------------------------------
_CLIFFORD_BRAKET = frozenset(
    {"H", "X", "Y", "Z", "S", "Si", "CNot", "CZ", "Swap", "CY", "I", "V", "Vi"}
)
_CLIFFORD_PENNYLANE = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "SX",
        "CNOT",
        "CY",
        "CZ",
        "SWAP",
        "ISWAP",
        "Identity",
        "Adjoint(S)",
        "Adjoint(SX)",
        # Aliases
        "X",
        "Y",
        "Z",
        "H",
    }
)

# Clifford gate sets (case-normalised)
_CLIFFORD_QISKIT = frozenset(
    {
        "h",
        "s",
        "sdg",
        "x",
        "y",
        "z",
        "cx",
        "cy",
        "cz",
        "swap",
        "id",
        "sx",
        "sxdg",
        "measure",
        "barrier",
        "reset",
    }
)
_CLIFFORD_CIRQ = frozenset(
    {"h", "x", "y", "z", "s", "cnot", "cz", "swap", "i", "measure"}
)


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

    elif module.startswith("braket"):
        # Braket: iterate circuit.instructions, match operator.name case-insensitively
        name_lower = gate_name.lower()
        actual = sum(
            1 for instr in c.instructions if instr.operator.name.lower() == name_lower
        )

    elif module.startswith("pennylane") or hasattr(circuit, "device"):
        # QNode: try to get the tape; if not available, do a dry run first
        name_lower = gate_name.lower()
        tape = None
        try:
            tape = c.tape
        except AttributeError:
            pass
        if tape is None:
            # Execute the circuit with a dry run to populate the tape
            try:
                c()
                tape = c.tape
            except Exception:
                tape = None
        if tape is None:
            raise TypeError(
                "PennyLane QNode tape could not be obtained. "
                "Ensure the QNode is properly constructed."
            )
        actual = sum(
            1 for op in tape.operations if type(op).__name__.lower() == name_lower
        )

    elif module.startswith("pytket"):
        name_lower = gate_name.lower()
        # Try to match by OpType name (case-insensitive)
        actual = sum(
            1 for cmd in c.get_commands() if cmd.op.type.name.lower() == name_lower
        )

    else:
        raise NotImplementedError(
            f"assert_gate_count supports Qiskit, Cirq, Braket, PennyLane, "
            f"and Pytket. Got circuit type: {type(circuit).__qualname__!r}."
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

    if module.startswith("pennylane") or hasattr(circuit, "device"):
        try:
            import pennylane as qml

            specs = qml.specs(c)()
            # Try resources.depth first (newer PennyLane), then fall back to "depth"
            if hasattr(specs, "get"):
                resources = specs.get("resources", None)
                if resources is not None and hasattr(resources, "depth"):
                    return int(resources.depth)
                depth_val = specs.get("depth", None)
                if depth_val is not None:
                    return int(depth_val)
            raise TypeError(
                "Could not extract depth from qml.specs() output. "
                "Upgrade PennyLane to a version that exposes 'resources' or 'depth'."
            )
        except ImportError as exc:
            raise TypeError(
                "pennylane is required for PennyLane circuit depth. "
                "Install it with: pip install pytest-quantum[pennylane]"
            ) from exc

    if module.startswith("pytket"):
        return int(c.depth())

    raise TypeError(
        f"assert_circuit_depth does not support circuit type "
        f"{type(circuit).__qualname__!r}.\n"
        "Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane."
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

    if module.startswith("pytket"):
        return int(c.n_qubits)

    raise TypeError(
        f"assert_circuit_width does not support circuit type "
        f"{type(circuit).__qualname__!r}.\n"
        "Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane."
    )


def assert_gates_in_basis_set(
    circuit: object,
    basis_gates: set[str],
    *,
    case_sensitive: bool = False,
) -> None:
    """Assert every gate in the circuit belongs to the specified basis gate set.

    Useful for verifying that a transpiled circuit only uses a target backend's
    native gate set (e.g. after ``qiskit.transpile`` with ``basis_gates=[...]``).

    Args:
        circuit: Qiskit, Cirq, Braket, or Pytket circuit.
        basis_gates: Set of allowed gate names.
        case_sensitive: If False (default), comparison is case-insensitive.

    Raises:
        AssertionError: Lists every non-basis gate found.
        NotImplementedError: For unsupported frameworks.

    Example::

        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from pytest_quantum import assert_gates_in_basis_set

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        transpiled = transpile(qc, basis_gates=["cx", "u3"])
        assert_gates_in_basis_set(transpiled, {"cx", "u3"})
    """
    module = type(circuit).__module__
    c = cast("Any", circuit)

    basis = {g.lower() for g in basis_gates} if not case_sensitive else set(basis_gates)

    def _normalise(name: str) -> str:
        return name if case_sensitive else name.lower()

    non_basis: list[str] = []

    if module.startswith("qiskit"):
        for instr in c.data:
            gate_name = instr.operation.name
            if _normalise(gate_name) not in basis:
                non_basis.append(gate_name)
    elif module.startswith("cirq"):
        for moment in c:
            for op in moment.operations:
                gate_name = str(op.gate)
                if _normalise(gate_name) not in basis:
                    non_basis.append(gate_name)
    elif module.startswith("braket"):
        for instr in c.instructions:
            gate_name = type(instr.operator).__name__
            if _normalise(gate_name) not in basis:
                non_basis.append(gate_name)
    elif module.startswith("pytket"):
        for cmd in c.get_commands():
            gate_name = cmd.op.type.name
            if _normalise(gate_name) not in basis:
                non_basis.append(gate_name)
    else:
        raise NotImplementedError(
            f"assert_gates_in_basis_set supports Qiskit, Cirq, Braket, Pytket; "
            f"got {module!r}"
        )

    if non_basis:
        unique = sorted(set(non_basis))
        raise AssertionError(
            f"Circuit contains {len(non_basis)} gate(s) not in basis set.\n"
            f"  Non-basis gates found : {unique}\n"
            f"  Allowed basis         : {sorted(basis_gates)}\n"
            f"  Hint: use transpile(circuit, basis_gates=[...]) first."
        )


def assert_circuit_is_clifford(circuit: object) -> None:
    """Assert a circuit uses only Clifford gates (H, S, S†, X, Y, Z, CNOT, CZ, SWAP).

    Clifford circuits are classically efficiently simulable.
    Supported: Qiskit, Cirq.

    Raises:
        AssertionError:      If non-Clifford gates found.
        NotImplementedError: If framework not supported.

    Example::

        def test_is_clifford():
            from qiskit import QuantumCircuit
            from pytest_quantum import assert_circuit_is_clifford

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            assert_circuit_is_clifford(qc)
    """
    module = type(circuit).__module__
    c: Any = circuit

    if module.startswith("qiskit"):
        ops = c.count_ops()
        non_clifford = sorted(g for g in ops if g not in _CLIFFORD_QISKIT)
        if non_clifford:
            raise AssertionError(
                f"Circuit contains non-Clifford gates: {non_clifford}\n"
                f"  Clifford set: "
                f"{sorted(g for g in _CLIFFORD_QISKIT if g not in ('measure', 'barrier', 'reset'))}"
            )
        return

    if module.startswith("cirq"):
        non_clifford = set()
        for moment in c:
            for op in moment.operations:
                name = str(op.gate).lower()
                if name not in _CLIFFORD_CIRQ:
                    non_clifford.add(str(op.gate))
        if non_clifford:
            raise AssertionError(
                f"Circuit contains non-Clifford gates: {sorted(non_clifford)}"
            )
        return

    if module.startswith("braket"):
        non_clifford = [
            type(instr.operator).__name__
            for instr in c.instructions
            if type(instr.operator).__name__ not in _CLIFFORD_BRAKET
        ]
        if non_clifford:
            raise AssertionError(
                f"Circuit contains non-Clifford gates: {sorted(set(non_clifford))}. "
                f"Clifford set: {sorted(_CLIFFORD_BRAKET)}"
            )
        return

    if module.startswith("pennylane") or hasattr(circuit, "device"):
        tape = None
        try:
            tape = c.tape
        except AttributeError:
            pass
        if tape is None:
            try:
                c()
                tape = c.tape
            except Exception:
                pass
        if tape is None:
            raise TypeError("Cannot check Clifford: QNode tape could not be obtained.")
        non_clifford = [
            op.name for op in tape.operations if op.name not in _CLIFFORD_PENNYLANE
        ]
        if non_clifford:
            raise AssertionError(
                f"Circuit contains non-Clifford operations: "
                f"{sorted(set(non_clifford))}. "
                f"Clifford set: {sorted(_CLIFFORD_PENNYLANE)}"
            )
        return

    if module.startswith("pytket"):
        try:
            from pytket.tableau import UnitaryTableau  # type: ignore[import-untyped]

            UnitaryTableau(c)  # raises if circuit contains non-Clifford gates
        except ImportError as exc:
            raise ImportError("pytket is required: pip install pytket") from exc
        except Exception as exc:
            raise AssertionError(f"Circuit contains non-Clifford gates: {exc}") from exc
        return

    raise NotImplementedError(
        f"assert_circuit_is_clifford supports Qiskit and Cirq (and also "
        f"Braket, PennyLane, Pytket). Got: {type(circuit).__qualname__!r}"
    )


def assert_has_diagram(circuit: object, expected: str, *, strict: bool = False) -> None:
    """Assert circuit's text representation contains expected pattern.

    For Qiskit: uses ``circuit.draw('text')``.
    For Cirq:   uses ``str(circuit)`` (``circuit.to_text_diagram()``).

    Args:
        circuit:  Any supported framework circuit.
        expected: Expected string (exact if *strict* is ``True``, substring
                  otherwise).
        strict:   If ``True``, require exact match after stripping leading /
                  trailing whitespace.  If ``False`` (default), just check
                  that *expected* is a substring of the diagram.

    Raises:
        AssertionError:      If diagram doesn't match.
        NotImplementedError: For frameworks without text diagram support.

    Example::

        from qiskit import QuantumCircuit
        from pytest_quantum import assert_has_diagram

        qc = QuantumCircuit(1)
        qc.h(0)
        assert_has_diagram(qc, "H")
    """
    module = type(circuit).__module__
    c: Any = circuit

    if module.startswith("qiskit"):
        diagram = str(c.draw("text"))
    elif module.startswith("cirq"):
        diagram = str(c)
    elif module.startswith("pytket"):
        try:
            diagram = str(c)
        except Exception as exc:
            raise NotImplementedError("pytket diagram not available") from exc
    else:
        raise NotImplementedError(
            f"assert_has_diagram supports Qiskit and Cirq; got {module!r}"
        )

    if strict:
        if diagram.strip() != expected.strip():
            raise AssertionError(
                f"Circuit diagram mismatch.\nExpected:\n{expected}\nGot:\n{diagram}"
            )
    else:
        if expected not in diagram:
            raise AssertionError(
                f"Expected pattern not found in circuit diagram.\n"
                f"Pattern: {expected!r}\n"
                f"Diagram:\n{diagram}"
            )
