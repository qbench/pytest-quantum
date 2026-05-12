"""Circuit structure assertions.

These assertions check static properties of a circuit — depth, gate counts,
qubit width — without executing it.  Useful for catching regressions in
compiler output or ensuring a circuit meets hardware constraints.
"""

from __future__ import annotations


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
    from pytest_quantum.adapters import get_adapter

    try:
        adapter = get_adapter(circuit)
        gates = adapter.count_gates(circuit)
    except (TypeError, NotImplementedError):
        raise NotImplementedError(
            f"assert_gate_count supports Qiskit, Cirq, Braket, PennyLane, "
            f"and Pytket. Got circuit type: {type(circuit).__qualname__!r}."
        ) from None

    # For case-insensitive matching (Qiskit uses lowercase, others may not):
    name_lower = gate_name.lower()
    actual = 0
    for g, count in gates.items():
        if g.lower() == name_lower:
            actual += count

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
    from pytest_quantum.adapters import get_adapter

    try:
        return get_adapter(circuit).get_depth(circuit)
    except NotImplementedError:
        raise TypeError(
            f"assert_circuit_depth does not support circuit type "
            f"{type(circuit).__qualname__!r}.\n"
            "Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane, Pytket."
        ) from None
    except TypeError:
        raise TypeError(
            f"assert_circuit_depth does not support circuit type "
            f"{type(circuit).__qualname__!r}.\n"
            "Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane, Pytket."
        ) from None


def _get_width(circuit: object) -> int:
    """Extract qubit count from any supported circuit type."""
    from pytest_quantum.adapters import get_adapter

    try:
        return get_adapter(circuit).get_width(circuit)
    except NotImplementedError:
        raise TypeError(
            f"assert_circuit_width does not support circuit type "
            f"{type(circuit).__qualname__!r}.\n"
            "Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane, Pytket."
        ) from None
    except TypeError:
        raise TypeError(
            f"assert_circuit_width does not support circuit type "
            f"{type(circuit).__qualname__!r}.\n"
            "Supported frameworks: Qiskit, Cirq, Amazon Braket, PennyLane, Pytket."
        ) from None


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
    from pytest_quantum.adapters import get_adapter

    basis = {g.lower() for g in basis_gates} if not case_sensitive else set(basis_gates)

    def _normalise(name: str) -> str:
        return name if case_sensitive else name.lower()

    try:
        adapter = get_adapter(circuit)
        gates = adapter.count_gates(circuit)
    except (TypeError, NotImplementedError):
        module = type(circuit).__module__
        raise NotImplementedError(
            f"assert_gates_in_basis_set supports Qiskit, Cirq, Braket, Pytket; "
            f"got {module!r}"
        ) from None

    non_basis: list[str] = []
    for gate, count in gates.items():
        if _normalise(gate) not in basis:
            non_basis.extend([gate] * count)

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
    from pytest_quantum.adapters import get_adapter

    try:
        adapter = get_adapter(circuit)
        is_cliff = adapter.is_clifford(circuit)
    except (TypeError, NotImplementedError):
        raise NotImplementedError(
            f"assert_circuit_is_clifford supports Qiskit and Cirq (and also "
            f"Braket, PennyLane, Pytket). Got: {type(circuit).__qualname__!r}"
        ) from None

    if not is_cliff:
        try:
            all_names = sorted(adapter.gate_names(circuit))
        except NotImplementedError:
            all_names = []
        raise AssertionError(
            f"Circuit contains non-Clifford gates: {all_names}\n"
            f"  Clifford set: consult adapter for {adapter.framework_name}"
        )


def assert_no_mid_circuit_measurement(circuit: object) -> None:
    """Assert a circuit has no mid-circuit measurements (all measurements are terminal).

    Mid-circuit measurements (measurements followed by further gate operations)
    are not supported on all hardware backends.  This assertion verifies that
    all measurements occur after all gate operations — i.e., measurements only
    appear in the final layer.

    Supported frameworks: Qiskit, Cirq.

    Args:
        circuit: A quantum circuit from a supported framework.

    Raises:
        AssertionError:      If mid-circuit measurements are detected.
        NotImplementedError: If framework is not supported.

    Example::

        from qiskit import QuantumCircuit
        from pytest_quantum import assert_no_mid_circuit_measurement

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        assert_no_mid_circuit_measurement(qc)  # passes — measurements are terminal
    """
    from pytest_quantum.adapters import get_adapter

    try:
        adapter = get_adapter(circuit)
        has_mid = adapter.has_mid_circuit_measurement(circuit)
    except (TypeError, NotImplementedError):
        module = type(circuit).__module__
        raise NotImplementedError(
            f"assert_no_mid_circuit_measurement supports Qiskit and Cirq; got {module!r}"
        ) from None

    if has_mid:
        raise AssertionError(
            "Mid-circuit measurements detected.\n"
            "  Hint: move all measurements to the end of the circuit."
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
    from pytest_quantum.adapters import get_adapter

    try:
        adapter = get_adapter(circuit)
        diagram = adapter.get_diagram(circuit)
    except (TypeError, NotImplementedError):
        module = type(circuit).__module__
        raise NotImplementedError(
            f"assert_has_diagram supports Qiskit and Cirq; got {module!r}"
        ) from None

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
