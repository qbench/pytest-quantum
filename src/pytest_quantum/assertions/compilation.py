"""Circuit compilation and transpilation assertions."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def assert_transpilation_equivalent(
    circuit: object,
    basis_gates_a: list[str],
    basis_gates_b: list[str] | None = None,
    *,
    optimization_level: int = 1,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert that a circuit compiled to different basis sets is unitarily equivalent.

    Transpiles the circuit to basis_gates_a (and optionally basis_gates_b),
    then verifies the unitaries match.

    Args:
        circuit:            Qiskit QuantumCircuit to transpile.
        basis_gates_a:      First target basis gate set.
        basis_gates_b:      Second target basis gate set (default: original circuit).
        optimization_level: Qiskit transpile optimization level 0-3 (default 1).
        atol:               Absolute tolerance for unitary comparison (default 1e-6).
        allow_global_phase: If True, ignore global phase differences (default True).

    Raises:
        AssertionError: If transpiled circuits are not equivalent.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum import assert_transpilation_equivalent
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        assert_transpilation_equivalent(
            qc,
            basis_gates_a=["cx", "u"],
            basis_gates_b=["ecr", "rz", "sx", "x"],
        )
    """
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for transpilation assertions. "
            "Install with: pip install pytest-quantum[qiskit]"
        ) from exc

    from pytest_quantum.converters.to_unitary import to_unitary

    circuit_a = transpile(
        circuit,
        basis_gates=basis_gates_a,
        optimization_level=optimization_level,
    )
    U_a = to_unitary(circuit_a)

    if basis_gates_b is None:
        U_b = to_unitary(circuit)
    else:
        circuit_b = transpile(
            circuit,
            basis_gates=basis_gates_b,
            optimization_level=optimization_level,
        )
        U_b = to_unitary(circuit_b)

    label_a = f"basis={basis_gates_a}"
    label_b = "original" if basis_gates_b is None else f"basis={basis_gates_b}"

    from pytest_quantum._internal import _unitaries_equivalent

    if _unitaries_equivalent(U_a, U_b, atol=atol, allow_global_phase=allow_global_phase):
        return

    max_diff = float(np.max(np.abs(U_a - U_b)))
    raise AssertionError(
        f"Transpiled circuits are not equivalent.\n"
        f"  {label_a} vs {label_b}\n"
        f"  Max |U_a - U_b|: {max_diff:.2e}   (tolerance: {atol:.2e})\n"
        f"  Hint: check that both basis sets are universal and that optimization_level "
        f"does not introduce approximations."
    )


def assert_transpilation_depth_below(
    circuit: object,
    max_depth: int,
    basis_gates: list[str] | None = None,
    *,
    optimization_level: int = 3,
) -> None:
    """Assert that a circuit transpiled to a basis set has depth <= max_depth.

    Useful as a regression test to detect when compiler changes silently
    increase circuit depth.

    Args:
        circuit:            Qiskit QuantumCircuit to transpile.
        max_depth:          Maximum allowed depth after transpilation.
        basis_gates:        Target basis gate set (default: no restriction).
        optimization_level: Qiskit transpile optimization level 0-3 (default 3).

    Raises:
        AssertionError: If transpiled depth exceeds max_depth.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum import assert_transpilation_depth_below

        assert_transpilation_depth_below(
            qc, max_depth=5, basis_gates=["cx", "rz", "sx", "x"]
        )
    """
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for transpilation assertions. "
            "Install with: pip install pytest-quantum[qiskit]"
        ) from exc

    transpiled = transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )
    depth = cast("Any", transpiled).depth()

    if depth <= max_depth:
        return

    raise AssertionError(
        f"Transpiled circuit depth {depth} exceeds max_depth {max_depth}.\n"
        f"  Basis gates      : {basis_gates or 'unrestricted'}\n"
        f"  Optimization     : level {optimization_level}\n"
        f"  Hint: increase max_depth or use a higher optimization_level to "
        f"reduce the circuit depth."
    )


def assert_gate_count_after_transpilation(
    circuit: object,
    gate_name: str,
    *,
    max_count: int | None = None,
    min_count: int | None = None,
    basis_gates: list[str] | None = None,
    optimization_level: int = 3,
) -> int:
    """Assert the count of a specific gate after transpilation is within bounds.

    Args:
        circuit:            Quantum circuit to transpile.
        gate_name:          Gate to count (e.g. "cx", "t", "rz").
        max_count:          Maximum allowed count (optional).
        min_count:          Minimum required count (optional).
        basis_gates:        Target basis (optional).
        optimization_level: Qiskit transpile optimization level (default 3).

    Returns:
        The actual gate count after transpilation.

    Raises:
        AssertionError: If count is outside [min_count, max_count].
        ValueError:     If neither max_count nor min_count is provided.

    Example::

        from pytest_quantum import assert_gate_count_after_transpilation

        # Assert Toffoli is decomposed into at most 6 CNOT gates
        assert_gate_count_after_transpilation(
            toffoli_circuit, "cx", max_count=6, basis_gates=["cx", "rz", "sx", "x"]
        )
    """
    if max_count is None and min_count is None:
        raise ValueError("At least one of max_count or min_count must be provided.")

    try:
        from qiskit import transpile
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for transpilation assertions. "
            "Install with: pip install pytest-quantum[qiskit]"
        ) from exc

    transpiled = transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )
    ops = cast("Any", transpiled).count_ops()
    count = int(ops.get(gate_name, 0))

    violations: list[str] = []
    if max_count is not None and count > max_count:
        violations.append(f"count {count} > max_count {max_count}")
    if min_count is not None and count < min_count:
        violations.append(f"count {count} < min_count {min_count}")

    if violations:
        raise AssertionError(
            f"Gate '{gate_name}' count after transpilation: {count}.\n"
            f"  Violations: {', '.join(violations)}\n"
            f"  Basis gates      : {basis_gates or 'unrestricted'}\n"
            f"  Optimization     : level {optimization_level}\n"
            f"  All gate counts  : {dict(ops)}"
        )

    return count
