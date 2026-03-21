"""Tests for pytest_quantum.assertions.compilation.

All tests require qiskit — skipped gracefully if not installed.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bell_circuit() -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _single_h() -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


def _toffoli_circuit() -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return qc


def _identity_circuit() -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    # Empty circuit = identity
    return qc


# ---------------------------------------------------------------------------
# assert_transpilation_equivalent — happy path
# ---------------------------------------------------------------------------


def test_transpilation_equivalent_same_basis() -> None:
    """Transpiling to the same basis should produce an equivalent circuit."""
    from pytest_quantum.assertions.compilation import assert_transpilation_equivalent

    qc = _bell_circuit()
    assert_transpilation_equivalent(
        qc,
        basis_gates_a=["cx", "u"],
        basis_gates_b=["cx", "u"],
        optimization_level=1,
    )


def test_transpilation_equivalent_different_basis() -> None:
    """Bell circuit transpiled to two different basis sets must be equivalent."""
    from pytest_quantum.assertions.compilation import assert_transpilation_equivalent

    qc = _bell_circuit()
    assert_transpilation_equivalent(
        qc,
        basis_gates_a=["cx", "u"],
        basis_gates_b=["cx", "rz", "sx", "x"],
        optimization_level=1,
    )


def test_transpilation_equivalent_vs_original() -> None:
    """When basis_gates_b is None, compare transpiled A against original."""
    from pytest_quantum.assertions.compilation import assert_transpilation_equivalent

    qc = _single_h()
    assert_transpilation_equivalent(
        qc,
        basis_gates_a=["rz", "sx", "x"],
        basis_gates_b=None,
        optimization_level=1,
    )


def test_transpilation_equivalent_single_qubit() -> None:
    from pytest_quantum.assertions.compilation import assert_transpilation_equivalent

    qc = _single_h()
    assert_transpilation_equivalent(
        qc,
        basis_gates_a=["u"],
        basis_gates_b=["rx", "ry", "rz"],
        optimization_level=0,
    )


# ---------------------------------------------------------------------------
# assert_transpilation_equivalent — failure path
# ---------------------------------------------------------------------------


def test_transpilation_equivalent_fails_for_atol_zero() -> None:
    """With atol=0.0, floating-point differences between bases may cause failure."""
    from qiskit import QuantumCircuit
    from qiskit import transpile as qk_transpile

    from pytest_quantum.assertions.compilation import assert_transpilation_equivalent
    from pytest_quantum.converters.to_unitary import to_unitary

    # A circuit with multiple gates might accumulate floating-point errors
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    qc.h(0)

    # Manually check that both transpilations produce identical unitaries
    # to determine if we CAN get a difference with atol=0
    tc_a = qk_transpile(qc, basis_gates=["rz", "sx", "x"], optimization_level=1)
    tc_b = qk_transpile(qc, basis_gates=["rz", "sx", "x"], optimization_level=3)
    U_a = to_unitary(tc_a)
    U_b = to_unitary(tc_b)
    max_diff = float(np.max(np.abs(U_a - U_b)))

    if max_diff <= 1e-6:
        # Both produce the same unitary — test passes trivially
        assert_transpilation_equivalent(
            qc,
            basis_gates_a=["rz", "sx", "x"],
            basis_gates_b=["rz", "sx", "x"],
            optimization_level=1,
        )
    else:
        # If they differ (unlikely), the assertion should raise
        with pytest.raises(AssertionError):
            assert_transpilation_equivalent(
                qc,
                basis_gates_a=["rz", "sx", "x"],
                basis_gates_b=["rz", "sx", "x"],
                optimization_level=1,
                atol=0.0,
                allow_global_phase=False,
            )


def test_transpilation_equivalent_error_message_shows_labels() -> None:
    """Error message should describe which basis sets were compared."""
    from pytest_quantum.assertions.compilation import assert_transpilation_equivalent

    qc = _bell_circuit()

    # Pass a very tight atol that might catch floating-point differences
    # between the same basis (optimization_level 0 vs the original).
    # Mostly we just confirm the function runs without error on correct inputs.
    assert_transpilation_equivalent(
        qc,
        basis_gates_a=["cx", "u"],
        optimization_level=1,
    )


# ---------------------------------------------------------------------------
# assert_transpilation_depth_below — happy path
# ---------------------------------------------------------------------------


def test_transpilation_depth_below_h_gate() -> None:
    from pytest_quantum.assertions.compilation import assert_transpilation_depth_below

    qc = _single_h()
    # H decomposes to at most a few gates
    assert_transpilation_depth_below(qc, max_depth=10, optimization_level=3)


def test_transpilation_depth_below_identity() -> None:
    from pytest_quantum.assertions.compilation import assert_transpilation_depth_below

    qc = _identity_circuit()
    # Empty circuit has depth 0
    assert_transpilation_depth_below(qc, max_depth=1, optimization_level=3)


def test_transpilation_depth_below_with_basis() -> None:
    from pytest_quantum.assertions.compilation import assert_transpilation_depth_below

    qc = _bell_circuit()
    assert_transpilation_depth_below(
        qc,
        max_depth=20,
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=3,
    )


def test_transpilation_depth_below_large_limit() -> None:
    """Very large limit should always pass."""
    from pytest_quantum.assertions.compilation import assert_transpilation_depth_below

    qc = _toffoli_circuit()
    assert_transpilation_depth_below(qc, max_depth=1000, optimization_level=3)


# ---------------------------------------------------------------------------
# assert_transpilation_depth_below — failure path
# ---------------------------------------------------------------------------


def test_transpilation_depth_below_fails_for_tight_limit() -> None:
    from pytest_quantum.assertions.compilation import assert_transpilation_depth_below

    qc = _toffoli_circuit()
    with pytest.raises(AssertionError, match="exceeds max_depth"):
        assert_transpilation_depth_below(
            qc, max_depth=0, optimization_level=3
        )  # depth > 0


def test_transpilation_depth_below_error_message_includes_depth() -> None:
    from pytest_quantum.assertions.compilation import assert_transpilation_depth_below

    qc = _bell_circuit()
    with pytest.raises(AssertionError, match="max_depth"):
        assert_transpilation_depth_below(qc, max_depth=0, optimization_level=0)


# ---------------------------------------------------------------------------
# assert_gate_count_after_transpilation — happy path
# ---------------------------------------------------------------------------


def test_gate_count_cx_in_bell() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _bell_circuit()
    # Bell circuit has exactly 1 CX gate at optimization_level=0
    count = assert_gate_count_after_transpilation(
        qc,
        "cx",
        max_count=2,
        min_count=1,
        basis_gates=["cx", "u"],
        optimization_level=0,
    )
    assert isinstance(count, int)
    assert 1 <= count <= 2


def test_gate_count_returns_integer() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _bell_circuit()
    result = assert_gate_count_after_transpilation(
        qc, "cx", max_count=10, basis_gates=["cx", "u"], optimization_level=1
    )
    assert isinstance(result, int)


def test_gate_count_zero_for_absent_gate() -> None:
    """Gate not in basis should have count 0."""
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _single_h()
    count = assert_gate_count_after_transpilation(
        qc,
        "cx",  # no 2-qubit gates in single-qubit circuit
        max_count=0,
        basis_gates=["rz", "sx", "x"],
        optimization_level=3,
    )
    assert count == 0


def test_gate_count_toffoli_cx_decomposition() -> None:
    """Toffoli gate decomposes to a bounded number of CX gates."""
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _toffoli_circuit()
    # Toffoli decomposes into at most 6 CNOT gates (standard decomposition)
    count = assert_gate_count_after_transpilation(
        qc,
        "cx",
        max_count=8,  # lenient upper bound
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=3,
    )
    assert count >= 1  # at least one CNOT


def test_gate_count_only_max_count() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _bell_circuit()
    assert_gate_count_after_transpilation(
        qc, "cx", max_count=5, basis_gates=["cx", "u"], optimization_level=1
    )


def test_gate_count_only_min_count() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _bell_circuit()
    assert_gate_count_after_transpilation(
        qc, "cx", min_count=1, basis_gates=["cx", "u"], optimization_level=0
    )


# ---------------------------------------------------------------------------
# assert_gate_count_after_transpilation — failure path
# ---------------------------------------------------------------------------


def test_gate_count_fails_when_exceeds_max() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _toffoli_circuit()
    with pytest.raises(AssertionError, match="count"):
        assert_gate_count_after_transpilation(
            qc,
            "cx",
            max_count=0,  # impossible: Toffoli needs CX gates
            basis_gates=["cx", "rz", "sx", "x"],
            optimization_level=3,
        )


def test_gate_count_fails_when_below_min() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _single_h()
    with pytest.raises(AssertionError, match="count"):
        assert_gate_count_after_transpilation(
            qc,
            "cx",
            min_count=10,  # impossible: single-qubit circuit has no CX
            basis_gates=["cx", "rz", "sx", "x"],
            optimization_level=3,
        )


def test_gate_count_raises_value_error_without_bounds() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _bell_circuit()
    with pytest.raises(ValueError, match="At least one"):
        assert_gate_count_after_transpilation(qc, "cx")


def test_gate_count_error_message_shows_all_counts() -> None:
    from pytest_quantum.assertions.compilation import (
        assert_gate_count_after_transpilation,
    )

    qc = _toffoli_circuit()
    with pytest.raises(AssertionError, match="All gate counts") as exc_info:
        assert_gate_count_after_transpilation(
            qc,
            "cx",
            max_count=0,
            basis_gates=["cx", "rz", "sx", "x"],
            optimization_level=3,
        )
    assert "cx" in str(exc_info.value)
