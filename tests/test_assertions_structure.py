"""Tests for circuit structure assertions."""

from __future__ import annotations

import pytest

from pytest_quantum.assertions.structure import assert_circuit_depth, assert_gate_count


class TestAssertCircuitDepth:
    def test_no_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            assert_circuit_depth("circuit", max_depth=None, min_depth=None)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="does not support"):
            assert_circuit_depth("not_a_circuit", max_depth=5)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("qiskit") is None,
        reason="qiskit not installed",
    )
    def test_qiskit_depth_within_bounds(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        # H + CX = depth 2
        assert_circuit_depth(qc, max_depth=5)
        assert_circuit_depth(qc, min_depth=1)
        assert_circuit_depth(qc, max_depth=5, min_depth=1)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("qiskit") is None,
        reason="qiskit not installed",
    )
    def test_qiskit_depth_too_deep_raises(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        for _ in range(10):
            qc.h(0)  # 10 layers deep
        with pytest.raises(AssertionError, match="exceeds max_depth"):
            assert_circuit_depth(qc, max_depth=5)


class TestAssertGateCount:
    def test_unsupported_framework_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Qiskit, Cirq"):
            assert_gate_count("not_a_circuit", "cx", 1)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("qiskit") is None,
        reason="qiskit not installed",
    )
    def test_correct_count_passes(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        assert_gate_count(qc, "cx", 2)
        assert_gate_count(qc, "h", 1)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("qiskit") is None,
        reason="qiskit not installed",
    )
    def test_wrong_count_raises(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        with pytest.raises(AssertionError, match="Gate count mismatch"):
            assert_gate_count(qc, "cx", 3)

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("qiskit") is None,
        reason="qiskit not installed",
    )
    def test_absent_gate_zero_count(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        assert_gate_count(qc, "t", 0)  # no T gates
