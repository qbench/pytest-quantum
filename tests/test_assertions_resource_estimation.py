import pytest
import numpy as np
from qiskit import QuantumCircuit
from pytest_quantum.assertions.resource_estimation import (
    assert_t_count_below,
    assert_ancilla_count_below,
    assert_clifford_t_depth_below,
)


class TestAssertTCountBelow:
    def test_passes_when_below(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.t(0)
        assert_t_count_below(qc, 5)

    def test_fails_when_at_limit(self):
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.t(1)
        with pytest.raises(AssertionError, match="T-gate count 2 is not below 2"):
            assert_t_count_below(qc, 2)

    def test_fails_when_above(self):
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.t(1)
        qc.tdg(0)
        with pytest.raises(AssertionError, match="T-gate count 3 is not below 2"):
            assert_t_count_below(qc, 2)

    def test_passes_no_t_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert_t_count_below(qc, 1)

    def test_counts_tdg(self):
        qc = QuantumCircuit(1)
        qc.tdg(0)
        qc.tdg(0)
        with pytest.raises(AssertionError, match="T-gate count 2"):
            assert_t_count_below(qc, 2)


class TestAssertAncillaCountBelow:
    def test_passes_when_below(self):
        qc = QuantumCircuit(4)
        assert_ancilla_count_below(qc, logical_qubits=2, max_ancilla=3)

    def test_fails_when_at_limit(self):
        qc = QuantumCircuit(5)
        with pytest.raises(AssertionError, match="Ancilla count 3"):
            assert_ancilla_count_below(qc, logical_qubits=2, max_ancilla=3)

    def test_passes_no_ancilla(self):
        qc = QuantumCircuit(2)
        assert_ancilla_count_below(qc, logical_qubits=2, max_ancilla=1)


class TestAssertCliffordTDepthBelow:
    def test_passes_when_below(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.t(0)
        assert_clifford_t_depth_below(qc, 5)

    def test_fails_when_at_limit(self):
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.tdg(1)
        with pytest.raises(AssertionError, match="T-depth"):
            assert_clifford_t_depth_below(qc, 2)

    def test_passes_no_t_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert_clifford_t_depth_below(qc, 1)
