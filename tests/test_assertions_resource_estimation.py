import pytest
from qiskit import QuantumCircuit

from pytest_quantum.assertions.resource_estimation import (
    assert_ancilla_count_below,
    assert_clifford_t_depth_below,
    assert_t_count_below,
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
        # Single T gate → T-depth = 1
        assert_clifford_t_depth_below(qc, 5)

    def test_fails_when_at_limit(self):
        # t(0) and tdg(1) are on different qubits → can be parallelised → T-depth = 1
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.tdg(1)
        with pytest.raises(AssertionError, match="T-depth"):
            assert_clifford_t_depth_below(qc, 1)

    def test_passes_no_t_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert_clifford_t_depth_below(qc, 1)

    def test_sequential_t_on_same_qubit(self):
        """Two T gates on the same qubit → T-depth = 2."""
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.t(0)
        with pytest.raises(AssertionError, match="T-depth 2 is not below 2"):
            assert_clifford_t_depth_below(qc, 2)

    def test_parallel_t_on_different_qubits(self):
        """T gates on different qubits in the same layer → T-depth = 1."""
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.t(1)
        assert_clifford_t_depth_below(qc, 2)

    def test_mixed_layers(self):
        """Interleaved T and Clifford gates creating multiple T-layers."""
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.h(0)
        qc.t(0)
        qc.h(0)
        qc.t(0)
        # Three sequential T gates on q0 (with Cliffords between) → T-depth = 3
        with pytest.raises(AssertionError, match="T-depth 3 is not below 3"):
            assert_clifford_t_depth_below(qc, 3)
