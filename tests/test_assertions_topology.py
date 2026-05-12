import pytest
from qiskit import QuantumCircuit
from pytest_quantum.assertions.topology import (
    assert_circuit_respects_topology,
    assert_routing_overhead_below,
)


class TestAssertCircuitRespectsTopology:
    def test_passes_valid_topology(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        coupling = [(0, 1), (1, 2)]
        assert_circuit_respects_topology(qc, coupling)

    def test_fails_invalid_topology(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)  # Not in coupling map
        coupling = [(0, 1), (1, 2)]
        with pytest.raises(AssertionError, match="violates coupling map"):
            assert_circuit_respects_topology(qc, coupling)

    def test_undirected_edges(self):
        qc = QuantumCircuit(2)
        qc.cx(1, 0)  # Reverse direction
        coupling = [(0, 1)]
        assert_circuit_respects_topology(qc, coupling)

    def test_single_qubit_gates_ignored(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(2)
        coupling = [(0, 1)]
        assert_circuit_respects_topology(qc, coupling)


class TestAssertRoutingOverheadBelow:
    def test_passes_low_overhead(self):
        original = QuantumCircuit(2)
        original.cx(0, 1)
        original.h(0)
        routed = QuantumCircuit(2)
        routed.cx(0, 1)
        routed.h(0)
        routed.x(1)  # 1 extra gate = 50% overhead
        assert_routing_overhead_below(original, routed, max_overhead=1.0)

    def test_fails_high_overhead(self):
        original = QuantumCircuit(2)
        original.cx(0, 1)
        routed = QuantumCircuit(2)
        routed.cx(0, 1)
        routed.swap(0, 1)
        routed.cx(0, 1)
        with pytest.raises(AssertionError, match="Routing overhead"):
            assert_routing_overhead_below(original, routed, max_overhead=0.5)

    def test_zero_overhead(self):
        original = QuantumCircuit(2)
        original.cx(0, 1)
        routed = QuantumCircuit(2)
        routed.cx(0, 1)
        assert_routing_overhead_below(original, routed, max_overhead=0.01)

    def test_empty_circuits(self):
        original = QuantumCircuit(2)
        routed = QuantumCircuit(2)
        assert_routing_overhead_below(original, routed, max_overhead=0.5)
