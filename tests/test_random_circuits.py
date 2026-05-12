import pytest
import numpy as np


class TestRandomQiskitCircuit:
    def test_creates_valid_circuit(self):
        from pytest_quantum.random import random_qiskit_circuit
        qc = random_qiskit_circuit(3, 5, seed=42)
        from qiskit import QuantumCircuit
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 3

    def test_seed_reproducibility(self):
        from pytest_quantum.random import random_qiskit_circuit
        qc1 = random_qiskit_circuit(2, 10, seed=123)
        qc2 = random_qiskit_circuit(2, 10, seed=123)
        assert str(qc1) == str(qc2)

    def test_custom_gate_set(self):
        from pytest_quantum.random import random_qiskit_circuit
        qc = random_qiskit_circuit(2, 10, gate_set=frozenset({"h", "x"}), seed=42)
        ops = set(qc.count_ops().keys())
        assert ops <= {"h", "x"}

    def test_single_qubit(self):
        from pytest_quantum.random import random_qiskit_circuit
        # With single qubit, two-qubit gates should be skipped gracefully
        qc = random_qiskit_circuit(1, 10, seed=42)
        assert qc.num_qubits == 1


class TestRandomCirqCircuit:
    def test_creates_valid_circuit(self):
        from pytest_quantum.random import random_cirq_circuit
        import cirq
        circuit = random_cirq_circuit(3, 5, seed=42)
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) <= 3

    def test_seed_reproducibility(self):
        from pytest_quantum.random import random_cirq_circuit
        c1 = random_cirq_circuit(2, 10, seed=123)
        c2 = random_cirq_circuit(2, 10, seed=123)
        assert str(c1) == str(c2)


class TestRandomBraketCircuit:
    def test_creates_valid_circuit(self):
        from pytest_quantum.random import random_braket_circuit
        from braket.circuits import Circuit
        circuit = random_braket_circuit(3, 5, seed=42)
        assert isinstance(circuit, Circuit)

    def test_seed_reproducibility(self):
        from pytest_quantum.random import random_braket_circuit
        c1 = random_braket_circuit(2, 10, seed=123)
        c2 = random_braket_circuit(2, 10, seed=123)
        assert str(c1) == str(c2)


class TestRandomPennyLaneCircuit:
    def test_creates_valid_qnode(self):
        from pytest_quantum.random import random_pennylane_circuit
        qnode = random_pennylane_circuit(2, 5, seed=42)
        # Should be callable
        result = qnode()
        assert len(result) == 4  # 2^2 = 4 amplitudes

    def test_seed_reproducibility(self):
        from pytest_quantum.random import random_pennylane_circuit
        q1 = random_pennylane_circuit(2, 10, seed=123)
        q2 = random_pennylane_circuit(2, 10, seed=123)
        r1 = q1()
        r2 = q2()
        np.testing.assert_array_almost_equal(r1, r2)
