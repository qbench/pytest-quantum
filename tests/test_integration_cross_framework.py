"""Cross-framework integration tests.

Proves that assert_circuits_equivalent actually works across Qiskit and Cirq —
the most common cross-framework comparison users need.

Skipped if either framework is absent.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("qiskit") is None
    or __import__("importlib").util.find_spec("cirq") is None,
    reason="qiskit and cirq both required for cross-framework tests",
)


class TestQiskitVsCirqEquivalence:
    """Cross-framework single-qubit circuit equivalence.

    Note on multi-qubit gates (CNOT, CZ, etc.):
    Qiskit uses little-endian qubit ordering (qubit 0 is the least significant
    bit) while Cirq uses big-endian ordering.  This means the 4x4 unitary
    matrices for cx(0,1) in Qiskit and CNOT(q[0],q[1]) in Cirq differ by a
    qubit-reversal permutation — they represent the same *logical* circuit
    but assert_circuits_equivalent would correctly report them as NOT
    equivalent at the matrix level.  For cross-framework multi-qubit
    comparison, apply a qubit reversal permutation to one circuit first, or
    compare within a single framework.
    """

    def test_hadamard_equivalent(self) -> None:
        import cirq
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuits_equivalent

        qc = QuantumCircuit(1)
        qc.h(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.H(q[0]))

        assert_circuits_equivalent(qc, cc)

    def test_pauli_x_equivalent(self) -> None:
        import cirq
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuits_equivalent

        qc = QuantumCircuit(1)
        qc.x(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.X(q[0]))

        assert_circuits_equivalent(qc, cc)

    def test_pauli_z_equivalent(self) -> None:
        import cirq
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuits_equivalent

        qc = QuantumCircuit(1)
        qc.z(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.Z(q[0]))

        assert_circuits_equivalent(qc, cc)

    def test_different_gates_not_equivalent(self) -> None:
        import cirq
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuits_equivalent

        qc = QuantumCircuit(1)
        qc.h(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.X(q[0]))

        with pytest.raises(AssertionError, match="NOT equivalent"):
            assert_circuits_equivalent(qc, cc)


class TestCrossFrameworkStructure:
    def test_assert_circuit_width_qiskit(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuit_width

        qc = QuantumCircuit(3)
        assert_circuit_width(qc, expected_qubits=3)

    def test_assert_circuit_width_cirq(self) -> None:
        import cirq

        from pytest_quantum import assert_circuit_width

        q = cirq.LineQubit.range(2)
        cc = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))
        assert_circuit_width(cc, expected_qubits=2)

    def test_assert_circuit_width_mismatch_raises(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuit_width

        qc = QuantumCircuit(3)
        with pytest.raises(AssertionError, match="qubit count mismatch"):
            assert_circuit_width(qc, expected_qubits=2)


class TestCrossFrameworkGateCount:
    def test_cirq_gate_count_h(self) -> None:
        """Cirq H gate matches by str(op.gate).lower() == 'h'."""
        import cirq

        from pytest_quantum import assert_gate_count

        q = cirq.LineQubit.range(2)
        cc = cirq.Circuit(cirq.H(q[0]), cirq.H(q[1]), cirq.CNOT(q[0], q[1]))
        assert_gate_count(cc, "h", 2)
        assert_gate_count(cc, "cnot", 1)

    def test_cirq_gate_count_zero(self) -> None:
        import cirq

        from pytest_quantum import assert_gate_count

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.H(q[0]))
        assert_gate_count(cc, "cnot", 0)

    def test_cirq_wrong_count_raises(self) -> None:
        import cirq

        from pytest_quantum import assert_gate_count

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.H(q[0]))
        with pytest.raises(AssertionError, match="Gate count mismatch"):
            assert_gate_count(cc, "h", 2)
