"""Tests for assert_gates_in_basis_set."""

from __future__ import annotations

import importlib.util

import pytest

from pytest_quantum import assert_gates_in_basis_set

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_CIRQ = importlib.util.find_spec("cirq") is not None


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
class TestGatesInBasisSetQiskit:
    def test_valid_basis_passes(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert_gates_in_basis_set(qc, {"h", "cx"})

    def test_missing_gate_fails(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.t(0)  # T gate not in basis
        with pytest.raises(AssertionError, match="t"):
            assert_gates_in_basis_set(qc, {"h", "cx"})

    def test_case_insensitive(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        assert_gates_in_basis_set(qc, {"H"})  # uppercase

    def test_empty_circuit_passes(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        assert_gates_in_basis_set(qc, {"h", "cx"})

    def test_multiple_non_basis_gates_listed(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.t(0)
        qc.s(0)
        with pytest.raises(AssertionError, match="Non-basis gates"):
            assert_gates_in_basis_set(qc, {"h", "cx"})


@pytest.mark.skipif(not HAS_CIRQ, reason="cirq not installed")
class TestGatesInBasisSetCirq:
    def test_valid_basis_passes(self) -> None:
        import cirq

        q = cirq.LineQubit.range(2)
        c = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))
        # Cirq gate str names include the class name
        gate_names = {str(op.gate) for m in c for op in m.operations}
        assert_gates_in_basis_set(c, gate_names)

    def test_missing_gate_fails(self) -> None:
        import cirq

        q = cirq.LineQubit.range(1)
        c = cirq.Circuit(cirq.H(q[0]))
        with pytest.raises(AssertionError):
            assert_gates_in_basis_set(c, {"CNOT"})  # H not in basis


def test_unsupported_framework_raises() -> None:
    with pytest.raises(NotImplementedError):
        assert_gates_in_basis_set("not_a_circuit", {"h"})
