"""Tests for assert_circuit_is_clifford."""

from __future__ import annotations

import importlib.util

import pytest

from pytest_quantum.assertions.structure import assert_circuit_is_clifford

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_CIRQ = importlib.util.find_spec("cirq") is not None

# ---------------------------------------------------------------------------
# Unsupported types
# ---------------------------------------------------------------------------


class TestUnsupportedFramework:
    def test_string_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="Qiskit and Cirq"):
            assert_circuit_is_clifford("not_a_circuit")

    def test_plain_object_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            assert_circuit_is_clifford(object())


# ---------------------------------------------------------------------------
# Qiskit tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
class TestCliffordQiskit:
    def test_h_cx_is_clifford(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert_circuit_is_clifford(qc)  # should not raise

    def test_all_clifford_gates_pass(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.s(0)
        qc.sdg(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.cx(0, 1)
        qc.cz(0, 1)
        qc.swap(0, 1)
        assert_circuit_is_clifford(qc)

    def test_t_gate_fails(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.t(0)
        with pytest.raises(AssertionError, match="non-Clifford"):
            assert_circuit_is_clifford(qc)

    def test_rz_gate_fails(self) -> None:
        import numpy as np
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.rz(np.pi / 4, 0)
        with pytest.raises(AssertionError, match="non-Clifford"):
            assert_circuit_is_clifford(qc)

    def test_tdg_gate_fails(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.tdg(0)
        with pytest.raises(AssertionError, match="non-Clifford"):
            assert_circuit_is_clifford(qc)

    def test_error_message_shows_gate_name(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.t(0)
        with pytest.raises(AssertionError) as exc_info:
            assert_circuit_is_clifford(qc)
        msg = str(exc_info.value)
        assert "t" in msg
        assert "Clifford set" in msg

    def test_empty_circuit_passes(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        assert_circuit_is_clifford(qc)

    def test_measure_barrier_reset_pass(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.barrier()
        qc.measure(0, 0)
        qc.reset(0)
        assert_circuit_is_clifford(qc)


# ---------------------------------------------------------------------------
# Cirq tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CIRQ, reason="cirq not installed")
class TestCliffordCirq:
    def test_h_cnot_is_clifford(self) -> None:
        import cirq

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))
        assert_circuit_is_clifford(circuit)

    def test_all_clifford_gates_pass(self) -> None:
        import cirq

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q[0]),
            cirq.X(q[0]),
            cirq.Y(q[0]),
            cirq.Z(q[0]),
            cirq.S(q[0]),
            cirq.CNOT(q[0], q[1]),
            cirq.CZ(q[0], q[1]),
            cirq.SWAP(q[0], q[1]),
        )
        assert_circuit_is_clifford(circuit)

    def test_t_gate_fails(self) -> None:
        import cirq

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.T(q[0]))
        with pytest.raises(AssertionError, match="non-Clifford"):
            assert_circuit_is_clifford(circuit)

    def test_error_message_shows_gate_name(self) -> None:
        import cirq

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.T(q[0]))
        with pytest.raises(AssertionError) as exc_info:
            assert_circuit_is_clifford(circuit)
        msg = str(exc_info.value)
        assert "non-Clifford" in msg

    def test_empty_circuit_passes(self) -> None:
        import cirq

        circuit = cirq.Circuit()
        assert_circuit_is_clifford(circuit)
