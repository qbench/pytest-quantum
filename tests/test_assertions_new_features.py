"""Tests for new v0.5.0 features: T2, T2*, IRB, mid-circuit measurement, QASM2."""

from __future__ import annotations

import pytest

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")


# ---------------------------------------------------------------------------
# assert_no_mid_circuit_measurement tests
# ---------------------------------------------------------------------------


class TestNoMidCircuitMeasurement:
    def test_terminal_measurements_pass(self):
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_no_mid_circuit_measurement

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        assert_no_mid_circuit_measurement(qc)  # should not raise

    def test_mid_circuit_measurement_fails(self):
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_no_mid_circuit_measurement

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)  # mid-circuit: gate follows below
        qc.x(0)  # gate after measurement — mid-circuit!
        with pytest.raises(AssertionError, match=r"[Mm]id-circuit"):
            assert_no_mid_circuit_measurement(qc)

    def test_empty_circuit_passes(self):
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_no_mid_circuit_measurement

        qc = QuantumCircuit(2)
        assert_no_mid_circuit_measurement(qc)

    def test_only_measurements_passes(self):
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_no_mid_circuit_measurement

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        assert_no_mid_circuit_measurement(qc)

    def test_unsupported_framework_raises(self):
        from pytest_quantum import assert_no_mid_circuit_measurement

        class FakeCircuit:
            pass

        with pytest.raises(NotImplementedError):
            assert_no_mid_circuit_measurement(FakeCircuit())

    def test_cirq_terminal_measurements_pass(self):
        cirq = pytest.importorskip("cirq")
        from pytest_quantum import assert_no_mid_circuit_measurement

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q[0]),
            cirq.CNOT(q[0], q[1]),
            cirq.measure(*q, key="m"),
        )
        assert_no_mid_circuit_measurement(circuit)


# ---------------------------------------------------------------------------
# assert_qasm2_roundtrip tests
# ---------------------------------------------------------------------------


class TestQasm2Roundtrip:
    def test_simple_circuit_passes(self):
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_qasm2_roundtrip

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert_qasm2_roundtrip(qc)

    def test_single_qubit_passes(self):
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_qasm2_roundtrip

        qc = QuantumCircuit(1)
        qc.x(0)
        assert_qasm2_roundtrip(qc)

    def test_non_qiskit_raises(self):
        from pytest_quantum import assert_qasm2_roundtrip

        class FakeCircuit:
            pass

        with pytest.raises(
            NotImplementedError, match="assert_qasm2_roundtrip only supports"
        ):
            assert_qasm2_roundtrip(FakeCircuit())


# ---------------------------------------------------------------------------
# T2/T2* tests (simulator — T1 is always infinite, so we just verify API)
# ---------------------------------------------------------------------------


class TestT2T2Star:
    def test_t2_ideal_backend_passes(self):
        """Ideal simulator has infinite T2; assert_t2_above should pass for any target."""
        from qiskit_aer import AerSimulator

        from pytest_quantum import assert_t2_above

        backend = AerSimulator()
        # With no noise model, T2 appears infinite — use small target
        t2 = assert_t2_above(backend, qubit=0, target_t2_us=0.001, shots=128)
        assert isinstance(t2, float)
        assert t2 >= 0.0

    def test_t2star_ideal_backend_passes(self):
        from qiskit_aer import AerSimulator

        from pytest_quantum import assert_t2star_above

        backend = AerSimulator()
        t2s = assert_t2star_above(backend, qubit=0, target_t2star_us=0.001, shots=128)
        assert isinstance(t2s, float)
        assert t2s >= 0.0


# ---------------------------------------------------------------------------
# Interleaved RB tests
# ---------------------------------------------------------------------------


class TestInterleavedRB:
    def test_x_gate_irb(self):
        """IRB with X gate on ideal simulator should show near-unit fidelity."""
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        from pytest_quantum import assert_interleaved_rb

        backend = AerSimulator()
        x_gate = QuantumCircuit(1)
        x_gate.x(0)

        result = assert_interleaved_rb(
            backend,
            qubit=0,
            gate_name="X",
            gate_circuit=x_gate,
            clifford_lengths=[1, 2, 3],
            num_sequences=3,
            shots=256,
            min_gate_fidelity=0.5,
        )
        assert "fidelity" in result
        assert "p_ref" in result
        assert "p_irb" in result
        assert isinstance(result["fidelity"], float)

    def test_irb_fails_on_impossible_fidelity(self):
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        from pytest_quantum import assert_interleaved_rb

        backend = AerSimulator()
        x_gate = QuantumCircuit(1)
        x_gate.x(0)

        with pytest.raises(AssertionError, match=r"[Ff]idelity"):
            assert_interleaved_rb(
                backend,
                qubit=0,
                gate_name="X",
                gate_circuit=x_gate,
                clifford_lengths=[1, 2],
                num_sequences=2,
                shots=64,
                min_gate_fidelity=1.01,  # impossible
            )


# ---------------------------------------------------------------------------
# multi_backend_runner and benchmark_suite fixtures
# ---------------------------------------------------------------------------


class TestMultiBackendRunner:
    def test_runs_on_qiskit_aer(self, multi_backend_runner):
        def make_bell():
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])  # single register → no spaces in bitstrings
            return qc

        results = multi_backend_runner.run_all(
            make_bell, shots=512, backends=["qiskit_aer"]
        )
        assert "qiskit_aer" in results
        counts = results["qiskit_aer"]
        total = sum(counts.values())
        assert total == 512
        # Bell state: only "00" and "11" should appear
        assert set(counts.keys()) <= {"00", "11"}


class TestBenchmarkSuite:
    def test_records_timing(self, benchmark_suite):
        import time

        with benchmark_suite.record("test_op"):
            time.sleep(0.001)

        assert benchmark_suite.mean_ms("test_op") >= 0.0

    def test_print_summary_no_crash(self, benchmark_suite, capsys):
        import time

        with benchmark_suite.record("op_a"):
            time.sleep(0.0005)
        with benchmark_suite.record("op_a"):
            time.sleep(0.0005)

        benchmark_suite.print_summary()
        out = capsys.readouterr().out
        assert "op_a" in out
