"""End-to-end integration tests with real Qiskit + Aer circuits.

These tests require qiskit and qiskit-aer to be installed.  They are skipped
automatically on machines where those packages are absent.

They exercise the full stack: real circuit → real simulator → our assertions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("qiskit") is None
    or __import__("importlib").util.find_spec("qiskit_aer") is None,
    reason="qiskit and qiskit-aer not installed",
)


# ---------------------------------------------------------------------------
# to_unitary — real Qiskit circuits
# ---------------------------------------------------------------------------


class TestToUnitaryQiskit:
    def test_hadamard_unitary(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum.converters.to_unitary import to_unitary

        qc = QuantumCircuit(1)
        qc.h(0)
        U = to_unitary(qc)

        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        assert U.shape == (2, 2)
        assert np.allclose(np.abs(U), np.abs(H), atol=1e-6)

    def test_identity_circuit_unitary(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum.converters.to_unitary import to_unitary

        qc = QuantumCircuit(1)
        # empty circuit = identity
        U = to_unitary(qc)
        assert np.allclose(U, np.eye(2), atol=1e-6)

    def test_cnot_unitary(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum.converters.to_unitary import to_unitary

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        U = to_unitary(qc)
        assert U.shape == (4, 4)
        # CNOT is unitary: U†U = I
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-6)

    def test_bell_state_circuit_unitary_shape(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum.converters.to_unitary import to_unitary

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        U = to_unitary(qc)
        assert U.shape == (4, 4)


# ---------------------------------------------------------------------------
# assert_unitary — real Qiskit circuits
# ---------------------------------------------------------------------------


class TestAssertUnitaryQiskit:
    def test_hadamard_passes(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_unitary

        qc = QuantumCircuit(1)
        qc.h(0)
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        assert_unitary(qc, H)

    def test_hadamard_with_global_phase(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_unitary

        qc = QuantumCircuit(1)
        qc.h(0)
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        # Same matrix but multiplied by global phase — should still pass
        assert_unitary(qc, np.exp(1j * 0.7) * H)

    def test_wrong_expected_fails(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_unitary

        qc = QuantumCircuit(1)
        qc.h(0)
        # Pauli X is not the same as Hadamard
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        with pytest.raises(AssertionError, match="does not implement"):
            assert_unitary(qc, X)

    def test_t_gate_unitary(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_unitary

        qc = QuantumCircuit(1)
        qc.t(0)
        T = np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(np.complex128)
        assert_unitary(qc, T)


# ---------------------------------------------------------------------------
# assert_circuits_equivalent — real Qiskit circuits
# ---------------------------------------------------------------------------


class TestAssertCircuitsEquivalentQiskit:
    def test_same_circuit_is_equivalent(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuits_equivalent

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert_circuits_equivalent(qc, qc)

    def test_different_decompositions_equivalent(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuits_equivalent

        # X = H Z H  (Hadamard conjugation identity)
        qc_x = QuantumCircuit(1)
        qc_x.x(0)

        qc_hzh = QuantumCircuit(1)
        qc_hzh.h(0)
        qc_hzh.z(0)
        qc_hzh.h(0)

        assert_circuits_equivalent(qc_x, qc_hzh)

    def test_different_circuits_not_equivalent(self) -> None:
        from qiskit import QuantumCircuit

        from pytest_quantum import assert_circuits_equivalent

        qc_h = QuantumCircuit(1)
        qc_h.h(0)

        qc_x = QuantumCircuit(1)
        qc_x.x(0)

        with pytest.raises(AssertionError, match="NOT equivalent"):
            assert_circuits_equivalent(qc_h, qc_x)


# ---------------------------------------------------------------------------
# assert_measurement_distribution — real AerSimulator shots
# ---------------------------------------------------------------------------


class TestMeasurementDistributionQiskit:
    def test_bell_state_distribution(self, aer_simulator: object) -> None:
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_measurement_distribution

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        qc_t = transpile(qc, aer_simulator)  # type: ignore[arg-type]
        counts = (
            aer_simulator.run(qc_t, shots=4000).result().get_counts()  # type: ignore[union-attr]
        )

        assert_measurement_distribution(
            counts,
            expected_probs={"00": 0.5, "11": 0.5},
            significance=0.01,  # strict: only fails if distribution is genuinely wrong
        )

    def test_all_zero_state(self, aer_simulator: object) -> None:
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_measurement_distribution

        qc = QuantumCircuit(2)
        qc.measure_all()

        qc_t = transpile(qc, aer_simulator)  # type: ignore[arg-type]
        counts = (
            aer_simulator.run(qc_t, shots=1000).result().get_counts()  # type: ignore[union-attr]
        )

        assert_measurement_distribution(counts, expected_probs={"00": 1.0})

    def test_hadamard_uniform_distribution(self, aer_simulator: object) -> None:
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_measurement_distribution

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        qc_t = transpile(qc, aer_simulator)  # type: ignore[arg-type]
        counts = (
            aer_simulator.run(qc_t, shots=2000).result().get_counts()  # type: ignore[union-attr]
        )

        assert_measurement_distribution(
            counts,
            expected_probs={"0": 0.5, "1": 0.5},
            significance=0.01,
        )

    def test_wrong_distribution_fails(self, aer_simulator: object) -> None:
        """Bell state shots claimed to be all-zero: should fail."""
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_measurement_distribution

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        qc_t = transpile(qc, aer_simulator)  # type: ignore[arg-type]
        counts = (
            aer_simulator.run(qc_t, shots=2000).result().get_counts()  # type: ignore[union-attr]
        )

        with pytest.raises(AssertionError, match="chi-square"):
            assert_measurement_distribution(
                counts,
                expected_probs={"00": 1.0},  # claim it's always |00>, it isn't
            )


# ---------------------------------------------------------------------------
# assert_counts_close — real AerSimulator shots
# ---------------------------------------------------------------------------


class TestCountsCloseQiskit:
    def test_same_circuit_twice_is_close(self, aer_simulator: object) -> None:
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_counts_close

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qc_t = transpile(qc, aer_simulator)  # type: ignore[arg-type]

        counts_a = (
            aer_simulator.run(qc_t, shots=2000).result().get_counts()  # type: ignore[union-attr]
        )
        counts_b = (
            aer_simulator.run(qc_t, shots=2000).result().get_counts()  # type: ignore[union-attr]
        )

        # Two independent runs of the same circuit should be within TVD 0.1
        assert_counts_close(counts_a, counts_b, max_tvd=0.1)
