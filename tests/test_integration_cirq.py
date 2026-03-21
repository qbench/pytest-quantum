"""End-to-end integration tests with real Cirq circuits.

Skipped automatically if cirq-core is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("cirq") is None,
    reason="cirq not installed",
)


class TestToUnitaryCirq:
    def test_hadamard_unitary(self) -> None:
        import cirq

        from pytest_quantum.converters.to_unitary import to_unitary

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.H(q[0]))
        U = to_unitary(circuit)

        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        assert U.shape == (2, 2)
        # cirq may use a different qubit ordering convention; check unitarity
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-6)
        # and that it matches H up to global phase
        overlap = abs(np.trace(U.conj().T @ H)) / 2
        assert math.isclose(overlap, 1.0, abs_tol=1e-6)

    def test_cnot_unitary_is_unitary(self) -> None:
        import cirq

        from pytest_quantum.converters.to_unitary import to_unitary

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.CNOT(q[0], q[1]))
        U = to_unitary(circuit)

        assert U.shape == (4, 4)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-6)

    def test_identity_circuit(self) -> None:
        import cirq

        from pytest_quantum.converters.to_unitary import to_unitary

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit()  # empty = identity
        # cirq raises on empty circuit unitary — skip this case
        # Instead test a known-identity decomposition: H H = I
        circuit = cirq.Circuit([cirq.H(q[0]), cirq.H(q[0])])
        U = to_unitary(circuit)
        assert np.allclose(U, np.eye(2), atol=1e-6)


class TestAssertUnitaryCirq:
    def test_hadamard_passes(self) -> None:
        import cirq

        from pytest_quantum import assert_unitary

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.H(q[0]))
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        assert_unitary(circuit, H)

    def test_pauli_x_fails_against_hadamard(self) -> None:
        import cirq

        from pytest_quantum import assert_unitary

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.X(q[0]))
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        with pytest.raises(AssertionError, match="does not implement"):
            assert_unitary(circuit, H)


class TestAssertCircuitsEquivalentCirq:
    def test_equivalent_circuits(self) -> None:
        import cirq

        from pytest_quantum import assert_circuits_equivalent

        q = cirq.LineQubit.range(1)
        # X = H Z H
        qc_x = cirq.Circuit(cirq.X(q[0]))
        qc_hzh = cirq.Circuit([cirq.H(q[0]), cirq.Z(q[0]), cirq.H(q[0])])
        assert_circuits_equivalent(qc_x, qc_hzh)

    def test_non_equivalent_circuits_fail(self) -> None:
        import cirq

        from pytest_quantum import assert_circuits_equivalent

        q = cirq.LineQubit.range(1)
        qc_h = cirq.Circuit(cirq.H(q[0]))
        qc_x = cirq.Circuit(cirq.X(q[0]))
        with pytest.raises(AssertionError, match="NOT equivalent"):
            assert_circuits_equivalent(qc_h, qc_x)


class TestCirqSimulatorFixture:
    def test_fixture_runs_circuit(self, cirq_simulator: object) -> None:
        import cirq

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="m"))
        result = cirq_simulator.run(circuit, repetitions=100)  # type: ignore[union-attr]
        counts = result.measurements["m"].flatten()
        # H on |0> → 50/50; with 100 shots we should see both outcomes
        assert 0 in counts
        assert 1 in counts
