"""End-to-end integration tests with real PennyLane circuits.

Skipped automatically if pennylane is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("pennylane") is None,
    reason="pennylane not installed",
)


class TestPennyLaneFixture:
    def test_fixture_creates_device(self, pennylane_device: object) -> None:
        import pennylane as qml

        dev = pennylane_device(wires=2)  # type: ignore[operator]
        assert dev is not None
        assert len(qml.wires.Wires(dev.wires)) == 2

    def test_fixture_statevector(self, pennylane_device: object) -> None:
        import pennylane as qml

        dev = pennylane_device(wires=1)  # type: ignore[operator]

        @qml.qnode(dev)
        def circuit() -> object:
            qml.Hadamard(0)
            return qml.state()

        state = circuit()
        expected = np.array([1, 1]) / np.sqrt(2)
        assert state.shape == (2,)
        assert math.isclose(
            float(abs(np.vdot(state, expected)) ** 2), 1.0, abs_tol=1e-6
        )

    def test_fixture_bell_state(self, pennylane_device: object) -> None:
        import pennylane as qml

        dev = pennylane_device(wires=2)  # type: ignore[operator]

        @qml.qnode(dev)
        def bell() -> object:
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        state = bell()
        bell_target = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert state.shape == (4,)
        fidelity = float(abs(np.vdot(state, bell_target)) ** 2)
        assert math.isclose(fidelity, 1.0, abs_tol=1e-6)


class TestToUnitaryPennyLane:
    def test_hadamard_unitary(self, pennylane_device: object) -> None:
        import pennylane as qml

        from pytest_quantum.converters.to_unitary import to_unitary

        dev = pennylane_device(wires=1)  # type: ignore[operator]

        @qml.qnode(dev)
        def circuit() -> object:
            qml.Hadamard(0)
            return qml.state()

        U = to_unitary(circuit)
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        assert U.shape == (2, 2)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-6)
        # Check it matches H up to global phase
        overlap = abs(np.trace(U.conj().T @ H)) / 2
        assert math.isclose(float(overlap), 1.0, abs_tol=1e-6)

    def test_cnot_unitary_shape(self, pennylane_device: object) -> None:
        import pennylane as qml

        from pytest_quantum.converters.to_unitary import to_unitary

        dev = pennylane_device(wires=2)  # type: ignore[operator]

        @qml.qnode(dev)
        def circuit() -> object:
            qml.CNOT(wires=[0, 1])
            return qml.state()

        U = to_unitary(circuit)
        assert U.shape == (4, 4)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-6)


class TestAssertUnitaryPennyLane:
    def test_hadamard_passes(self, pennylane_device: object) -> None:
        import pennylane as qml

        from pytest_quantum import assert_unitary

        dev = pennylane_device(wires=1)  # type: ignore[operator]

        @qml.qnode(dev)
        def circuit() -> object:
            qml.Hadamard(0)
            return qml.state()

        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        assert_unitary(circuit, H)

    def test_wrong_gate_fails(self, pennylane_device: object) -> None:
        import pennylane as qml

        from pytest_quantum import assert_unitary

        dev = pennylane_device(wires=1)  # type: ignore[operator]

        @qml.qnode(dev)
        def circuit() -> object:
            qml.PauliX(0)
            return qml.state()

        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        with pytest.raises(AssertionError, match="does not implement"):
            assert_unitary(circuit, H)


class TestAssertCircuitWidthPennyLane:
    def test_width_correct(self, pennylane_device: object) -> None:
        import pennylane as qml

        from pytest_quantum import assert_circuit_width

        dev = pennylane_device(wires=3)  # type: ignore[operator]

        @qml.qnode(dev)
        def circuit() -> object:
            return qml.state()

        assert_circuit_width(circuit, expected_qubits=3)

    def test_width_mismatch_raises(self, pennylane_device: object) -> None:
        import pennylane as qml

        from pytest_quantum import assert_circuit_width

        dev = pennylane_device(wires=2)  # type: ignore[operator]

        @qml.qnode(dev)
        def circuit() -> object:
            return qml.state()

        with pytest.raises(AssertionError, match="qubit count mismatch"):
            assert_circuit_width(circuit, expected_qubits=3)


class TestAssertMeasurementDistributionPennyLane:
    def test_bell_state_distribution(self, pennylane_device: object) -> None:
        import pennylane as qml

        from pytest_quantum import assert_measurement_distribution

        dev = pennylane_device(wires=2, shots=8000)  # type: ignore[operator]

        @qml.qnode(dev)
        def bell() -> object:
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

        raw = bell()
        # PennyLane returns {"00": N, "11": M} style
        counts = {k: int(v) for k, v in raw.items()}

        assert_measurement_distribution(
            counts,
            expected_probs={"00": 0.5, "11": 0.5},
            significance=0.001,
        )
