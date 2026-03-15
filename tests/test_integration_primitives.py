"""Integration tests for Qiskit Primitives fixtures."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None,
    reason="qiskit not installed",
)


class TestQiskitSamplerFixture:
    def test_fixture_exists(self, qiskit_sampler: object) -> None:
        assert qiskit_sampler is not None

    def test_bell_distribution(self, qiskit_sampler: object) -> None:
        from qiskit.circuit import QuantumCircuit

        from pytest_quantum import assert_sampler_distribution

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        result = qiskit_sampler.run([(qc,)], shots=4000).result()  # type: ignore[union-attr]
        assert_sampler_distribution(result, {"00": 0.5, "11": 0.5}, significance=0.01)

    def test_uniform_single_qubit(self, qiskit_sampler: object) -> None:
        from qiskit.circuit import QuantumCircuit

        from pytest_quantum import assert_sampler_distribution

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        result = qiskit_sampler.run([(qc,)], shots=2000).result()  # type: ignore[union-attr]
        assert_sampler_distribution(result, {"0": 0.5, "1": 0.5}, significance=0.01)

    def test_deterministic_zero_state(self, qiskit_sampler: object) -> None:
        from qiskit.circuit import QuantumCircuit

        from pytest_quantum import assert_sampler_distribution

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        result = qiskit_sampler.run([(qc,)], shots=1000).result()  # type: ignore[union-attr]
        assert_sampler_distribution(result, {"0": 1.0}, significance=0.05)


class TestQiskitEstimatorFixture:
    def test_fixture_exists(self, qiskit_estimator: object) -> None:
        assert qiskit_estimator is not None

    def test_z_expectation_ground_state(self, qiskit_estimator: object) -> None:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from pytest_quantum import assert_estimator_close

        qc = QuantumCircuit(1)  # |0> state
        obs = SparsePauliOp("Z")
        result = qiskit_estimator.run([(qc, obs)]).result()  # type: ignore[union-attr]
        assert_estimator_close(result, expected=1.0, atol=0.01)

    def test_z_expectation_excited_state(self, qiskit_estimator: object) -> None:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from pytest_quantum import assert_estimator_close

        qc = QuantumCircuit(1)
        qc.x(0)  # |1> state, <Z> = -1.0
        obs = SparsePauliOp("Z")
        result = qiskit_estimator.run([(qc, obs)]).result()  # type: ignore[union-attr]
        assert_estimator_close(result, expected=-1.0, atol=0.01)

    def test_x_expectation_plus_state(self, qiskit_estimator: object) -> None:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from pytest_quantum import (
            assert_estimator_close,
            assert_ground_state_energy_close,
        )

        qc = QuantumCircuit(1)
        qc.h(0)  # |+> state, <X> = 1.0
        obs = SparsePauliOp("X")
        result = qiskit_estimator.run([(qc, obs)]).result()  # type: ignore[union-attr]
        assert_estimator_close(result, expected=1.0, atol=0.01)
        # Also test assert_ground_state_energy_close via plain float
        assert_ground_state_energy_close(1.0, expected_energy=1.0, atol=0.01)

    def test_estimator_wrong_value_fails(self, qiskit_estimator: object) -> None:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from pytest_quantum import assert_estimator_close

        qc = QuantumCircuit(1)  # |0>, <Z>=1.0
        obs = SparsePauliOp("Z")
        result = qiskit_estimator.run([(qc, obs)]).result()  # type: ignore[union-attr]
        with pytest.raises(AssertionError, match="Expectation value"):
            assert_estimator_close(result, expected=-1.0, atol=0.01)

    def test_y_expectation_zero_state(self, qiskit_estimator: object) -> None:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from pytest_quantum import assert_estimator_close

        qc = QuantumCircuit(1)  # |0> state, <Y> = 0
        obs = SparsePauliOp("Y")
        result = qiskit_estimator.run([(qc, obs)]).result()  # type: ignore[union-attr]
        assert_estimator_close(result, expected=0.0, atol=0.01)
