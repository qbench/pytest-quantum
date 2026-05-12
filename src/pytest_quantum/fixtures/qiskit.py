"""Qiskit / Aer fixtures for pytest-quantum."""
from __future__ import annotations

import pytest

from pytest_quantum.plugin import _require


@pytest.fixture(scope="session")
def aer_simulator() -> object:
    """Session-scoped AerSimulator for shot-based Qiskit tests.

    Example::

        def test_bell(aer_simulator):
            from qiskit import QuantumCircuit, transpile
            qc = QuantumCircuit(2)
            qc.h(0); qc.cx(0, 1); qc.measure_all()
            counts = aer_simulator.run(transpile(qc, aer_simulator), shots=1024) \\
                .result().get_counts()
            assert "00" in counts
    """
    _require("qiskit_aer", "qiskit")
    from qiskit_aer import AerSimulator

    return AerSimulator()


@pytest.fixture(scope="session")
def aer_statevector_simulator() -> object:
    """Session-scoped AerSimulator configured for statevector simulation."""
    _require("qiskit_aer", "qiskit")
    from qiskit_aer import AerSimulator

    return AerSimulator(method="statevector")


@pytest.fixture
def aer_noise_simulator() -> object:
    """Function-scoped AerSimulator with configurable depolarizing noise.

    Returns a factory ``make_simulator(error_rate)`` — call it with the
    single-qubit depolarizing error probability you want to simulate.

    Scope is *function* (not session) because the noise model is
    parameterised per-test.

    Example::

        def test_noisy_bell(aer_noise_simulator):
            from qiskit import QuantumCircuit, transpile
            from pytest_quantum import assert_measurement_distribution

            sim = aer_noise_simulator(error_rate=0.01)
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()

            # With 1% noise, distribution is still close to Bell — use wider tolerance
            assert_measurement_distribution(
                counts,
                expected_probs={"00": 0.5, "11": 0.5},
                significance=0.001,
            )
    """
    _require("qiskit_aer", "qiskit")

    def make_simulator(error_rate: float = 0.01) -> object:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise_model = NoiseModel()
        # Single-qubit gate error
        single_qubit_error = depolarizing_error(error_rate, 1)
        # Two-qubit gate error (typically ~10x higher)
        two_qubit_error = depolarizing_error(min(error_rate * 10, 1.0), 2)
        noise_model.add_all_qubit_quantum_error(
            single_qubit_error, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "u"]
        )
        noise_model.add_all_qubit_quantum_error(
            two_qubit_error, ["cx", "cz", "cy", "swap"]
        )
        return AerSimulator(noise_model=noise_model)

    return make_simulator


@pytest.fixture(scope="session")
def qiskit_sampler() -> object:
    """Session-scoped Qiskit StatevectorSampler (Qiskit 1.0+ primitives).

    Returns a StatevectorSampler instance. Use with assert_sampler_distribution.

    Example::

        def test_bell_sampler(qiskit_sampler):
            from qiskit.circuit import QuantumCircuit
            from pytest_quantum import assert_sampler_distribution

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            result = qiskit_sampler.run([(qc,)]).result()
            assert_sampler_distribution(result, {"00": 0.5, "11": 0.5})
    """
    _require("qiskit", "qiskit")
    from qiskit.primitives import StatevectorSampler

    return StatevectorSampler()


@pytest.fixture(scope="session")
def qiskit_estimator() -> object:
    """Session-scoped Qiskit StatevectorEstimator (Qiskit 1.0+ primitives).

    Returns a StatevectorEstimator instance. Use with assert_estimator_close.

    Example::

        def test_z_expectation(qiskit_estimator):
            from qiskit.circuit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp
            from pytest_quantum import assert_estimator_close

            qc = QuantumCircuit(1)  # |0> state, <Z> = 1.0
            obs = SparsePauliOp("Z")
            result = qiskit_estimator.run([(qc, obs)]).result()
            assert_estimator_close(result, expected=1.0, atol=0.01)
    """
    _require("qiskit", "qiskit")
    from qiskit.primitives import StatevectorEstimator

    return StatevectorEstimator()
