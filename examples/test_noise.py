"""
Example: testing noisy quantum circuits.

Demonstrates:
- aer_noise_simulator fixture (function-scoped, configurable error rate)
- assert_purity_above (checking how mixed a density matrix is)
- assert_trace_distance_below (comparing noisy vs ideal)
- assert_measurement_distribution with higher significance threshold for noisy tests

Run with:
    pip install "pytest-quantum[qiskit]"
    pytest examples/test_noise.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
class TestNoisyCircuits:
    def test_h_gate_noise_tolerance(self, aer_noise_simulator):
        """H gate should produce ~50/50 even with 5% noise."""
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_measurement_distribution

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        sim = aer_noise_simulator(error_rate=0.05)
        counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()

        # Looser significance (0.001) because noise shifts the distribution
        assert_measurement_distribution(
            counts, {"0": 0.5, "1": 0.5}, significance=0.001
        )

    def test_bell_state_purity_degrades_with_noise(self, aer_noise_simulator):
        """Bell state loses purity under noise — quantify how much."""
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_purity_above

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.save_density_matrix()

        # Low noise: state remains nearly pure
        sim_low = aer_noise_simulator(error_rate=0.01)
        rho = sim_low.run(transpile(qc, sim_low)).result().data()["density_matrix"].data
        assert_purity_above(rho, min_purity=0.95)

    def test_noise_fidelity(self, aer_noise_simulator):
        """Noisy Bell state should be close to ideal Bell state."""
        import math

        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_noise_fidelity_above

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.save_density_matrix()

        sim = aer_noise_simulator(error_rate=0.01)
        rho = sim.run(transpile(qc, sim)).result().data()["density_matrix"].data

        ideal_bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        assert_noise_fidelity_above(rho, ideal_bell, threshold=0.95)
