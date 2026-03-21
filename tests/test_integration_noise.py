"""Integration tests for the noise simulator fixture.

Skipped if qiskit-aer is not installed.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("qiskit_aer") is None,
    reason="qiskit-aer not installed",
)


class TestAerNoiseSimulator:
    def test_fixture_returns_callable(self, aer_noise_simulator: object) -> None:
        assert callable(aer_noise_simulator)

    def test_noise_simulator_runs_circuit(self, aer_noise_simulator: object) -> None:
        from qiskit import QuantumCircuit, transpile

        sim = aer_noise_simulator(error_rate=0.01)  # type: ignore[operator]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        counts = sim.run(transpile(qc, sim), shots=1000).result().get_counts()
        assert sum(counts.values()) == 1000

    def test_noisy_bell_dominant_outcomes_correct(self, aer_noise_simulator: object) -> None:
        """With small noise (1%), Bell state still produces mostly 00 and 11."""
        from qiskit import QuantumCircuit, transpile

        sim = aer_noise_simulator(error_rate=0.01)  # type: ignore[operator]
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        counts = sim.run(transpile(qc, sim), shots=4000).result().get_counts()
        total = sum(counts.values())

        # Dominant outcomes should still be 00 and 11 (combined >85% with 1% noise)
        dominant = counts.get("00", 0) + counts.get("11", 0)
        assert dominant / total > 0.85, (
            f"Expected 00+11 to dominate with 1% noise, got {dominant/total:.2%}. "
            f"Full counts: {counts}"
        )

    def test_high_noise_breaks_distribution(self, aer_noise_simulator: object) -> None:
        """With 50% depolarizing noise, even Bell state breaks down noticeably."""
        from qiskit import QuantumCircuit, transpile

        # Compare ideal vs very noisy
        from qiskit_aer import AerSimulator

        from pytest_quantum import assert_counts_close

        ideal = AerSimulator()
        noisy = aer_noise_simulator(error_rate=0.5)  # type: ignore[operator]

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        counts_ideal = ideal.run(transpile(qc, ideal), shots=2000).result().get_counts()
        counts_noisy = noisy.run(transpile(qc, noisy), shots=2000).result().get_counts()

        # 50% noise should make them clearly different at tight TVD
        with pytest.raises(AssertionError, match="TVD"):
            assert_counts_close(counts_ideal, counts_noisy, max_tvd=0.05)

    def test_zero_noise_is_ideal(self, aer_noise_simulator: object) -> None:
        """error_rate=0 should behave like the ideal simulator."""
        from qiskit import QuantumCircuit, transpile

        from pytest_quantum import assert_measurement_distribution

        sim = aer_noise_simulator(error_rate=0.0)  # type: ignore[operator]
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        counts = sim.run(transpile(qc, sim), shots=4000).result().get_counts()
        assert_measurement_distribution(
            counts,
            expected_probs={"00": 0.5, "11": 0.5},
            significance=0.01,
        )
