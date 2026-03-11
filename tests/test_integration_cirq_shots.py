"""Integration tests for cirq_sampler fixture."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cirq") is None,
    reason="cirq not installed",
)


class TestCirqSampler:
    def test_fixture_is_callable(self, cirq_sampler: object) -> None:
        assert callable(cirq_sampler)

    def test_hadamard_shot_distribution(self, cirq_sampler: object) -> None:
        import cirq

        from pytest_quantum import assert_measurement_distribution

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.H(q[0]), cirq.measure(q[0], key="m"))
        counts = cirq_sampler(circuit, repetitions=2000)  # type: ignore[operator]
        assert "0" in counts
        assert "1" in counts
        assert sum(counts.values()) == 2000
        assert_measurement_distribution(
            counts,
            expected_probs={"0": 0.5, "1": 0.5},
            significance=0.01,
        )

    def test_bell_state_distribution(self, cirq_sampler: object) -> None:
        import cirq

        from pytest_quantum import assert_measurement_distribution

        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q[0]),
            cirq.CNOT(q[0], q[1]),
            cirq.measure(q[0], q[1], key="m"),
        )
        counts = cirq_sampler(circuit, repetitions=4000)  # type: ignore[operator]
        assert sum(counts.values()) == 4000
        assert_measurement_distribution(
            counts,
            expected_probs={"00": 0.5, "11": 0.5},
            significance=0.01,
        )

    def test_no_measurement_raises(self, cirq_sampler: object) -> None:
        import cirq

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.H(q[0]))  # no measurement
        with pytest.raises(ValueError, match="measurement"):
            cirq_sampler(circuit, repetitions=100)  # type: ignore[operator]

    def test_deterministic_zero_state(self, cirq_sampler: object) -> None:
        import cirq

        q = cirq.LineQubit.range(1)
        # No gate — stays in |0>, all measurements should be "0"
        circuit = cirq.Circuit(cirq.measure(q[0], key="m"))
        counts = cirq_sampler(circuit, repetitions=1000)  # type: ignore[operator]
        assert counts == {"0": 1000}

    def test_default_repetitions(self, cirq_sampler: object) -> None:
        import cirq

        q = cirq.LineQubit.range(1)
        circuit = cirq.Circuit(cirq.measure(q[0], key="m"))
        counts = cirq_sampler(circuit)  # type: ignore[operator]
        assert sum(counts.values()) == 1024
