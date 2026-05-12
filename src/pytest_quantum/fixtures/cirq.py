"""Cirq fixtures for pytest-quantum."""

from __future__ import annotations

import pytest

from pytest_quantum.plugin import _require


@pytest.fixture(scope="session")
def cirq_simulator() -> object:
    """Session-scoped cirq.Simulator.

    Example::

        def test_cirq(cirq_simulator):
            import cirq

            q = cirq.LineQubit.range(1)
            circuit = cirq.Circuit(cirq.H(q[0]))
            sv = cirq_simulator.simulate(circuit).final_state_vector
            assert sv.shape == (2,)
    """
    _require("cirq", "cirq")
    import cirq

    return cirq.Simulator()


@pytest.fixture(scope="session")
def cirq_sampler() -> object:
    """Session-scoped Cirq sampler for shot-based simulation.

    Returns a callable ``run(circuit, repetitions=1024)`` that executes
    a Cirq circuit with measurements and returns a count dict.

    The circuit must contain measurement gates (cirq.measure).

    Example::

        def test_cirq_bell(cirq_sampler):
            import cirq

            q = cirq.LineQubit.range(2)
            circuit = cirq.Circuit(
                cirq.H(q[0]),
                cirq.CNOT(q[0], q[1]),
                cirq.measure(q[0], q[1], key="result"),
            )
            counts = cirq_sampler(circuit, repetitions=2000)
            assert "00" in counts
    """
    _require("cirq", "cirq")
    import cirq
    import numpy as np

    simulator = cirq.Simulator()  # Create once, reuse

    def run(circuit: object, repetitions: int = 1024) -> dict[str, int]:
        result = simulator.run(
            circuit,  # type: ignore[arg-type]
            repetitions=repetitions,
        )
        # Collect all measurement keys and concatenate bits
        all_bits = []
        for key in sorted(result.measurements.keys()):
            all_bits.append(result.measurements[key])
        if not all_bits:
            raise ValueError(
                "Circuit has no measurement gates. Add cirq.measure() to the circuit."
            )
        combined = np.concatenate(all_bits, axis=1)
        counts: dict[str, int] = {}
        for row in combined:
            key = "".join(str(b) for b in row)
            counts[key] = counts.get(key, 0) + 1
        return counts

    return run
