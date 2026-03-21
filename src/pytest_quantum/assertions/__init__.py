"""Quantum-aware assertion helpers for pytest.

All functions raise ``AssertionError`` with detailed, human-readable messages
on failure — designed to integrate naturally with pytest's output.
"""

from __future__ import annotations

from pytest_quantum.assertions.distributions import (
    assert_counts_close,
    assert_measurement_distribution,
)
from pytest_quantum.assertions.states import (
    assert_state_fidelity_above,
    assert_states_close,
)
from pytest_quantum.assertions.structure import assert_circuit_depth, assert_gate_count
from pytest_quantum.assertions.unitary import assert_circuits_equivalent, assert_unitary

__all__ = [
    "assert_circuit_depth",
    "assert_circuits_equivalent",
    "assert_counts_close",
    "assert_gate_count",
    "assert_measurement_distribution",
    "assert_state_fidelity_above",
    "assert_states_close",
    "assert_unitary",
]
