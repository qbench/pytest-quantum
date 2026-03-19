"""Integration tests for Stim stabilizer simulator.

These tests require stim to be installed.
"""

from __future__ import annotations

import pytest

stim = pytest.importorskip("stim", reason="stim not installed")

from pytest_quantum import (
    assert_hellinger_close,
    assert_measurement_distribution,
    assert_stabilizer_state,
    assert_stim_detector_error_rate_below,
    assert_stim_logical_error_rate_below,
)

# ---------------------------------------------------------------------------
# stim_sampler fixture
# ---------------------------------------------------------------------------


def test_stim_sampler_returns_dict(stim_sampler):
    c = stim.Circuit("""
        H 0
        CNOT 0 1
        M 0 1
    """)
    counts = stim_sampler(c, shots=1000)
    assert isinstance(counts, dict)
    assert all(isinstance(k, str) for k in counts)
    assert all(isinstance(v, int) for v in counts.values())


def test_stim_sampler_total_shots(stim_sampler):
    c = stim.Circuit("""
        H 0
        M 0
    """)
    counts = stim_sampler(c, shots=500)
    assert sum(counts.values()) == 500


def test_stim_bell_state_distribution(stim_sampler):
    c = stim.Circuit("""
        H 0
        CNOT 0 1
        M 0 1
    """)
    counts = stim_sampler(c, shots=4000)
    # Bell state: should only produce "00" and "11"
    assert set(counts.keys()).issubset({"00", "11"})
    assert "00" in counts
    assert "11" in counts


def test_stim_bell_distribution_close(stim_sampler):
    c = stim.Circuit("""
        H 0
        CNOT 0 1
        M 0 1
    """)
    counts = stim_sampler(c, shots=4000)
    assert_measurement_distribution(
        counts,
        expected_probs={"00": 0.5, "11": 0.5},
        significance=0.001,
    )


def test_stim_hellinger_two_runs(stim_sampler):
    c = stim.Circuit("""
        H 0
        CNOT 0 1
        M 0 1
    """)
    counts_a = stim_sampler(c, shots=2000)
    counts_b = stim_sampler(c, shots=2000)
    # Two runs of the same circuit should be close
    assert_hellinger_close(counts_a, counts_b, max_distance=0.15)


def test_stim_all_zeros_circuit(stim_sampler):
    c = stim.Circuit("""
        M 0 1
    """)
    counts = stim_sampler(c, shots=100)
    assert counts == {"00": 100}


# ---------------------------------------------------------------------------
# New stim assertion tests
# ---------------------------------------------------------------------------


def test_assert_stim_logical_error_rate_below_repetition_code():
    """Repetition code at low noise should have low logical error rate."""
    c = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.001,
    )
    # At p=0.001 and distance 3 the logical error rate should be well below 5%
    assert_stim_logical_error_rate_below(c, max_error_rate=0.05, shots=2000, seed=42)


def test_assert_stim_logical_error_rate_below_no_observables_raises():
    """Circuit without observables should raise ValueError."""
    c = stim.Circuit("""
        H 0
        CNOT 0 1
        M 0 1
    """)
    with pytest.raises(ValueError, match="no observables"):
        assert_stim_logical_error_rate_below(c, max_error_rate=0.1, shots=100)


def test_assert_stim_detector_error_rate_below():
    """Repetition code with low noise should have low detector error rate."""
    c = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.001,
    )
    # Mean detector error rate at p=0.001 should be well below 5%
    assert_stim_detector_error_rate_below(c, max_error_rate=0.05, shots=2000, seed=0)


def test_assert_stabilizer_state_bell():
    """Bell state prepared by TableauSimulator should satisfy +XX and +ZZ."""
    sim = stim.TableauSimulator()
    sim.h(0)
    sim.cnot(0, 1)
    assert_stabilizer_state(sim, ["+XX", "+ZZ"])


def test_assert_stabilizer_state_failure():
    """Checking wrong stabilizers should raise AssertionError."""
    sim = stim.TableauSimulator()
    # |0> is stabilized by +Z, not +X
    with pytest.raises(AssertionError, match="Stabilizer state check failed"):
        assert_stabilizer_state(sim, ["+X"])
