"""Integration tests for Stim stabilizer simulator.

These tests require stim to be installed.
"""

from __future__ import annotations

import pytest

stim = pytest.importorskip("stim", reason="stim not installed")

from pytest_quantum import assert_hellinger_close, assert_measurement_distribution

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
