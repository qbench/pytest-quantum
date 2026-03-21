"""Tests for pytest_quantum.assertions.information (pure numpy)."""

import pytest

from pytest_quantum.assertions.information import (
    assert_cross_entropy_below,
    assert_hellinger_close,
    assert_kl_divergence_below,
)

# ---------------------------------------------------------------------------
# assert_hellinger_close
# ---------------------------------------------------------------------------


def test_hellinger_identical_distributions_zero():
    counts = {"00": 500, "11": 500}
    assert_hellinger_close(counts, counts, max_distance=1e-10)


def test_hellinger_close_distributions_pass():
    a = {"00": 490, "11": 510}
    b = {"00": 500, "11": 500}
    assert_hellinger_close(a, b, max_distance=0.05)


def test_hellinger_disjoint_fails():
    a = {"00": 1000}
    b = {"11": 1000}
    with pytest.raises(AssertionError, match="Hellinger"):
        assert_hellinger_close(a, b, max_distance=0.5)


def test_hellinger_disjoint_is_1():
    # H=1 for disjoint distributions — verify the value
    a = {"00": 1000}
    b = {"11": 1000}
    with pytest.raises(AssertionError, match=r"1\.0"):
        assert_hellinger_close(a, b, max_distance=0.99)


def test_hellinger_missing_key_treated_as_zero():
    a = {"00": 600, "11": 400}
    b = {"00": 600, "11": 400, "01": 0}
    assert_hellinger_close(a, b, max_distance=1e-10)


def test_hellinger_three_outcomes():
    a = {"00": 333, "01": 333, "11": 334}
    b = {"00": 340, "01": 330, "11": 330}
    assert_hellinger_close(a, b, max_distance=0.05)


def test_hellinger_both_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        assert_hellinger_close({}, {})


def test_hellinger_one_empty_vs_nonempty():
    # All probability mass in one dict but not the other → H close to 1
    with pytest.raises(AssertionError, match="Hellinger"):
        assert_hellinger_close({"00": 100}, {}, max_distance=0.5)


# ---------------------------------------------------------------------------
# assert_kl_divergence_below
# ---------------------------------------------------------------------------


def test_kl_identical_distributions_zero():
    counts = {"00": 500, "11": 500}
    probs = {"00": 0.5, "11": 0.5}
    assert_kl_divergence_below(counts, probs, max_kl=1e-8)


def test_kl_close_distributions_pass():
    counts = {"00": 490, "11": 510}
    probs = {"00": 0.5, "11": 0.5}
    assert_kl_divergence_below(counts, probs, max_kl=0.05)


def test_kl_divergent_fails():
    counts = {"00": 900, "11": 100}
    probs = {"00": 0.5, "11": 0.5}
    with pytest.raises(AssertionError, match="KL divergence"):
        assert_kl_divergence_below(counts, probs, max_kl=0.1)


def test_kl_zero_expected_prob_raises():
    counts = {"00": 500, "11": 500}
    probs = {"00": 1.0, "11": 0.0}
    with pytest.raises(ValueError, match="infinite"):
        assert_kl_divergence_below(counts, probs, max_kl=1.0)


def test_kl_empty_counts_passes():
    assert_kl_divergence_below({}, {"00": 0.5, "11": 0.5}, max_kl=0.1)


def test_kl_zero_total_counts_passes():
    assert_kl_divergence_below({"00": 0}, {"00": 0.5}, max_kl=0.1)


def test_kl_bell_distribution():
    counts = {"00": 512, "11": 512}
    probs = {"00": 0.5, "11": 0.5}
    assert_kl_divergence_below(counts, probs, max_kl=0.01)


# ---------------------------------------------------------------------------
# assert_cross_entropy_below
# ---------------------------------------------------------------------------


def test_ce_identical_distributions():
    counts = {"00": 500, "11": 500}
    probs = {"00": 0.5, "11": 0.5}
    # H(P, P) = -sum P(x) log2 P(x) = 1 bit for uniform binary
    assert_cross_entropy_below(counts, probs, max_ce=2.0)


def test_ce_close_distributions_pass():
    counts = {"00": 490, "11": 510}
    probs = {"00": 0.5, "11": 0.5}
    assert_cross_entropy_below(counts, probs, max_ce=2.0)


def test_ce_high_entropy_fails():
    # Highly non-uniform observed vs uniform expected → high CE
    counts = {"00": 1000}
    probs = {"00": 0.1, "11": 0.9}
    # CE = -1.0 * log2(0.1) ≈ 3.32 > max_ce=2.0
    with pytest.raises(AssertionError, match="Cross-entropy"):
        assert_cross_entropy_below(counts, probs, max_ce=2.0)


def test_ce_zero_expected_prob_raises():
    counts = {"00": 500, "11": 500}
    probs = {"00": 1.0, "11": 0.0}
    with pytest.raises(ValueError, match="undefined"):
        assert_cross_entropy_below(counts, probs, max_ce=10.0)


def test_ce_empty_counts_passes():
    assert_cross_entropy_below({}, {"00": 0.5, "11": 0.5}, max_ce=1.0)


def test_ce_four_outcome_distribution():
    counts = {"00": 250, "01": 250, "10": 250, "11": 250}
    probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    # H(P, P) = 2 bits for uniform 4-outcome dist
    assert_cross_entropy_below(counts, probs, max_ce=3.0)
