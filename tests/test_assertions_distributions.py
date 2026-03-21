"""Tests for shot-distribution assertions."""

from __future__ import annotations

import pytest

from pytest_quantum.assertions.distributions import (
    assert_counts_close,
    assert_measurement_distribution,
)


class TestAssertMeasurementDistribution:
    def test_perfect_counts_pass(self) -> None:
        # Exactly 50/50 Bell state
        assert_measurement_distribution(
            counts={"00": 500, "11": 500},
            expected_probs={"00": 0.5, "11": 0.5},
        )

    def test_near_counts_pass(self) -> None:
        # Within normal statistical fluctuation
        assert_measurement_distribution(
            counts={"00": 485, "11": 515},
            expected_probs={"00": 0.5, "11": 0.5},
        )

    def test_wrong_distribution_fails(self) -> None:
        # All-zero output is wildly inconsistent with Bell state
        with pytest.raises(AssertionError, match="chi-square"):
            assert_measurement_distribution(
                counts={"00": 1000, "11": 0},
                expected_probs={"00": 0.5, "11": 0.5},
            )

    def test_failure_message_contains_pvalue(self) -> None:
        with pytest.raises(AssertionError, match="p-value"):
            assert_measurement_distribution(
                counts={"00": 1000, "11": 0},
                expected_probs={"00": 0.5, "11": 0.5},
            )

    def test_probs_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            assert_measurement_distribution(
                counts={"00": 500, "11": 500},
                expected_probs={"00": 0.4, "11": 0.4},
            )

    def test_empty_counts_raises(self) -> None:
        with pytest.raises(AssertionError, match="empty"):
            assert_measurement_distribution(
                counts={},
                expected_probs={"00": 0.5, "11": 0.5},
            )

    def test_custom_significance_threshold(self) -> None:
        # At significance=0.001 this slight imbalance should still pass
        assert_measurement_distribution(
            counts={"00": 490, "11": 510},
            expected_probs={"00": 0.5, "11": 0.5},
            significance=0.001,
        )

    def test_unexpected_outcomes_fail(self) -> None:
        # Expected only "00" and "11" but got "01" outcomes
        with pytest.raises(AssertionError):
            assert_measurement_distribution(
                counts={"00": 100, "01": 400, "11": 500},
                expected_probs={"00": 0.5, "11": 0.5},
            )

    def test_low_bucket_warning(self) -> None:
        # 0.001 probability * 100 shots = 0.1 expected count < 5
        with pytest.warns(UserWarning, match="expected count"):
            assert_measurement_distribution(
                counts={"00": 50, "01": 0, "11": 50},
                expected_probs={"00": 0.499, "01": 0.001, "11": 0.5},
                min_expected_per_bucket=5,
            )


class TestAssertCountsClose:
    def test_identical_counts_pass(self) -> None:
        counts = {"00": 500, "11": 500}
        assert_counts_close(counts, counts)

    def test_close_counts_pass(self) -> None:
        a = {"00": 495, "11": 505}
        b = {"00": 505, "11": 495}
        assert_counts_close(a, b, max_tvd=0.05)

    def test_very_different_counts_fail(self) -> None:
        a = {"00": 1000}
        b = {"11": 1000}
        with pytest.raises(AssertionError, match="TVD"):
            assert_counts_close(a, b, max_tvd=0.05)

    def test_custom_max_tvd(self) -> None:
        a = {"00": 800, "11": 200}
        b = {"00": 500, "11": 500}
        # TVD = 0.3, fails at 0.05
        with pytest.raises(AssertionError):
            assert_counts_close(a, b, max_tvd=0.05)
        # Should pass at max_tvd=0.35
        assert_counts_close(a, b, max_tvd=0.35)
