"""Tests for pytest_quantum.stats — pure numpy/scipy, no quantum SDK needed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pytest_quantum.stats.shots import min_shots, recommended_shots
from pytest_quantum.stats.tests import chi_square_test, fidelity, tvd, tvd_from_counts

# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------


class TestFidelity:
    def test_identical_states(self) -> None:
        psi = np.array([1.0, 0.0], dtype=complex)
        assert math.isclose(fidelity(psi, psi), 1.0, abs_tol=1e-12)

    def test_orthogonal_states(self) -> None:
        zero = np.array([1.0, 0.0], dtype=complex)
        one = np.array([0.0, 1.0], dtype=complex)
        assert math.isclose(fidelity(zero, one), 0.0, abs_tol=1e-12)

    def test_plus_vs_zero(self) -> None:
        zero = np.array([1.0, 0.0], dtype=complex)
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        assert math.isclose(fidelity(zero, plus), 0.5, abs_tol=1e-9)

    def test_global_phase_invariant(self) -> None:
        psi = np.array([1.0, 0.0], dtype=complex)
        phi = np.exp(1j * 0.7) * psi  # same state, different global phase
        assert math.isclose(fidelity(psi, phi), 1.0, abs_tol=1e-12)

    def test_unnormalised_inputs(self) -> None:
        # fidelity should normalise internally
        psi = np.array([2.0, 0.0], dtype=complex)
        phi = np.array([3.0, 0.0], dtype=complex)
        assert math.isclose(fidelity(psi, phi), 1.0, abs_tol=1e-12)

    def test_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same size"):
            fidelity(np.array([1, 0]), np.array([1, 0, 0, 0]))

    def test_zero_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="zero-norm"):
            fidelity(np.zeros(2, dtype=complex), np.array([1, 0], dtype=complex))

    def test_2_qubit_bell_state(self) -> None:
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert math.isclose(fidelity(bell, bell), 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# tvd
# ---------------------------------------------------------------------------


class TestTvd:
    def test_identical_distributions(self) -> None:
        p = np.array([0.5, 0.5])
        assert math.isclose(tvd(p, p), 0.0, abs_tol=1e-12)

    def test_disjoint_support(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert math.isclose(tvd(p, q), 1.0, abs_tol=1e-12)

    def test_known_value(self) -> None:
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        assert math.isclose(tvd(p, q), 0.1, abs_tol=1e-12)


class TestTvdFromCounts:
    def test_identical_counts(self) -> None:
        counts = {"00": 500, "11": 500}
        assert math.isclose(tvd_from_counts(counts, counts), 0.0, abs_tol=1e-12)

    def test_different_counts(self) -> None:
        a = {"00": 800, "11": 200}
        b = {"00": 500, "11": 500}
        distance = tvd_from_counts(a, b)
        assert distance > 0.0
        assert distance <= 1.0

    def test_disjoint_counts(self) -> None:
        a = {"00": 1000}
        b = {"11": 1000}
        assert math.isclose(tvd_from_counts(a, b), 1.0, abs_tol=1e-12)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            tvd_from_counts({}, {"00": 1})


# ---------------------------------------------------------------------------
# chi_square_test
# ---------------------------------------------------------------------------


class TestChiSquareTest:
    def test_perfect_match(self) -> None:
        counts = {"00": 500, "11": 500}
        probs = {"00": 0.5, "11": 0.5}
        _, pvalue = chi_square_test(counts, probs)
        assert pvalue == pytest.approx(1.0, abs=1e-6)

    def test_obvious_mismatch(self) -> None:
        # Bell state expected but we got all-zero
        counts = {"00": 1000, "11": 0}
        probs = {"00": 0.5, "11": 0.5}
        _, pvalue = chi_square_test(counts, probs)
        assert pvalue < 1e-10

    def test_numpy_array_inputs(self) -> None:
        f_obs = np.array([490.0, 510.0])
        f_exp = np.array([0.5, 0.5])
        stat, pvalue = chi_square_test(f_obs, f_exp, total_shots=1000)
        assert pvalue > 0.05  # consistent with 50/50

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            chi_square_test(np.array([1.0, 2.0]), np.array([0.5, 0.3, 0.2]))


# ---------------------------------------------------------------------------
# min_shots
# ---------------------------------------------------------------------------


class TestMinShots:
    def test_one_percent_tvd(self) -> None:
        n = min_shots(0.01)
        assert n >= 1000  # should be in the thousands

    def test_five_percent_tvd(self) -> None:
        n = min_shots(0.05)
        assert 200 <= n <= 2000

    def test_ten_percent_tvd(self) -> None:
        n = min_shots(0.10)
        assert 50 <= n <= 500

    def test_stricter_alpha_gives_more_shots(self) -> None:
        n_default = min_shots(0.05)
        n_strict = min_shots(0.05, alpha=0.01)
        assert n_strict > n_default

    def test_higher_power_gives_more_shots(self) -> None:
        n_low = min_shots(0.05, power=0.80)
        n_high = min_shots(0.05, power=0.95)
        assert n_high > n_low

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            min_shots(0.0)
        with pytest.raises(ValueError, match="epsilon"):
            min_shots(1.5)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            min_shots(0.05, alpha=0.0)

    def test_returns_int(self) -> None:
        assert isinstance(min_shots(0.05), int)


# ---------------------------------------------------------------------------
# recommended_shots
# ---------------------------------------------------------------------------


class TestRecommendedShots:
    def test_uniform_distribution(self) -> None:
        n = recommended_shots({"0": 0.5, "1": 0.5})
        assert n == 10  # 5 / 0.5

    def test_rare_outcome_drives_count(self) -> None:
        n = recommended_shots({"00": 0.499, "01": 0.001, "11": 0.5})
        assert n == 5000  # 5 / 0.001

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            recommended_shots({})

    def test_all_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            recommended_shots({"00": 0.0, "11": 0.0})

    def test_returns_int(self) -> None:
        assert isinstance(recommended_shots({"0": 0.5, "1": 0.5}), int)
