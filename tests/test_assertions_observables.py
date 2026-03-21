"""Tests for expectation value and observable assertions.

All tests use pure Python/numpy — no quantum SDK required.
"""

from __future__ import annotations

import numpy as np
import pytest

from pytest_quantum.assertions.observables import (
    _extract_expectation_value,
    assert_expectation_value_close,
    assert_ground_state_energy_close,
)

# ---------------------------------------------------------------------------
# _extract_expectation_value
# ---------------------------------------------------------------------------


class TestExtractExpectationValue:
    def test_int(self) -> None:
        assert _extract_expectation_value(1) == 1.0

    def test_float(self) -> None:
        assert _extract_expectation_value(-1.5) == -1.5

    def test_numpy_scalar(self) -> None:
        v = np.float64(0.42)
        assert abs(_extract_expectation_value(v) - 0.42) < 1e-12

    def test_numpy_0d_array(self) -> None:
        v = np.array(0.99)
        assert abs(_extract_expectation_value(v) - 0.99) < 1e-12

    def test_numpy_1_element_array(self) -> None:
        v = np.array([0.75])
        assert abs(_extract_expectation_value(v) - 0.75) < 1e-12

    def test_numpy_multi_element_raises(self) -> None:
        with pytest.raises(TypeError, match="scalar"):
            _extract_expectation_value(np.array([0.1, 0.2, 0.3]))

    def test_unrecognised_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot extract"):
            _extract_expectation_value(object())

    def test_string_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            _extract_expectation_value("not_a_number")

    def test_complex_numpy_scalar_real(self) -> None:
        # numpy complex64 scalar — float() should work if imaginary part ~ 0
        v = np.complex128(1.0 + 0j)
        assert abs(_extract_expectation_value(v) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# assert_expectation_value_close
# ---------------------------------------------------------------------------


class TestAssertExpectationValueClose:
    def test_exact_match_passes(self) -> None:
        assert_expectation_value_close(1.0, expected=1.0)

    def test_within_tolerance_passes(self) -> None:
        assert_expectation_value_close(0.95, expected=1.0, atol=0.1)

    def test_at_boundary_passes(self) -> None:
        # diff = atol exactly — should pass (<=)
        assert_expectation_value_close(0.9, expected=1.0, atol=0.1)

    def test_just_outside_tolerance_fails(self) -> None:
        with pytest.raises(AssertionError, match="Expectation value mismatch"):
            assert_expectation_value_close(0.89, expected=1.0, atol=0.1)

    def test_negative_values(self) -> None:
        assert_expectation_value_close(-1.0, expected=-1.0, atol=0.01)

    def test_negative_difference_within_tolerance(self) -> None:
        assert_expectation_value_close(-0.95, expected=-1.0, atol=0.1)

    def test_numpy_input_passes(self) -> None:
        v = np.float64(0.99)
        assert_expectation_value_close(v, expected=1.0, atol=0.05)

    def test_error_message_content(self) -> None:
        with pytest.raises(AssertionError) as exc_info:
            assert_expectation_value_close(0.5, expected=1.0, atol=0.1)
        msg = str(exc_info.value)
        assert "Expected" in msg
        assert "Actual" in msg
        assert "tolerance" in msg
        assert "Hint" in msg

    def test_zero_vs_one_fails(self) -> None:
        with pytest.raises(AssertionError):
            assert_expectation_value_close(0.0, expected=1.0, atol=0.05)

    def test_large_atol_accepts_wide_range(self) -> None:
        assert_expectation_value_close(0.0, expected=1.0, atol=1.5)


# ---------------------------------------------------------------------------
# assert_ground_state_energy_close
# ---------------------------------------------------------------------------


class TestAssertGroundStateEnergyClose:
    def test_exact_energy_passes(self) -> None:
        assert_ground_state_energy_close(-1.8572, expected_energy=-1.8572)

    def test_within_tolerance_passes(self) -> None:
        assert_ground_state_energy_close(-1.85, expected_energy=-1.8572, atol=0.05)

    def test_outside_tolerance_fails(self) -> None:
        with pytest.raises(AssertionError, match="Ground state energy"):
            assert_ground_state_energy_close(-2.0, expected_energy=-1.8572, atol=0.05)

    def test_error_message_content(self) -> None:
        with pytest.raises(AssertionError) as exc_info:
            assert_ground_state_energy_close(-2.0, expected_energy=-1.8572, atol=0.05)
        msg = str(exc_info.value)
        assert "Expected energy" in msg
        assert "Measured energy" in msg
        assert "Hint" in msg

    def test_zero_energy(self) -> None:
        assert_ground_state_energy_close(0.0, expected_energy=0.0, atol=0.01)

    def test_positive_energy_difference(self) -> None:
        assert_ground_state_energy_close(1.0, expected_energy=0.95, atol=0.1)

    def test_numpy_input(self) -> None:
        v = np.float64(-1.8572)
        assert_ground_state_energy_close(v, expected_energy=-1.8572, atol=0.001)

    def test_energy_too_low_fails(self) -> None:
        with pytest.raises(AssertionError):
            assert_ground_state_energy_close(-3.0, expected_energy=-1.8572, atol=0.05)

    def test_energy_too_high_fails(self) -> None:
        with pytest.raises(AssertionError):
            assert_ground_state_energy_close(0.0, expected_energy=-1.8572, atol=0.05)

    def test_default_atol_is_permissive(self) -> None:
        # Default atol=0.1 — small deviations accepted
        assert_ground_state_energy_close(-1.80, expected_energy=-1.8572)
