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


class TestVqeConverges:
    def test_simple_minimum_found(self) -> None:

        from pytest_quantum import assert_vqe_converges

        # f(x) = (x - 2)^2, minimum at x=2, value=0
        def cost(p: np.ndarray) -> float:
            return float((p[0] - 2.0) ** 2)

        assert_vqe_converges(cost, [0.0], expected_minimum=0.0, atol=0.01)

    def test_energy_decreases(self) -> None:
        import numpy as np

        from pytest_quantum import assert_vqe_converges

        def cost(p: np.ndarray) -> float:
            return float(np.sin(p[0]) + 1.0)

        assert_vqe_converges(cost, [2.0])  # no expected_minimum, just checks decrease

    def test_non_converging_raises(self) -> None:
        from pytest_quantum import assert_vqe_converges

        # Flat cost function — no decrease
        def flat_cost(p: object) -> float:
            return 5.0

        with pytest.raises(AssertionError, match="did not converge"):
            assert_vqe_converges(flat_cost, [0.0], max_iterations=10)

    def test_wrong_minimum_raises(self) -> None:
        from pytest_quantum import assert_vqe_converges

        def cost(p: object) -> float:
            import numpy as np

            return float((np.asarray(p)[0] - 2.0) ** 2)

        with pytest.raises(AssertionError, match="missed expected minimum"):
            assert_vqe_converges(cost, [0.0], expected_minimum=-5.0, atol=0.01)


class TestCostDecreases:
    def test_decreasing_history_passes(self) -> None:
        from pytest_quantum import assert_cost_decreases

        assert_cost_decreases([10.0, 7.0, 4.0, 1.0])

    def test_flat_history_fails(self) -> None:
        from pytest_quantum import assert_cost_decreases

        with pytest.raises(AssertionError, match="did not decrease"):
            assert_cost_decreases([5.0, 5.0, 5.0], min_decrease=0.1)

    def test_min_decrease_enforced(self) -> None:
        from pytest_quantum import assert_cost_decreases

        with pytest.raises(AssertionError):
            assert_cost_decreases([5.0, 4.9], min_decrease=1.0)

    def test_too_few_entries_raises(self) -> None:
        from pytest_quantum import assert_cost_decreases

        with pytest.raises(ValueError, match="at least 2"):
            assert_cost_decreases([5.0])
