"""Tests for density matrix assertions.

All tests use pure numpy — no quantum SDK required.
"""

from __future__ import annotations

import numpy as np
import pytest

from pytest_quantum.assertions.density import (
    _partial_trace,
    _trace_distance,
    _validate_dm,
    assert_density_matrix_close,
    assert_partial_trace_close,
    assert_purity_above,
    assert_trace_distance_below,
)

# ---------------------------------------------------------------------------
# Helper matrices
# ---------------------------------------------------------------------------

# Pure |0> state
RHO_ZERO = np.array([[1, 0], [0, 0]], dtype=np.complex128)
# Pure |+> state  (H|0>)
RHO_PLUS = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
# Maximally mixed 2x2
RHO_MIXED = np.eye(2, dtype=np.complex128) / 2
# Pure |1> state — orthogonal to |0>
RHO_ONE = np.array([[0, 0], [0, 1]], dtype=np.complex128)
# Bell state |00>+|11>/sqrt(2)  density matrix (4x4)
_bell = np.array([[1], [0], [0], [1]], dtype=np.complex128) / np.sqrt(2)
RHO_BELL = _bell @ _bell.conj().T


# ---------------------------------------------------------------------------
# _validate_dm
# ---------------------------------------------------------------------------


class TestValidateDm:
    def test_valid_matrix_passes(self) -> None:
        _validate_dm(RHO_ZERO, "rho")  # should not raise

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square 2D matrix"):
            _validate_dm(np.zeros((2, 3), dtype=np.complex128), "rho")

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="square 2D matrix"):
            _validate_dm(np.array([1, 0], dtype=np.complex128), "rho")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_dm(np.zeros((0, 0), dtype=np.complex128), "rho")

    def test_zero_trace_raises(self) -> None:
        with pytest.raises(ValueError, match="zero trace"):
            _validate_dm(np.zeros((2, 2), dtype=np.complex128), "rho")


# ---------------------------------------------------------------------------
# assert_density_matrix_close
# ---------------------------------------------------------------------------


class TestAssertDensityMatrixClose:
    def test_identical_matrices_pass(self) -> None:
        assert_density_matrix_close(RHO_ZERO, RHO_ZERO)

    def test_close_matrices_pass(self) -> None:
        rho2 = RHO_ZERO + 1e-8 * np.eye(2, dtype=np.complex128)
        # Both are close after normalisation
        assert_density_matrix_close(RHO_ZERO, rho2, atol=1e-5)

    def test_different_matrices_fail(self) -> None:
        with pytest.raises(AssertionError, match="not close"):
            assert_density_matrix_close(RHO_ZERO, RHO_ONE)

    def test_shape_mismatch_raises_assertion(self) -> None:
        rho4 = np.eye(4, dtype=np.complex128) / 4
        with pytest.raises(AssertionError, match="Shape mismatch"):
            assert_density_matrix_close(RHO_ZERO, rho4)

    def test_non_square_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="square 2D matrix"):
            assert_density_matrix_close(np.zeros((2, 3), dtype=np.complex128), RHO_ZERO)

    def test_zero_trace_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="zero trace"):
            assert_density_matrix_close(np.zeros((2, 2), dtype=np.complex128), RHO_ZERO)

    def test_unnormalized_matrices_compared_after_normalization(self) -> None:
        # 2 * RHO_ZERO should compare equal to RHO_ZERO after normalisation
        assert_density_matrix_close(2 * RHO_ZERO, RHO_ZERO)

    def test_error_message_includes_max_diff_and_trace_distance(self) -> None:
        with pytest.raises(AssertionError) as exc_info:
            assert_density_matrix_close(RHO_ZERO, RHO_ONE)
        msg = str(exc_info.value)
        assert "Max" in msg
        assert "Trace distance" in msg

    def test_mixed_vs_pure_fails(self) -> None:
        with pytest.raises(AssertionError):
            assert_density_matrix_close(RHO_ZERO, RHO_MIXED)


# ---------------------------------------------------------------------------
# assert_trace_distance_below
# ---------------------------------------------------------------------------


class TestAssertTraceDistanceBelow:
    def test_identical_states_distance_zero(self) -> None:
        # T(ρ,ρ) = 0, well below any threshold
        assert_trace_distance_below(RHO_ZERO, RHO_ZERO, max_distance=0.01)

    def test_maximally_mixed_vs_itself(self) -> None:
        assert_trace_distance_below(RHO_MIXED, RHO_MIXED, max_distance=0.0)

    def test_orthogonal_states_exceed_threshold(self) -> None:
        # |0> and |1> are orthogonal — T = 1.0
        with pytest.raises(AssertionError, match="Trace distance"):
            assert_trace_distance_below(RHO_ZERO, RHO_ONE, max_distance=0.01)

    def test_close_threshold_with_small_distance(self) -> None:
        # Pure |0> vs slightly mixed: small distance
        epsilon = 0.01
        rho_eps = (1 - epsilon) * RHO_ZERO + epsilon * RHO_MIXED
        # distance should be small
        assert_trace_distance_below(RHO_ZERO, rho_eps, max_distance=0.02)

    def test_shape_mismatch_raises(self) -> None:
        rho4 = np.eye(4, dtype=np.complex128) / 4
        with pytest.raises(AssertionError, match="Shape mismatch"):
            assert_trace_distance_below(RHO_ZERO, rho4)

    def test_error_message_content(self) -> None:
        with pytest.raises(AssertionError) as exc_info:
            assert_trace_distance_below(RHO_ZERO, RHO_ONE, max_distance=0.01)
        msg = str(exc_info.value)
        assert "T=0" in msg
        assert "T=1" in msg
        assert "Hint" in msg

    def test_threshold_exactly_met_passes(self) -> None:
        # If actual distance is 0.5 and max_distance is 0.5, should pass
        td = _trace_distance(RHO_ZERO, RHO_MIXED)
        assert_trace_distance_below(RHO_ZERO, RHO_MIXED, max_distance=td + 1e-9)


# ---------------------------------------------------------------------------
# assert_purity_above
# ---------------------------------------------------------------------------


class TestAssertPurityAbove:
    def test_pure_state_passes_with_high_threshold(self) -> None:
        # Pure state has purity = 1.0
        assert_purity_above(RHO_ZERO, min_purity=0.999)

    def test_maximally_mixed_fails_high_threshold(self) -> None:
        # Maximally mixed 2x2 has purity = 0.5
        with pytest.raises(AssertionError, match="Purity"):
            assert_purity_above(RHO_MIXED, min_purity=0.95)

    def test_maximally_mixed_passes_low_threshold(self) -> None:
        # Purity of maximally mixed = 1/d = 0.5 for 2x2
        assert_purity_above(RHO_MIXED, min_purity=0.4)

    def test_partial_mixture_threshold(self) -> None:
        # Mix 80% |0> + 20% mixed
        rho = 0.8 * RHO_ZERO + 0.2 * RHO_MIXED
        purity = float(np.real(np.trace(rho @ rho)))
        assert purity > 0.5  # sanity check
        assert_purity_above(rho, min_purity=purity - 0.01)

    def test_error_message_content(self) -> None:
        with pytest.raises(AssertionError) as exc_info:
            assert_purity_above(RHO_MIXED, min_purity=0.95)
        msg = str(exc_info.value)
        assert "Purity" in msg
        assert "Pure state=1.0" in msg
        assert "Hint" in msg

    def test_unnormalised_matrix_handled(self) -> None:
        # 3 * RHO_ZERO is still a valid pure state after normalisation
        assert_purity_above(3 * RHO_ZERO, min_purity=0.99)

    def test_non_square_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="square 2D matrix"):
            assert_purity_above(np.zeros((2, 3), dtype=np.complex128))

    def test_4x4_mixed_state(self) -> None:
        # Maximally mixed 4x4: purity = 1/4 = 0.25
        rho4 = np.eye(4, dtype=np.complex128) / 4
        with pytest.raises(AssertionError, match="4×4"):
            assert_purity_above(rho4, min_purity=0.5)


# ---------------------------------------------------------------------------
# assert_partial_trace_close
# ---------------------------------------------------------------------------


class TestAssertPartialTraceClose:
    def test_bell_state_partial_trace_qubit0(self) -> None:
        """Tracing out qubit 1 from Bell state gives maximally mixed I/2."""
        assert_partial_trace_close(RHO_BELL, keep_qubits=[0], expected=RHO_MIXED)

    def test_bell_state_partial_trace_qubit1(self) -> None:
        """Tracing out qubit 0 from Bell state gives maximally mixed I/2."""
        assert_partial_trace_close(RHO_BELL, keep_qubits=[1], expected=RHO_MIXED)

    def test_product_state_partial_trace(self) -> None:
        """Partial trace of |00><00| keeping qubit 0 gives |0><0|."""
        # |00><00| = tensor product of |0><0| ⊗ |0><0|
        rho_zero_zero = np.kron(RHO_ZERO, RHO_ZERO)
        assert_partial_trace_close(rho_zero_zero, keep_qubits=[0], expected=RHO_ZERO)

    def test_product_state_partial_trace_second_qubit(self) -> None:
        """Partial trace of |0+><0+| keeping qubit 1 gives |+><+|."""
        rho_zero_plus = np.kron(RHO_ZERO, RHO_PLUS)
        assert_partial_trace_close(rho_zero_plus, keep_qubits=[1], expected=RHO_PLUS)

    def test_wrong_expected_fails(self) -> None:
        """Bell state partial trace is not the same as |0><0|."""
        with pytest.raises(AssertionError, match="does not match expected"):
            assert_partial_trace_close(RHO_BELL, keep_qubits=[0], expected=RHO_ZERO)

    def test_non_power_of_2_dimension_raises(self) -> None:
        rho3 = np.eye(3, dtype=np.complex128) / 3
        with pytest.raises(ValueError, match="power of 2"):
            assert_partial_trace_close(rho3, keep_qubits=[0], expected=RHO_ZERO)

    def test_error_message_includes_qubit_info(self) -> None:
        with pytest.raises(AssertionError) as exc_info:
            assert_partial_trace_close(RHO_BELL, keep_qubits=[0], expected=RHO_ZERO)
        assert "keeping qubits" in str(exc_info.value)
        assert "[0]" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _trace_distance helper
# ---------------------------------------------------------------------------


class TestTraceDistance:
    def test_identical_states_zero(self) -> None:
        td = _trace_distance(RHO_ZERO, RHO_ZERO)
        assert abs(td) < 1e-10

    def test_orthogonal_states_one(self) -> None:
        td = _trace_distance(RHO_ZERO, RHO_ONE)
        assert abs(td - 1.0) < 1e-6

    def test_symmetry(self) -> None:
        td_ab = _trace_distance(RHO_ZERO, RHO_MIXED)
        td_ba = _trace_distance(RHO_MIXED, RHO_ZERO)
        assert abs(td_ab - td_ba) < 1e-10

    def test_triangle_inequality(self) -> None:
        td_ac = _trace_distance(RHO_ZERO, RHO_PLUS)
        td_ab = _trace_distance(RHO_ZERO, RHO_MIXED)
        td_bc = _trace_distance(RHO_MIXED, RHO_PLUS)
        assert td_ac <= td_ab + td_bc + 1e-10


# ---------------------------------------------------------------------------
# _partial_trace helper
# ---------------------------------------------------------------------------


class TestPartialTrace:
    def test_2_qubit_trace_out_second(self) -> None:
        """Trace out qubit 1 from |00><00|  →  |0><0|."""
        rho_prod = np.kron(RHO_ZERO, RHO_ONE)
        reduced = _partial_trace(rho_prod, 2, keep=[0])
        assert np.allclose(reduced, RHO_ZERO, atol=1e-10)

    def test_2_qubit_trace_out_first(self) -> None:
        """Trace out qubit 0 from |00><00|  →  |0><0| (second qubit)."""
        rho_prod = np.kron(RHO_ZERO, RHO_ONE)
        reduced = _partial_trace(rho_prod, 2, keep=[1])
        assert np.allclose(reduced, RHO_ONE, atol=1e-10)

    def test_full_trace_gives_scalar(self) -> None:
        """Trace out all qubits returns 1x1 matrix with value = trace."""
        reduced = _partial_trace(RHO_ZERO, 1, keep=[])
        assert reduced.shape == (1, 1)
        assert abs(float(reduced[0, 0]) - 1.0) < 1e-10
