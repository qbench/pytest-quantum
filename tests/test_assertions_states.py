"""Tests for statevector assertions."""

from __future__ import annotations

import numpy as np
import pytest

from pytest_quantum.assertions.states import (
    assert_state_fidelity_above,
    assert_states_close,
)

ZERO = np.array([1.0, 0.0], dtype=np.complex128)
ONE = np.array([0.0, 1.0], dtype=np.complex128)
PLUS = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
BELL = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2)


class TestAssertStateFidelityAbove:
    def test_identical_states_passes(self) -> None:
        assert_state_fidelity_above(ZERO, ZERO)

    def test_global_phase_passes(self) -> None:
        phase = np.exp(1j * 1.23)
        assert_state_fidelity_above(phase * ZERO, ZERO)

    def test_orthogonal_states_fail(self) -> None:
        with pytest.raises(AssertionError, match="fidelity too low"):
            assert_state_fidelity_above(ZERO, ONE)

    def test_threshold_respected(self) -> None:
        # |+⟩ vs |0⟩ → fidelity = 0.5, threshold 0.4 should pass
        assert_state_fidelity_above(PLUS, ZERO, threshold=0.4)

    def test_threshold_violation_raises(self) -> None:
        # |+⟩ vs |0⟩ → fidelity = 0.5, threshold 0.6 should fail
        with pytest.raises(AssertionError, match="fidelity too low"):
            assert_state_fidelity_above(PLUS, ZERO, threshold=0.6)

    def test_unnormalised_inputs(self) -> None:
        # Should normalise internally
        assert_state_fidelity_above(3.0 * ZERO, 7.0 * ZERO)

    def test_bell_state(self) -> None:
        assert_state_fidelity_above(BELL, BELL, threshold=0.999)

    def test_failure_message_contains_fidelity(self) -> None:
        with pytest.raises(AssertionError, match=r"0\.\d+"):
            assert_state_fidelity_above(PLUS, ZERO, threshold=0.99)


class TestAssertStatesClose:
    def test_identical_passes(self) -> None:
        assert_states_close(ZERO, ZERO)

    def test_global_phase_passes(self) -> None:
        phase = np.exp(1j * 2.5)
        assert_states_close(phase * PLUS, PLUS)

    def test_different_states_fail(self) -> None:
        with pytest.raises(AssertionError, match="not close"):
            assert_states_close(ZERO, ONE)

    def test_atol_respected(self) -> None:
        noisy = ZERO + np.array([0.0, 1e-8], dtype=np.complex128)
        assert_states_close(noisy, ZERO, atol=1e-6)

    def test_atol_violation_raises(self) -> None:
        very_noisy = ZERO + np.array([0.0, 0.1], dtype=np.complex128)
        with pytest.raises(AssertionError, match="not close"):
            assert_states_close(very_noisy, ZERO, atol=1e-6)

    def test_size_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError, match="size mismatch"):
            assert_states_close(ZERO, BELL)
