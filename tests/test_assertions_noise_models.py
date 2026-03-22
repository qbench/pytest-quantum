"""Tests for pytest_quantum.assertions.noise_models (pure numpy, no SDK required)."""

import numpy as np
import pytest

from pytest_quantum.assertions.noise_models import (
    assert_amplitude_damping_channel,
    assert_channel_diamond_norm_below,
    assert_channel_preserves_trace,
    assert_dephasing_channel,
    assert_depolarizing_channel,
    assert_no_leakage,
)

# ---------------------------------------------------------------------------
# Shared Pauli matrices
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# Helpers: build canonical Kraus operators analytically
# ---------------------------------------------------------------------------


def depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Standard single-qubit depolarizing channel Kraus operators."""
    return [
        np.sqrt(1 - p) * I2,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Canonical amplitude damping Kraus operators."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


def dephasing_kraus(p: float) -> list[np.ndarray]:
    """Canonical dephasing (phase damping) Kraus operators."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(p)]], dtype=complex)
    return [K0, K1]


# ---------------------------------------------------------------------------
# assert_depolarizing_channel
# ---------------------------------------------------------------------------


def test_depolarizing_correct_rate_passes():
    p = 0.10
    assert_depolarizing_channel(depolarizing_kraus(p), p)


def test_depolarizing_small_rate_passes():
    p = 0.02
    assert_depolarizing_channel(depolarizing_kraus(p), p)


def test_depolarizing_zero_rate_passes():
    # p=0 is just the identity channel
    assert_depolarizing_channel(depolarizing_kraus(0.0), 0.0, atol=1e-6)


def test_depolarizing_wrong_rate_fails():
    p_actual = 0.10
    p_expected = 0.30
    with pytest.raises(AssertionError, match="Depolarizing error rate mismatch"):
        assert_depolarizing_channel(depolarizing_kraus(p_actual), p_expected)


def test_depolarizing_wrong_rate_shows_values():
    p_actual = 0.05
    p_expected = 0.50
    with pytest.raises(AssertionError, match="Estimated rate"):
        assert_depolarizing_channel(depolarizing_kraus(p_actual), p_expected)


def test_depolarizing_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        assert_depolarizing_channel([], 0.1)


# ---------------------------------------------------------------------------
# assert_amplitude_damping_channel
# ---------------------------------------------------------------------------


def test_amplitude_damping_correct_gamma_passes():
    gamma = 0.20
    assert_amplitude_damping_channel(amplitude_damping_kraus(gamma), gamma)


def test_amplitude_damping_small_gamma_passes():
    gamma = 0.01
    assert_amplitude_damping_channel(amplitude_damping_kraus(gamma), gamma)


def test_amplitude_damping_gamma_zero_passes():
    # gamma=0 => identity-like channel (no decay)
    assert_amplitude_damping_channel(amplitude_damping_kraus(0.0), 0.0, atol=1e-6)


def test_amplitude_damping_gamma_one_passes():
    # gamma=1 => full decay to |0>
    assert_amplitude_damping_channel(amplitude_damping_kraus(1.0), 1.0, atol=1e-6)


def test_amplitude_damping_wrong_gamma_fails():
    gamma_actual = 0.20
    gamma_expected = 0.80
    with pytest.raises(AssertionError, match="Amplitude damping parameter mismatch"):
        assert_amplitude_damping_channel(
            amplitude_damping_kraus(gamma_actual), gamma_expected
        )


def test_amplitude_damping_wrong_gamma_shows_values():
    gamma_actual = 0.10
    gamma_expected = 0.90
    with pytest.raises(AssertionError, match="Estimated gamma"):
        assert_amplitude_damping_channel(
            amplitude_damping_kraus(gamma_actual), gamma_expected
        )


def test_amplitude_damping_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        assert_amplitude_damping_channel([], 0.1)


def test_amplitude_damping_wrong_shape_raises():
    K = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="2x2"):
        assert_amplitude_damping_channel([K], 0.1)


# ---------------------------------------------------------------------------
# assert_dephasing_channel
# ---------------------------------------------------------------------------


def test_dephasing_correct_rate_passes():
    p = 0.15
    assert_dephasing_channel(dephasing_kraus(p), p)


def test_dephasing_small_rate_passes():
    p = 0.05
    assert_dephasing_channel(dephasing_kraus(p), p)


def test_dephasing_zero_rate_passes():
    assert_dephasing_channel(dephasing_kraus(0.0), 0.0, atol=1e-6)


def test_dephasing_wrong_rate_fails():
    p_actual = 0.10
    p_expected = 0.80
    with pytest.raises(AssertionError, match="Dephasing rate mismatch"):
        assert_dephasing_channel(dephasing_kraus(p_actual), p_expected)


def test_dephasing_wrong_rate_shows_values():
    p_actual = 0.05
    p_expected = 0.50
    with pytest.raises(AssertionError, match="Estimated rate"):
        assert_dephasing_channel(dephasing_kraus(p_actual), p_expected)


def test_dephasing_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        assert_dephasing_channel([], 0.1)


# ---------------------------------------------------------------------------
# assert_no_leakage
# ---------------------------------------------------------------------------


def test_no_leakage_pure_computational_state_passes():
    # |0><0| for a qubit — no leakage
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    leakage = assert_no_leakage(rho, computational_subspace_dim=2)
    assert leakage == pytest.approx(0.0, abs=1e-10)


def test_no_leakage_superposition_state_passes():
    # |+><+| — still within computational subspace
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    leakage = assert_no_leakage(rho, computational_subspace_dim=2)
    assert leakage == pytest.approx(0.0, abs=1e-10)


def test_no_leakage_qutrit_with_leakage_fails():
    # A qutrit state with 50 % population in |2> (leakage level)
    rho = np.diag([0.25, 0.25, 0.50]).astype(complex)
    with pytest.raises(AssertionError, match="Leakage exceeds"):
        assert_no_leakage(rho, computational_subspace_dim=2, max_leakage=0.01)


def test_no_leakage_qutrit_small_leakage_passes():
    # 0.5 % leakage into |2> — below 1 % threshold
    rho = np.diag([0.50, 0.495, 0.005]).astype(complex)
    leakage = assert_no_leakage(rho, computational_subspace_dim=2, max_leakage=0.01)
    assert leakage == pytest.approx(0.005, abs=1e-6)


def test_no_leakage_returns_leakage_value():
    # 10 % leakage, high threshold: should pass and return 0.10
    rho = np.diag([0.45, 0.45, 0.10]).astype(complex)
    leakage = assert_no_leakage(rho, computational_subspace_dim=2, max_leakage=0.20)
    assert leakage == pytest.approx(0.10, abs=1e-6)


def test_no_leakage_non_square_matrix_raises():
    rho = np.array([[1, 0, 0]], dtype=complex)
    with pytest.raises(ValueError, match="square"):
        assert_no_leakage(rho, computational_subspace_dim=2)


def test_no_leakage_dim_exceeds_matrix_raises():
    rho = np.eye(2, dtype=complex) / 2
    with pytest.raises(ValueError, match="exceeds"):
        assert_no_leakage(rho, computational_subspace_dim=5)


def test_no_leakage_maximally_mixed_two_qubit_passes():
    # Maximally mixed state on 2-qubit space (dim=4), no leakage
    rho = np.eye(4, dtype=complex) / 4
    leakage = assert_no_leakage(rho, computational_subspace_dim=4)
    assert leakage == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# assert_channel_preserves_trace
# ---------------------------------------------------------------------------


def test_channel_preserves_trace_identity_passes():
    assert_channel_preserves_trace([I2])


def test_channel_preserves_trace_depolarizing_passes():
    p = 0.10
    assert_channel_preserves_trace(depolarizing_kraus(p))


def test_channel_preserves_trace_amplitude_damping_passes():
    gamma = 0.30
    assert_channel_preserves_trace(amplitude_damping_kraus(gamma))


def test_channel_preserves_trace_dephasing_passes():
    p = 0.25
    assert_channel_preserves_trace(dephasing_kraus(p))


def test_channel_preserves_trace_invalid_fails():
    # These do not satisfy the TP condition
    kraus = [np.sqrt(0.6) * I2, np.sqrt(0.8) * X]
    with pytest.raises(AssertionError, match="not trace-preserving"):
        assert_channel_preserves_trace(kraus)


def test_channel_preserves_trace_single_invalid_fails():
    # A non-unitary single operator breaks the TP condition
    K = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    with pytest.raises(AssertionError, match="not trace-preserving"):
        assert_channel_preserves_trace([K])


def test_channel_preserves_trace_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        assert_channel_preserves_trace([])


def test_channel_preserves_trace_non_square_raises():
    K = np.array([[1, 0, 0]], dtype=complex)
    with pytest.raises(ValueError, match="square"):
        assert_channel_preserves_trace([K])


# ---------------------------------------------------------------------------
# assert_channel_diamond_norm_below
# ---------------------------------------------------------------------------


def test_diamond_norm_identical_channels_is_zero():
    p = 0.10
    kraus = depolarizing_kraus(p)
    # Same channel: diamond norm == 0
    assert_channel_diamond_norm_below(kraus, kraus, max_diamond_norm=0.01)


def test_diamond_norm_identity_vs_itself_passes():
    assert_channel_diamond_norm_below([I2], [I2], max_diamond_norm=0.01)


def test_diamond_norm_similar_rates_passes():
    # Two depolarizing channels with similar rates: small diamond norm
    kraus_a = depolarizing_kraus(0.10)
    kraus_b = depolarizing_kraus(0.11)
    # Distance should be small; threshold=0.5 is very generous
    assert_channel_diamond_norm_below(kraus_a, kraus_b, max_diamond_norm=0.5)


def test_diamond_norm_very_different_channels_fails():
    # Identity channel vs X gate (unitary) — maximally distinguishable
    kraus_identity = [I2]
    kraus_x = [X]
    with pytest.raises(AssertionError, match="Diamond-norm distance"):
        # Threshold 0.0 — any nonzero distance should fail
        assert_channel_diamond_norm_below(
            kraus_identity, kraus_x, max_diamond_norm=0.0, atol=1e-6
        )


def test_diamond_norm_depolarizing_vs_identity_large_p_fails():
    # Large depolarizing error vs identity: should exceed tight threshold
    p = 0.50
    kraus_dep = depolarizing_kraus(p)
    kraus_id = [I2]
    with pytest.raises(AssertionError, match="Diamond-norm distance"):
        assert_channel_diamond_norm_below(
            kraus_dep, kraus_id, max_diamond_norm=0.0, atol=1e-6
        )


def test_diamond_norm_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        assert_channel_diamond_norm_below([], [I2], max_diamond_norm=0.5)


def test_diamond_norm_shape_mismatch_raises():
    kraus_2 = [I2]
    kraus_4 = [np.eye(4, dtype=complex)]
    with pytest.raises(ValueError, match="same Kraus operator shape"):
        assert_channel_diamond_norm_below(kraus_2, kraus_4, max_diamond_norm=0.5)
