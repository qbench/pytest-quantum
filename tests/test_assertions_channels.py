"""Tests for pytest_quantum.assertions.channels (pure numpy, no SDK required)."""

import numpy as np
import pytest

from pytest_quantum.assertions.channels import (
    assert_channel_is_cptp,
    assert_commutes_with,
    assert_hermitian,
    assert_noise_fidelity_above,
    assert_positive_semidefinite,
    assert_process_fidelity_above,
)

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


# ---------------------------------------------------------------------------
# assert_hermitian
# ---------------------------------------------------------------------------


def test_hermitian_identity_passes():
    assert_hermitian(I2)


def test_hermitian_pauli_x_passes():
    assert_hermitian(X)


def test_hermitian_pauli_y_passes():
    assert_hermitian(Y)


def test_hermitian_pauli_z_passes():
    assert_hermitian(Z)


def test_hermitian_density_matrix_passes():
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert_hermitian(rho)


def test_hermitian_non_hermitian_fails():
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    with pytest.raises(AssertionError, match="Hermitian"):
        assert_hermitian(A)


def test_hermitian_atol_param():
    A = I2 + 1e-9 * np.array([[0, 1], [0, 0]], dtype=complex)
    assert_hermitian(A, atol=1e-7)


def test_hermitian_not_square_raises():
    with pytest.raises(ValueError, match="square"):
        assert_hermitian(np.array([[1, 2, 3]], dtype=complex))


def test_hermitian_4x4_passes():
    A = np.kron(X, Z)
    assert_hermitian(A)


# ---------------------------------------------------------------------------
# assert_positive_semidefinite
# ---------------------------------------------------------------------------


def test_psd_identity_passes():
    assert_positive_semidefinite(I2)


def test_psd_zero_matrix_passes():
    assert_positive_semidefinite(np.zeros((2, 2), dtype=complex))


def test_psd_density_matrix_passes():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    assert_positive_semidefinite(rho)


def test_psd_negative_eigenvalue_fails():
    A = np.array([[1, 2], [2, -3]], dtype=complex)
    with pytest.raises(AssertionError, match="positive semi-definite"):
        assert_positive_semidefinite(A)


def test_psd_non_hermitian_raises():
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    with pytest.raises(AssertionError, match="Hermitian"):
        assert_positive_semidefinite(A)


def test_psd_atol_param():
    # Slightly negative eigenvalue within tolerance should pass
    A = np.array([[1, 0], [0, -1e-9]], dtype=complex)
    assert_positive_semidefinite(A, atol=1e-7)


# ---------------------------------------------------------------------------
# assert_commutes_with
# ---------------------------------------------------------------------------


def test_commutes_identity_with_pauli_x():
    assert_commutes_with(I2, X)


def test_commutes_identity_with_pauli_z():
    assert_commutes_with(I2, Z)


def test_commutes_diagonal_matrices():
    D1 = np.diag([1, 2]).astype(complex)
    D2 = np.diag([3, 4]).astype(complex)
    assert_commutes_with(D1, D2)


def test_not_commutes_pauli_x_z_fails():
    with pytest.raises(AssertionError, match="commute"):
        assert_commutes_with(X, Z)


def test_not_commutes_pauli_x_y_fails():
    with pytest.raises(AssertionError, match="commute"):
        assert_commutes_with(X, Y)


def test_commutes_with_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        assert_commutes_with(I2, np.eye(4, dtype=complex))


# ---------------------------------------------------------------------------
# assert_channel_is_cptp
# ---------------------------------------------------------------------------


def test_cptp_identity_channel_passes():
    # Single Kraus op = I satisfies K†K = I
    assert_channel_is_cptp([I2])


def test_cptp_unitary_channel_passes():
    assert_channel_is_cptp([H])


def test_cptp_depolarizing_channel_passes():
    # Standard depolarizing: K0 = sqrt(1-p)*I, K1..K3 = sqrt(p/3)*Pauli
    p = 0.1
    kraus = [
        np.sqrt(1 - p) * I2,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]
    assert_channel_is_cptp(kraus)


def test_cptp_invalid_kraus_fails():
    # These Kraus ops don't sum to I
    kraus = [np.sqrt(0.5) * I2, np.sqrt(0.8) * X]
    with pytest.raises(AssertionError, match="completeness"):
        assert_channel_is_cptp(kraus)


def test_cptp_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        assert_channel_is_cptp([])


def test_cptp_mismatched_shapes_raises():
    with pytest.raises(ValueError, match="same shape"):
        assert_channel_is_cptp([I2, np.eye(4, dtype=complex)])


# ---------------------------------------------------------------------------
# assert_process_fidelity_above
# ---------------------------------------------------------------------------


def test_process_fidelity_identity_with_itself():
    assert_process_fidelity_above([I2], [I2], threshold=0.99)


def test_process_fidelity_unitary_with_itself():
    assert_process_fidelity_above(H, H, threshold=0.99)


def test_process_fidelity_identity_vs_x_below_threshold():
    with pytest.raises(AssertionError, match="fidelity"):
        assert_process_fidelity_above([I2], [X], threshold=0.5)


def test_process_fidelity_depolarizing_vs_identity():
    p = 0.05
    kraus = [
        np.sqrt(1 - p) * I2,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]
    # Depolarizing channel vs identity: fidelity < 1 for p > 0
    # threshold=0 ensures the call runs and doesn't raise
    assert_process_fidelity_above(kraus, [I2], threshold=0.0)


# ---------------------------------------------------------------------------
# assert_noise_fidelity_above
# ---------------------------------------------------------------------------


def test_noise_fidelity_pure_state_with_itself():
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert_noise_fidelity_above(rho, psi, threshold=0.99)


def test_noise_fidelity_ideal_vs_mixed_state():
    psi = np.array([1, 0], dtype=complex)
    # Mixed state (maximally mixed) should have fidelity 0.5 with |0>
    rho_mixed = np.eye(2, dtype=complex) / 2
    with pytest.raises(AssertionError, match="fidelity"):
        assert_noise_fidelity_above(rho_mixed, psi, threshold=0.9)


def test_noise_fidelity_with_density_matrix_ideal():
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert_noise_fidelity_above(rho, rho, threshold=0.99)


def test_noise_fidelity_bell_state():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(bell, bell.conj())
    assert_noise_fidelity_above(rho, bell, threshold=0.99)
