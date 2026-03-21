"""Tests for pytest_quantum.random generators."""

from __future__ import annotations

import numpy as np
import pytest

from pytest_quantum import assert_channel_is_cptp
from pytest_quantum.random import (
    depolarizing_kraus,
    random_density_matrix,
    random_kraus_channel,
    random_statevector,
    random_unitary,
)

# ---------------------------------------------------------------------------
# random_statevector
# ---------------------------------------------------------------------------


def test_random_statevector_shape():
    sv = random_statevector(2, seed=0)
    assert sv.shape == (4,)


def test_random_statevector_normalised():
    sv = random_statevector(3, seed=1)
    assert abs(np.linalg.norm(sv) - 1.0) < 1e-12


def test_random_statevector_dtype():
    sv = random_statevector(1, seed=2)
    assert sv.dtype == np.complex128


def test_random_statevector_different_seeds():
    sv_a = random_statevector(2, seed=0)
    sv_b = random_statevector(2, seed=99)
    assert not np.allclose(sv_a, sv_b)


def test_random_statevector_same_seed_reproducible():
    sv_a = random_statevector(2, seed=42)
    sv_b = random_statevector(2, seed=42)
    assert np.allclose(sv_a, sv_b)


# ---------------------------------------------------------------------------
# random_density_matrix
# ---------------------------------------------------------------------------


def test_random_density_matrix_shape():
    rho = random_density_matrix(2, seed=0)
    assert rho.shape == (4, 4)


def test_random_density_matrix_trace_one():
    rho = random_density_matrix(2, seed=1)
    assert abs(np.trace(rho) - 1.0) < 1e-12


def test_random_density_matrix_psd():
    rho = random_density_matrix(2, seed=2)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-10)


def test_random_density_matrix_rank1():
    """rank=1 gives a pure state (purity=1)."""
    rho = random_density_matrix(2, rank=1, seed=3)
    purity = float(np.real(np.trace(rho @ rho)))
    assert abs(purity - 1.0) < 1e-10


def test_random_density_matrix_dtype():
    rho = random_density_matrix(1, seed=4)
    assert rho.dtype == np.complex128


# ---------------------------------------------------------------------------
# random_unitary
# ---------------------------------------------------------------------------


def test_random_unitary_shape():
    U = random_unitary(2, seed=0)
    assert U.shape == (4, 4)


def test_random_unitary_is_unitary():
    U = random_unitary(2, seed=5)
    prod = U @ U.conj().T
    assert np.allclose(prod, np.eye(4), atol=1e-12)


def test_random_unitary_dtype():
    U = random_unitary(1, seed=6)
    assert U.dtype == np.complex128


def test_random_unitary_different_seeds():
    U_a = random_unitary(2, seed=0)
    U_b = random_unitary(2, seed=1)
    assert not np.allclose(U_a, U_b)


# ---------------------------------------------------------------------------
# random_kraus_channel
# ---------------------------------------------------------------------------


def test_random_kraus_channel_count():
    kraus = random_kraus_channel(1, n_kraus=3, seed=0)
    assert len(kraus) == 3


def test_random_kraus_channel_shape():
    kraus = random_kraus_channel(1, n_kraus=4, seed=1)
    for K in kraus:
        assert K.shape == (2, 2)


def test_random_kraus_channel_cptp():
    """Generated Kraus operators must satisfy completeness: ∑ K†K = I."""
    kraus = random_kraus_channel(1, n_kraus=4, seed=2)
    assert_channel_is_cptp(kraus, atol=1e-10)


def test_random_kraus_channel_two_qubit():
    kraus = random_kraus_channel(2, n_kraus=4, seed=3)
    for K in kraus:
        assert K.shape == (4, 4)
    assert_channel_is_cptp(kraus, atol=1e-10)


# ---------------------------------------------------------------------------
# depolarizing_kraus
# ---------------------------------------------------------------------------


def test_depolarizing_kraus_count():
    ops = depolarizing_kraus(1, 0.1)
    assert len(ops) == 4


def test_depolarizing_kraus_cptp():
    ops = depolarizing_kraus(1, 0.1)
    assert_channel_is_cptp(ops)


def test_depolarizing_kraus_zero_error_identity():
    """p=0 gives [I, 0, 0, 0] → effectively the identity channel."""
    ops = depolarizing_kraus(1, 0.0)
    # First operator must be exactly I
    assert np.allclose(ops[0], np.eye(2))
    # Remaining operators have zero norm
    for K in ops[1:]:
        assert np.allclose(K, np.zeros((2, 2)))


def test_depolarizing_kraus_full_depolarization():
    """p=1 is a valid channel (fully depolarizing)."""
    ops = depolarizing_kraus(1, 1.0)
    assert_channel_is_cptp(ops)


def test_depolarizing_kraus_invalid_n_qubits():
    with pytest.raises(ValueError, match="n_qubits=1"):
        depolarizing_kraus(2, 0.1)


def test_depolarizing_kraus_invalid_error_rate():
    with pytest.raises(ValueError, match="error_rate must be in"):
        depolarizing_kraus(1, 1.5)
