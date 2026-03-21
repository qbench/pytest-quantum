"""Tests for pytest_quantum.assertions.entanglement (pure numpy)."""

import numpy as np
import pytest

from pytest_quantum.assertions.entanglement import (
    assert_bloch_sphere_close,
    assert_entanglement_entropy_below,
    assert_schmidt_rank_at_most,
)

# Common states
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
KET_MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)

# 2-qubit states
KET_00 = np.array([1, 0, 0, 0], dtype=complex)
BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # |Phi+>
PROD = np.kron(KET_PLUS, KET_PLUS)  # |+>|+> product state


# ---------------------------------------------------------------------------
# assert_entanglement_entropy_below
# ---------------------------------------------------------------------------


def test_entropy_product_state_is_zero():
    # |00> is a product state — entropy should be 0
    assert_entanglement_entropy_below(KET_00, partition=[0], max_entropy=1e-10)


def test_entropy_product_plus_plus_is_zero():
    assert_entanglement_entropy_below(PROD, partition=[0], max_entropy=1e-10)


def test_entropy_bell_state_is_one_ebit():
    # Bell state has exactly 1 ebit of entanglement
    assert_entanglement_entropy_below(BELL, partition=[0], max_entropy=1.0)


def test_entropy_bell_state_above_zero():
    # Bell state entropy ~1 > 0 — this should FAIL
    with pytest.raises(AssertionError, match="entropy"):
        assert_entanglement_entropy_below(BELL, partition=[0], max_entropy=0.01)


def test_entropy_3qubit_ghz_on_partition():
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = 1 / np.sqrt(2)
    ghz[7] = 1 / np.sqrt(2)
    # Partition qubit 0 — reduced state is maximally mixed on 1 qubit → 1 ebit
    assert_entanglement_entropy_below(ghz, partition=[0], max_entropy=1.0)


def test_entropy_explicit_n_qubits():
    assert_entanglement_entropy_below(
        KET_00, partition=[0], max_entropy=0.01, n_qubits=2
    )


def test_entropy_invalid_length_raises():
    sv = np.array([1, 0, 0], dtype=complex)  # length 3 — not power of 2
    with pytest.raises(ValueError, match="power of 2"):
        assert_entanglement_entropy_below(sv, partition=[0], max_entropy=1.0)


# ---------------------------------------------------------------------------
# assert_bloch_sphere_close
# ---------------------------------------------------------------------------


def test_bloch_ket_0_is_north_pole():
    # |0> → theta=0 (north pole), phi=0
    assert_bloch_sphere_close(KET_0, expected_theta=0.0, expected_phi=0.0)


def test_bloch_ket_1_is_south_pole():
    # |1> → theta=pi (south pole), phi=0
    assert_bloch_sphere_close(KET_1, expected_theta=np.pi, expected_phi=0.0)


def test_bloch_ket_plus_equator():
    # |+> → theta=pi/2, phi=0
    assert_bloch_sphere_close(KET_PLUS, expected_theta=np.pi / 2, expected_phi=0.0)


def test_bloch_ket_minus_equator():
    # |-> → theta=pi/2, phi=pi
    assert_bloch_sphere_close(KET_MINUS, expected_theta=np.pi / 2, expected_phi=np.pi)


def test_bloch_y_plus_state():
    # |i+> = (|0> + i|1>)/sqrt(2) → theta=pi/2, phi=pi/2
    ket_iplus = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    assert_bloch_sphere_close(
        ket_iplus, expected_theta=np.pi / 2, expected_phi=np.pi / 2
    )


def test_bloch_wrong_position_fails():
    with pytest.raises(AssertionError, match="Bloch sphere"):
        assert_bloch_sphere_close(KET_0, expected_theta=np.pi, expected_phi=0.0)


def test_bloch_not_length_2_raises():
    with pytest.raises(ValueError, match="single-qubit"):
        assert_bloch_sphere_close(BELL, expected_theta=0.0, expected_phi=0.0)


def test_bloch_unnormalized_state():
    # Should normalise internally
    sv = 3.0 * KET_0
    assert_bloch_sphere_close(sv, expected_theta=0.0, expected_phi=0.0)


# ---------------------------------------------------------------------------
# assert_schmidt_rank_at_most
# ---------------------------------------------------------------------------


def test_schmidt_rank_product_state_is_1():
    assert_schmidt_rank_at_most(KET_00, partition=[0], max_rank=1)


def test_schmidt_rank_product_plus_plus_is_1():
    assert_schmidt_rank_at_most(PROD, partition=[0], max_rank=1)


def test_schmidt_rank_bell_state_is_2():
    assert_schmidt_rank_at_most(BELL, partition=[0], max_rank=2)


def test_schmidt_rank_bell_state_not_1_fails():
    with pytest.raises(AssertionError, match="Schmidt rank"):
        assert_schmidt_rank_at_most(BELL, partition=[0], max_rank=1)


def test_schmidt_rank_explicit_n_qubits():
    assert_schmidt_rank_at_most(BELL, partition=[0], max_rank=2, n_qubits=2)


def test_schmidt_rank_3qubit_product():
    prod3 = np.kron(KET_PLUS, np.kron(KET_PLUS, KET_PLUS))
    assert_schmidt_rank_at_most(prod3, partition=[0, 1], max_rank=1)


def test_schmidt_rank_invalid_length_raises():
    sv = np.array([1, 0, 0], dtype=complex)
    with pytest.raises(ValueError, match="power of 2"):
        assert_schmidt_rank_at_most(sv, partition=[0], max_rank=1)
