"""Tests for the shared internal utilities module.

Verifies ``_unitaries_equivalent``, ``_kraus_to_choi``, ``_is_ibm_backend``,
and ``_backend_name`` without requiring any quantum SDK to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pytest_quantum._internal import (
    _backend_name,
    _is_ibm_backend,
    _kraus_to_choi,
    _unitaries_equivalent,
)


# ---------------------------------------------------------------------------
# _unitaries_equivalent
# ---------------------------------------------------------------------------


class TestUnitariesEquivalent:
    """Tests for :func:`_unitaries_equivalent`."""

    def test_identical_matrices(self) -> None:
        """Identical matrices are equivalent."""
        U = np.eye(2, dtype=np.complex128)
        assert _unitaries_equivalent(U, U, atol=1e-8, allow_global_phase=True)

    def test_global_phase_difference(self) -> None:
        """Matrices differing by a global phase are equivalent when allowed."""
        U = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        phase = np.exp(1j * np.pi / 4)
        V = phase * U
        assert _unitaries_equivalent(U, V, atol=1e-8, allow_global_phase=True)

    def test_non_equivalent_matrices(self) -> None:
        """Non-equivalent matrices return False."""
        U = np.eye(2, dtype=np.complex128)
        V = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # Pauli-X
        assert not _unitaries_equivalent(U, V, atol=1e-8, allow_global_phase=True)

    def test_global_phase_not_allowed(self) -> None:
        """Phase-different matrices are NOT equivalent when allow_global_phase=False."""
        U = np.eye(2, dtype=np.complex128)
        phase = np.exp(1j * np.pi / 4)
        V = phase * U
        assert not _unitaries_equivalent(U, V, atol=1e-8, allow_global_phase=False)

    def test_close_but_not_exact(self) -> None:
        """Matrices within tolerance are equivalent."""
        U = np.eye(2, dtype=np.complex128)
        V = U + 1e-9 * np.ones((2, 2), dtype=np.complex128)
        assert _unitaries_equivalent(U, V, atol=1e-6, allow_global_phase=False)

    def test_hadamard_self_equivalent(self) -> None:
        """Hadamard matrix is equivalent to itself."""
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        assert _unitaries_equivalent(H, H, atol=1e-10, allow_global_phase=True)


# ---------------------------------------------------------------------------
# _kraus_to_choi
# ---------------------------------------------------------------------------


class TestKrausToChoi:
    """Tests for :func:`_kraus_to_choi`."""

    def test_identity_channel(self) -> None:
        """Identity Kraus ops ``[I]`` give the maximally entangled state projector.

        For a single-qubit identity channel:
            |Φ⟩ = |00⟩ + |11⟩ (un-normalised)
            Choi = |Φ⟩⟨Φ| = [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]
        """
        I = np.eye(2, dtype=np.complex128)
        choi = _kraus_to_choi([I])

        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = 1.0
        expected[0, 3] = 1.0
        expected[3, 0] = 1.0
        expected[3, 3] = 1.0

        np.testing.assert_allclose(choi, expected, atol=1e-12)

    def test_choi_shape(self) -> None:
        """Choi matrix has correct shape ``(d², d²)``."""
        K = np.eye(3, dtype=np.complex128)
        choi = _kraus_to_choi([K])
        assert choi.shape == (9, 9)

    def test_depolarising_channel(self) -> None:
        """Depolarising channel Kraus ops produce a valid Choi matrix."""
        # Single-qubit depolarising channel with p=1 (fully depolarising)
        I = np.eye(2, dtype=np.complex128) / 2
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2

        choi = _kraus_to_choi([I, X, Y, Z])
        # Choi matrix should be Hermitian
        np.testing.assert_allclose(choi, choi.conj().T, atol=1e-12)
        # Trace should be d = 2 (for trace-preserving channel)
        np.testing.assert_allclose(np.trace(choi), 2.0, atol=1e-12)


# ---------------------------------------------------------------------------
# _is_ibm_backend
# ---------------------------------------------------------------------------


class TestIsIbmBackend:
    """Tests for :func:`_is_ibm_backend`."""

    def test_mock_object_returns_false(self) -> None:
        """A plain mock object is not an IBM backend."""
        assert not _is_ibm_backend(object())

    def test_none_returns_false(self) -> None:
        """None is not an IBM backend."""
        assert not _is_ibm_backend(None)

    def test_string_returns_false(self) -> None:
        """A string is not an IBM backend."""
        assert not _is_ibm_backend("ibm_brisbane")


# ---------------------------------------------------------------------------
# _backend_name
# ---------------------------------------------------------------------------


class TestBackendName:
    """Tests for :func:`_backend_name`."""

    def test_string_name_attribute(self) -> None:
        """Object with a string ``.name`` returns that string."""

        class Backend:
            name = "fake_backend"

        assert _backend_name(Backend()) == "fake_backend"

    def test_callable_name_attribute(self) -> None:
        """Object with a callable ``.name()`` returns its return value."""

        class Backend:
            def name(self) -> str:
                return "callable_backend"

        assert _backend_name(Backend()) == "callable_backend"

    def test_no_name_returns_repr(self) -> None:
        """Object with no ``.name`` returns ``repr()``."""

        class Backend:
            def __repr__(self) -> str:
                return "<MyBackend>"

        assert _backend_name(Backend()) == "<MyBackend>"

    def test_none_name_returns_repr(self) -> None:
        """Object with ``name=None`` returns ``repr()``."""

        class Backend:
            name = None

            def __repr__(self) -> str:
                return "<NoneNameBackend>"

        assert _backend_name(Backend()) == "<NoneNameBackend>"
