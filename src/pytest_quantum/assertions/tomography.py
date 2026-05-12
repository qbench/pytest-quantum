"""Quantum state and process tomography assertions."""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Single-qubit Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULIS = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def assert_state_tomography_close(
    measurements: dict[str, float | NDArray],
    expected_state: NDArray,
    *,
    atol: float = 0.05,
) -> None:
    """Assert that tomographic measurements reconstruct a state close to *expected_state*.

    Uses linear inversion tomography for single-qubit states. For multi-qubit
    states, expects Pauli basis labels as tensor products (e.g. ``"XZ"``).

    Args:
        measurements: Dict mapping Pauli basis labels (``"X"``, ``"Y"``, ``"Z"``
            for single qubit, or tensor products like ``"XZ"`` for multi-qubit)
            to expectation values (floats or arrays).
        expected_state: Expected density matrix (2D array) or pure state vector
            (1D array). If 1D, it is converted to a density matrix via outer product.
        atol: Tolerance for trace distance comparison.

    Raises:
        AssertionError: If the trace distance exceeds *atol*.

    Example::

        measurements = {"X": 0.0, "Y": 0.0, "Z": 1.0}
        assert_state_tomography_close(measurements, np.array([1, 0]))
    """
    expected = np.asarray(expected_state, dtype=np.complex128)
    if expected.ndim == 1:
        expected = np.outer(expected, expected.conj())

    n_qubits = int(np.log2(expected.shape[0]))
    dim = 2 ** n_qubits

    # Reconstruct density matrix via linear inversion
    rho = np.eye(dim, dtype=np.complex128) / dim
    if n_qubits == 1:
        for label in ("X", "Y", "Z"):
            if label in measurements:
                exp_val = float(np.real(measurements[label]))
                rho = rho + 0.5 * exp_val * _PAULIS[label]
    else:
        # Multi-qubit: iterate over all Pauli tensor products in measurements
        for label, exp_val in measurements.items():
            if all(c in "IXYZ" for c in label) and len(label) == n_qubits:
                pauli = _PAULIS[label[0]]
                for c in label[1:]:
                    pauli = np.kron(pauli, _PAULIS[c])
                rho = rho + (1.0 / dim) * float(np.real(exp_val)) * pauli

    # Trace distance: T(rho, sigma) = 0.5 * ||rho - sigma||_1
    diff = rho - expected
    eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
    trace_dist = 0.5 * np.sum(np.sqrt(np.maximum(eigenvalues, 0)))

    if trace_dist > atol:
        raise AssertionError(
            f"State tomography trace distance {trace_dist:.6f} exceeds "
            f"tolerance {atol}.\n"
            f"Reconstructed state diagonal: {np.diag(rho).real}\n"
            f"Expected state diagonal: {np.diag(expected).real}"
        )


def assert_process_tomography_close(
    chi_matrix: NDArray,
    expected_chi: NDArray,
    *,
    atol: float = 0.05,
) -> None:
    """Assert that a process (chi) matrix is close to *expected_chi*.

    Compares using the Frobenius norm and validates that the chi matrix
    is Hermitian and positive semidefinite.

    Args:
        chi_matrix: The measured process matrix.
        expected_chi: The expected process matrix.
        atol: Tolerance for Frobenius norm comparison.

    Raises:
        AssertionError: If the chi matrix is not Hermitian, not positive
            semidefinite, or the Frobenius distance exceeds *atol*.

    Example::

        chi = np.eye(4) / 4  # identity process
        assert_process_tomography_close(chi, np.eye(4) / 4)
    """
    chi = np.asarray(chi_matrix, dtype=np.complex128)
    expected = np.asarray(expected_chi, dtype=np.complex128)

    # Check Hermitian
    if not np.allclose(chi, chi.conj().T, atol=1e-10):
        raise AssertionError(
            "Chi matrix is not Hermitian.\n"
            f"Max deviation: {np.max(np.abs(chi - chi.conj().T)):.2e}"
        )

    # Check positive semidefinite
    eigenvalues = np.linalg.eigvalsh(chi)
    if np.any(eigenvalues < -1e-10):
        raise AssertionError(
            f"Chi matrix is not positive semidefinite.\n"
            f"Minimum eigenvalue: {np.min(eigenvalues):.2e}"
        )

    # Frobenius distance
    frob_dist = np.linalg.norm(chi - expected, "fro")
    if frob_dist > atol:
        raise AssertionError(
            f"Process tomography Frobenius distance {frob_dist:.6f} exceeds "
            f"tolerance {atol}."
        )
