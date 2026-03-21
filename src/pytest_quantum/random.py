"""Random quantum state and circuit generators for property-based testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def random_statevector(
    n_qubits: int,
    *,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Generate a random normalised statevector (Haar-random pure state).

    Args:
        n_qubits: Number of qubits (statevector length = 2**n_qubits).
        seed:     Optional random seed for reproducibility.

    Returns:
        Normalised complex128 array of shape ``(2**n_qubits,)``.

    Example::

        from pytest_quantum.random import random_statevector

        sv = random_statevector(2, seed=0)
        assert sv.shape == (4,)
        assert abs(sum(abs(sv) ** 2) - 1) < 1e-12
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    sv = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    result: NDArray[np.complex128] = (sv / np.linalg.norm(sv)).astype(np.complex128)
    return result


def random_density_matrix(
    n_qubits: int,
    rank: int | None = None,
    *,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Generate a random valid density matrix (PSD, trace 1).

    Args:
        n_qubits: Number of qubits (matrix size = 2**n_qubits × 2**n_qubits).
        rank:     Matrix rank (default: full rank = 2**n_qubits).
                  Use ``rank=1`` for a random pure state density matrix.
        seed:     Optional random seed.

    Returns:
        Complex128 array of shape ``(2**n_qubits, 2**n_qubits)``.

    Example::

        from pytest_quantum.random import random_density_matrix

        rho = random_density_matrix(1, seed=42)
        assert rho.shape == (2, 2)
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    r = rank if rank is not None else dim
    A = rng.standard_normal((dim, r)) + 1j * rng.standard_normal((dim, r))
    rho = A @ A.conj().T
    rho_n: NDArray[np.complex128] = (rho / np.trace(rho)).astype(np.complex128)
    return rho_n


def random_unitary(
    n_qubits: int,
    *,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Generate a Haar-random unitary matrix (CUE).

    Uses QR decomposition of the Ginibre ensemble with phase correction to
    guarantee an exact Haar distribution.

    Args:
        n_qubits: Number of qubits (matrix size = 2**n_qubits × 2**n_qubits).
        seed:     Optional random seed.

    Returns:
        Unitary complex128 array of shape ``(2**n_qubits, 2**n_qubits)``.

    Example::

        from pytest_quantum.random import random_unitary
        import numpy as np

        U = random_unitary(2, seed=0)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    # Phase correction: multiply each column by the sign of the diagonal of R
    phases = np.diag(R) / np.abs(np.diag(R))
    return (Q * phases).astype(np.complex128)


def random_kraus_channel(
    n_qubits: int,
    n_kraus: int = 4,
    *,
    seed: int | None = None,
) -> list[NDArray[np.complex128]]:
    """Generate random valid Kraus operators for a CPTP channel.

    Constructs a random Stinespring isometry and extracts Kraus operators
    that satisfy the completeness relation ``∑ K†K = I``.

    Args:
        n_qubits: Number of qubits.
        n_kraus:  Number of Kraus operators (default 4).
        seed:     Optional random seed.

    Returns:
        List of *n_kraus* complex128 arrays of shape
        ``(2**n_qubits, 2**n_qubits)``.

    Example::

        from pytest_quantum.random import random_kraus_channel
        from pytest_quantum import assert_channel_is_cptp

        kraus = random_kraus_channel(1, seed=7)
        assert_channel_is_cptp(kraus)
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    total_rows = n_kraus * dim
    # Build a tall random matrix and QR-decompose to get an isometry
    A = rng.standard_normal((total_rows, dim)) + 1j * rng.standard_normal(
        (total_rows, dim)
    )
    V, _ = np.linalg.qr(A)
    # Each block of `dim` rows is one Kraus operator
    return [V[i * dim : (i + 1) * dim, :].astype(np.complex128) for i in range(n_kraus)]


def depolarizing_kraus(
    n_qubits: int,
    error_rate: float,
) -> list[NDArray[np.complex128]]:
    """Return Kraus operators for the single-qubit depolarising channel.

    Channel definition::

        E(ρ) = (1−p)ρ + (p/3)(XρX + YρY + ZρZ)

    Args:
        n_qubits:   Must be 1; the depolarising channel is defined per qubit.
        error_rate: Depolarisation probability *p* in [0, 1].

    Returns:
        List of 4 Kraus operators:
        ``[√(1−p)·I, √(p/3)·X, √(p/3)·Y, √(p/3)·Z]``.

    Raises:
        ValueError: If *n_qubits* ≠ 1 or *error_rate* is not in [0, 1].

    Example::

        from pytest_quantum.random import depolarizing_kraus
        from pytest_quantum import assert_channel_is_cptp

        assert_channel_is_cptp(depolarizing_kraus(1, 0.1))
    """
    if n_qubits != 1:
        raise ValueError(
            "depolarizing_kraus only supports single-qubit channels (n_qubits=1)"
        )
    if not (0 <= error_rate <= 1):
        raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")
    p = error_rate
    eye2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return [
        np.sqrt(1 - p) * eye2,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]
