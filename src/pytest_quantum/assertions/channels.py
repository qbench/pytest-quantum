"""Quantum channel and operator assertions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def assert_hermitian(matrix: Any, *, atol: float = 1e-8) -> None:
    """Assert matrix is Hermitian: A == A†.

    Args:
        matrix: Square matrix (any array-like convertible to numpy).
        atol:   Absolute tolerance for element-wise comparison (default 1e-8).

    Raises:
        AssertionError: If the matrix is not Hermitian, with shape info and
            maximum deviation from Hermiticity.
        ValueError: If the matrix is not square.
    """
    A: NDArray[np.complex128] = np.asarray(matrix, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"assert_hermitian requires a square 2D matrix, got shape {A.shape}"
        )
    diff = A - A.conj().T
    max_dev = float(np.max(np.abs(diff)))
    if not np.allclose(A, A.conj().T, atol=atol):
        raise AssertionError(
            f"Matrix is not Hermitian (A ≠ A†).\n"
            f"  Shape     : {A.shape}\n"
            f"  Max |A - A†| : {max_dev:.2e}   (tolerance: {atol:.2e})"
        )


def assert_positive_semidefinite(matrix: Any, *, atol: float = 1e-8) -> None:
    """Assert matrix is positive semi-definite: all eigenvalues >= -atol.

    Validates Hermiticity first, then checks that the smallest eigenvalue
    is >= -atol.

    Args:
        matrix: Square Hermitian matrix (any array-like).
        atol:   Tolerance for eigenvalue non-negativity (default 1e-8).

    Raises:
        AssertionError: If any eigenvalue is < -atol.
        ValueError: If the matrix is not square.
    """
    A: NDArray[np.complex128] = np.asarray(matrix, dtype=np.complex128)
    assert_hermitian(A, atol=atol)
    eigvals = np.linalg.eigvalsh(A)
    min_eig = float(np.min(eigvals))
    if min_eig < -atol:
        raise AssertionError(
            f"Matrix is not positive semi-definite.\n"
            f"  Minimum eigenvalue: {min_eig:.6e}   (threshold: -{atol:.2e})\n"
            f"  All eigenvalues    : {eigvals.tolist()}"
        )


def assert_commutes_with(op_a: Any, op_b: Any, *, atol: float = 1e-8) -> None:
    """Assert two square matrices commute: AB == BA.

    Args:
        op_a: First square matrix.
        op_b: Second square matrix (same size as op_a).
        atol: Absolute tolerance for the commutator (default 1e-8).

    Raises:
        AssertionError: If [A, B] = AB - BA has any element > atol,
            showing max(|AB - BA|).
        ValueError: If matrices are not square or have mismatched sizes.
    """
    A: NDArray[np.complex128] = np.asarray(op_a, dtype=np.complex128)
    B: NDArray[np.complex128] = np.asarray(op_b, dtype=np.complex128)
    for name, M in (("op_a", A), ("op_b", B)):
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"{name} must be a square 2D matrix, got shape {M.shape}")
    if A.shape != B.shape:
        raise ValueError(
            f"op_a and op_b must have the same shape; got {A.shape} and {B.shape}"
        )
    commutator = A @ B - B @ A
    max_comm = float(np.max(np.abs(commutator)))
    if not np.allclose(A @ B, B @ A, atol=atol):
        raise AssertionError(
            f"Matrices do not commute: [A, B] ≠ 0.\n"
            f"  Max |AB - BA|: {max_comm:.2e}   (tolerance: {atol:.2e})"
        )


def assert_channel_is_cptp(kraus_ops: list[Any], *, atol: float = 1e-8) -> None:
    """Assert Kraus operators satisfy completeness: sum_i K_i† K_i == I.

    This is the necessary and sufficient condition for a channel to be
    completely positive and trace-preserving (CPTP).

    Args:
        kraus_ops: Non-empty list of Kraus operators (square matrices of
            equal shape).
        atol: Absolute tolerance for the completeness check (default 1e-8).

    Raises:
        AssertionError: If the completeness relation is violated, showing
            Frobenius norm of the deviation.
        ValueError: If the list is empty, operators have different shapes,
            or operators are not square.
    """
    if not kraus_ops:
        raise ValueError("kraus_ops must be a non-empty list of Kraus operators.")

    ops: list[NDArray[np.complex128]] = [
        np.asarray(K, dtype=np.complex128) for K in kraus_ops
    ]
    shape0 = ops[0].shape
    if shape0[0] != shape0[1] or ops[0].ndim != 2:
        raise ValueError(
            f"Kraus operators must be square 2D matrices, got shape {shape0}"
        )
    for i, K in enumerate(ops):
        if K.shape != shape0:
            raise ValueError(
                f"All Kraus operators must have the same shape; "
                f"ops[0] has shape {shape0} but ops[{i}] has shape {K.shape}"
            )

    d = shape0[0]
    completeness: NDArray[np.complex128] = sum(
        K.conj().T @ K
        for K in ops  # type: ignore[assignment]
    )
    identity = np.eye(d, dtype=np.complex128)
    deviation_fro = float(np.linalg.norm(completeness - identity, ord="fro"))
    if not np.allclose(completeness, identity, atol=atol):
        raise AssertionError(
            f"Kraus operators do not satisfy the CPTP completeness relation (ΣK†K ≠ I).\n"
            f"  Number of Kraus operators: {len(ops)}\n"
            f"  Matrix dimension: {d}x{d}\n"
            f"  ||ΣK†K - I||_fro: {deviation_fro:.6f}  (tolerance: {atol:.2e})\n"
            f"  Hint: Check that your Kraus operators form a valid quantum channel.\n"
            f"        Use depolarizing_kraus(n_qubits=1, error_rate=0.1) for a valid example."
        )


def assert_process_fidelity_above(
    channel_a: Any,
    channel_b: Any,
    threshold: float = 0.99,
    *,
    atol: float = 1e-8,
) -> None:
    """Assert process fidelity F_process(A, B) >= threshold.

    Supports:
    - List of numpy Kraus operators
    - Qiskit Kraus/Choi/SuperOp objects (uses qiskit.quantum_info.process_fidelity)
    - numpy unitary matrices (converts to single-Kraus channel)

    Process fidelity for unitary channels: F = |Tr(A† B)|² / d²
    For general channels uses Choi matrix inner product.

    Args:
        channel_a: First channel (Kraus list, unitary matrix, or Qiskit object).
        channel_b: Second channel (same supported types as channel_a).
        threshold: Minimum acceptable process fidelity (default 0.99).
        atol:      Tolerance for unitary check (default 1e-8).

    Raises:
        AssertionError: If process fidelity < threshold.
        TypeError: If channel types are not supported.
    """
    # Qiskit objects — delegate to qiskit.quantum_info.process_fidelity
    mod_a = type(channel_a).__module__
    mod_b = type(channel_b).__module__
    if mod_a.startswith("qiskit") or mod_b.startswith("qiskit"):
        try:
            from qiskit.quantum_info import process_fidelity

            fidelity = float(process_fidelity(channel_a, channel_b))
        except ImportError as exc:
            raise ImportError(
                "qiskit is required for Qiskit channel fidelity. "
                "Install it with: pip install pytest-quantum[qiskit]"
            ) from exc
        if fidelity < threshold:
            raise AssertionError(
                f"Process fidelity {fidelity:.4f} < threshold {threshold}.\n"
                f"  Fidelity:  {fidelity:.4f}\n"
                f"  Threshold: {threshold}\n"
                f"  Gap:       {threshold - fidelity:.4f}"
            )
        return

    # Normalise inputs: lists → Kraus ops; 2D numpy → single-Kraus list
    def _to_kraus_list(ch: Any) -> list[NDArray[np.complex128]]:
        if isinstance(ch, list):
            return [np.asarray(K, dtype=np.complex128) for K in ch]
        arr = np.asarray(ch, dtype=np.complex128)
        if arr.ndim == 2:
            return [arr]
        raise TypeError(
            f"Expected a list of Kraus operators or a 2D unitary matrix; "
            f"got {type(ch).__qualname__!r} with shape {arr.shape}"
        )

    kraus_a = _to_kraus_list(channel_a)
    kraus_b = _to_kraus_list(channel_b)

    # Unitary path: single Kraus == unitary matrix
    if len(kraus_a) == 1 and len(kraus_b) == 1:
        U = kraus_a[0]
        V = kraus_b[0]
        d = U.shape[0]
        fidelity = float(abs(np.trace(U.conj().T @ V)) ** 2) / (d * d)
    else:
        choi_a = _kraus_to_choi(kraus_a)
        choi_b = _kraus_to_choi(kraus_b)
        d = kraus_a[0].shape[0]
        fidelity = float(np.real(np.trace(choi_a.conj().T @ choi_b))) / (d * d)

    if fidelity < threshold:
        raise AssertionError(
            f"Process fidelity {fidelity:.4f} < threshold {threshold}.\n"
            f"  Fidelity:  {fidelity:.4f}\n"
            f"  Threshold: {threshold}\n"
            f"  Gap:       {threshold - fidelity:.4f}"
        )


def assert_noise_fidelity_above(
    noisy_dm: Any,
    ideal_state: Any,
    threshold: float = 0.99,
) -> None:
    """Assert state fidelity between noisy density matrix and ideal state.

    F(rho, |psi>) = <psi|rho|psi>
    F(rho, sigma) = Tr(sigma @ rho)

    Args:
        noisy_dm:    Density matrix (2D numpy array).
        ideal_state: Pure statevector (1D) or density matrix (2D).
        threshold:   Minimum acceptable fidelity (default 0.99).

    Raises:
        AssertionError: If fidelity < threshold.
        ValueError: If shapes are inconsistent.
    """
    rho: NDArray[np.complex128] = np.asarray(noisy_dm, dtype=np.complex128)
    ideal: NDArray[np.complex128] = np.asarray(ideal_state, dtype=np.complex128)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"noisy_dm must be a square 2D matrix, got shape {rho.shape}")

    if ideal.ndim == 1:
        # Pure statevector: F = <psi|rho|psi>
        psi = ideal / np.linalg.norm(ideal)
        fidelity = float(np.real(psi.conj() @ rho @ psi))
    elif ideal.ndim == 2:
        # Density matrix: F = Tr(sigma @ rho)
        sigma = ideal / np.trace(ideal)
        fidelity = float(np.real(np.trace(sigma @ rho)))
    else:
        raise ValueError(
            f"ideal_state must be 1D (statevector) or 2D (density matrix), "
            f"got shape {ideal.shape}"
        )

    if fidelity < threshold:
        raise AssertionError(
            f"State fidelity {fidelity:.6f} is below threshold {threshold}.\n"
            f"  F(rho, ideal) = {fidelity:.6f}  <  threshold = {threshold}\n"
            f"  noisy_dm shape: {rho.shape}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


from pytest_quantum._internal import _kraus_to_choi
