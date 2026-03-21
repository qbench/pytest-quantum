"""Density matrix assertions for mixed quantum states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def assert_density_matrix_close(
    rho: object,
    sigma: object,
    *,
    atol: float = 1e-6,
) -> None:
    """Assert two density matrices are close element-wise after normalisation.

    Args:
        rho:   First density matrix, shape (d, d).
        sigma: Second density matrix, shape (d, d).
        atol:  Absolute tolerance (default 1e-6).

    Raises:
        AssertionError: If matrices differ by more than atol.
        ValueError: If shapes mismatch or matrices are invalid.

    Example::

        import numpy as np
        from pytest_quantum import assert_density_matrix_close

        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        assert_density_matrix_close(rho, rho)
    """
    rho_arr = np.asarray(rho, dtype=np.complex128)
    sigma_arr = np.asarray(sigma, dtype=np.complex128)
    _validate_dm(rho_arr, "rho")
    _validate_dm(sigma_arr, "sigma")
    if rho_arr.shape != sigma_arr.shape:
        raise AssertionError(
            f"Shape mismatch: rho={rho_arr.shape}, sigma={sigma_arr.shape}"
        )
    rho_n = rho_arr / np.trace(rho_arr)
    sigma_n = sigma_arr / np.trace(sigma_arr)
    if np.allclose(rho_n, sigma_n, atol=atol):
        return
    max_diff = float(np.max(np.abs(rho_n - sigma_n)))
    td = _trace_distance(rho_n, sigma_n)
    raise AssertionError(
        f"Density matrices are not close.\n"
        f"  Max |ρ - σ|  : {max_diff:.2e}   (tolerance: {atol:.2e})\n"
        f"  Trace distance: {td:.4f}\n"
        f"  Hint: use assert_trace_distance_below for a physically motivated bound."
    )


def assert_trace_distance_below(
    rho: object,
    sigma: object,
    *,
    max_distance: float = 0.01,
) -> None:
    """Assert trace distance T(ρ,σ) = ½ Tr(|ρ-σ|) is at most max_distance.

    T=0 means identical states, T=1 means perfectly distinguishable.
    Physical meaning: maximum probability of distinguishing the two states
    in any single measurement equals T(ρ,σ).

    Args:
        rho:          First density matrix.
        sigma:        Second density matrix.
        max_distance: Maximum allowed trace distance (default 0.01).

    Raises:
        AssertionError: If T(ρ,σ) > max_distance.

    Example::

        import numpy as np
        from pytest_quantum import assert_trace_distance_below

        rho = np.eye(2, dtype=complex) / 2
        assert_trace_distance_below(rho, rho, max_distance=0.01)
    """
    rho_arr = np.asarray(rho, dtype=np.complex128)
    sigma_arr = np.asarray(sigma, dtype=np.complex128)
    _validate_dm(rho_arr, "rho")
    _validate_dm(sigma_arr, "sigma")
    if rho_arr.shape != sigma_arr.shape:
        raise AssertionError(
            f"Shape mismatch: rho={rho_arr.shape}, sigma={sigma_arr.shape}"
        )
    rho_n = rho_arr / np.trace(rho_arr)
    sigma_n = sigma_arr / np.trace(sigma_arr)
    td = _trace_distance(rho_n, sigma_n)
    if td <= max_distance:
        return
    raise AssertionError(
        f"Trace distance T(ρ,σ) = {td:.4f} exceeds max_distance {max_distance:.4f}.\n"
        f"  T=0: identical states.  T=1: perfectly distinguishable.\n"
        f"  Hint: check for noise — increase max_distance or reduce error_rate."
    )


def assert_purity_above(
    rho: object,
    *,
    min_purity: float = 0.95,
) -> None:
    """Assert purity Tr(ρ²) ≥ min_purity.

    Pure state: Tr(ρ²)=1.0. Maximally mixed d×d state: Tr(ρ²)=1/d.

    Args:
        rho:        Density matrix, shape (d, d).
        min_purity: Minimum required purity (default 0.95).

    Raises:
        AssertionError: If Tr(ρ²) < min_purity.

    Example::

        import numpy as np
        from pytest_quantum import assert_purity_above

        psi = np.array([[1], [0]], dtype=complex)
        rho = psi @ psi.conj().T
        assert_purity_above(rho, min_purity=0.99)
    """
    rho_arr = np.asarray(rho, dtype=np.complex128)
    _validate_dm(rho_arr, "rho")
    rho_n = rho_arr / np.trace(rho_arr)
    purity = float(np.real(np.trace(rho_n @ rho_n)))
    if purity >= min_purity:
        return
    d = rho_arr.shape[0]
    raise AssertionError(
        f"Purity Tr(ρ²) = {purity:.4f} is below min_purity {min_purity:.4f}.\n"
        f"  Pure state=1.0, maximally mixed ({d}×{d})={1 / d:.4f}.\n"
        f"  Hint: high depolarizing noise reduces purity."
    )


def assert_partial_trace_close(
    rho: object,
    keep_qubits: list[int],
    expected: object,
    *,
    atol: float = 1e-6,
) -> None:
    """Assert the reduced density matrix (partial trace) is close to expected.

    Traces out all qubits NOT in keep_qubits, then compares the
    resulting reduced density matrix to expected.

    Args:
        rho:         Full density matrix, shape (2**n, 2**n) for n qubits.
        keep_qubits: List of qubit indices to keep (0-indexed, big-endian).
        expected:    Expected reduced density matrix.
        atol:        Absolute tolerance (default 1e-6).

    Raises:
        AssertionError: If reduced density matrix differs from expected.
        ValueError: If dimensions are inconsistent.

    Example::

        import numpy as np
        from pytest_quantum import assert_partial_trace_close

        # Bell state: |00> + |11> / sqrt(2) — partial trace gives I/2
        bell = np.array([[1], [0], [0], [1]], dtype=complex) / np.sqrt(2)
        rho = bell @ bell.conj().T
        mixed = np.eye(2, dtype=complex) / 2
        assert_partial_trace_close(rho, keep_qubits=[0], expected=mixed)
    """
    rho_arr = np.asarray(rho, dtype=np.complex128)
    expected_arr = np.asarray(expected, dtype=np.complex128)
    _validate_dm(rho_arr, "rho")
    d = rho_arr.shape[0]
    n = round(np.log2(d))
    if 2**n != d:
        raise ValueError(f"rho dimension {d} is not a power of 2")
    reduced = _partial_trace(rho_arr, n, list(keep_qubits))
    rho_n = reduced / np.trace(reduced)
    expected_n = expected_arr / np.trace(expected_arr)
    if np.allclose(rho_n, expected_n, atol=atol):
        return
    max_diff = float(np.max(np.abs(rho_n - expected_n)))
    raise AssertionError(
        f"Reduced density matrix (keeping qubits {list(keep_qubits)}) "
        f"does not match expected.\n"
        f"  Max |ρ_reduced - expected|: {max_diff:.2e}   (tolerance: {atol:.2e})"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_dm(arr: NDArray[np.complex128], name: str) -> None:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    tr = np.trace(arr)
    if abs(tr) < 1e-10:
        raise ValueError(f"{name} has zero trace — not a valid density matrix")


def _trace_distance(
    rho: NDArray[np.complex128],
    sigma: NDArray[np.complex128],
) -> float:
    diff = rho - sigma
    eigvals = np.linalg.eigvalsh(diff.conj().T @ diff)
    eigvals = np.maximum(eigvals, 0.0)
    return 0.5 * float(np.sum(np.sqrt(eigvals)))


def _partial_trace(
    rho: NDArray[np.complex128],
    n_qubits: int,
    keep: list[int],
) -> NDArray[np.complex128]:
    """Partial trace via einsum: correct for any subset of qubits.

    Convention: qubit 0 is most significant (big-endian / leftmost in
    |q0 q1 q2...>).
    """
    import string

    trace_out = [i for i in range(n_qubits) if i not in keep]

    # Reshape rho from (2^n, 2^n) to (2, 2, ..., 2) with 2*n indices.
    # First n indices are "row" qubits, last n are "col" qubits.
    rho_tensor = rho.reshape([2] * (2 * n_qubits))

    # Build einsum labels: row labels a,b,c,... and col labels p,q,r,...
    # (use two non-overlapping ranges of the alphabet)
    row_labels = list(string.ascii_lowercase[:n_qubits])
    col_labels = list(string.ascii_lowercase[n_qubits : 2 * n_qubits])

    # For traced-out qubits: force row_label == col_label (diagonal trace)
    for q in trace_out:
        col_labels[q] = row_labels[q]

    input_str = "".join(row_labels) + "".join(col_labels)
    out_row = "".join(row_labels[q] for q in sorted(keep))
    out_col = "".join(col_labels[q] for q in sorted(keep))
    output_str = out_row + out_col

    reduced = np.einsum(input_str + "->" + output_str, rho_tensor)

    d_keep = 2 ** len(keep)
    return reduced.reshape(d_keep, d_keep)
