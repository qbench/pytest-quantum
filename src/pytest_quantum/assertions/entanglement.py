"""Entanglement and geometric state assertions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytest_quantum.assertions.density import _partial_trace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def assert_entanglement_entropy_below(
    statevector: object,
    partition: list[int],
    max_entropy: float,
    *,
    n_qubits: int | None = None,
) -> None:
    """Assert von Neumann entanglement entropy S(rho_A) <= max_entropy.

    For a pure state |psi>, partition qubits into subsystem A (partition
    indices) and subsystem B (rest). Computes S(rho_A) = -Tr(rho_A log2 rho_A).

    max_entropy=0 means the state is separable (product state) on this
    partition. max_entropy=1 means at most 1 ebit of entanglement.

    Args:
        statevector: Pure state amplitudes (1D complex array of length 2^n).
        partition:   Qubit indices to keep for subsystem A (big-endian).
        max_entropy: Maximum entanglement entropy in bits (nats if log2 is
            replaced — here we use log2 so units are bits).
        n_qubits:    Total qubit count (inferred from len if not given).

    Raises:
        AssertionError: If S(rho_A) > max_entropy.
        ValueError: If statevector length is not a power of 2.
    """
    sv: NDArray[np.complex128] = np.asarray(statevector, dtype=np.complex128).flatten()
    n = n_qubits if n_qubits is not None else round(np.log2(len(sv)))
    if 2**n != len(sv):
        raise ValueError(
            f"Statevector length {len(sv)} is not a power of 2 "
            f"(expected 2^{n} = {2**n})."
        )
    rho: NDArray[np.complex128] = np.outer(sv, sv.conj())
    rho_a = _partial_trace(rho, n, list(partition))
    eigenvalues = np.linalg.eigvalsh(rho_a)
    # Keep only positive eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    entropy = float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    if entropy > max_entropy + 1e-12:
        raise AssertionError(
            f"Entanglement entropy S(rho_A) = {entropy:.6f} bits exceeds "
            f"max_entropy = {max_entropy}.\n"
            f"  Partition (subsystem A): {partition}\n"
            f"  Subsystem A dim: {rho_a.shape[0]}"
        )


def assert_bloch_sphere_close(
    statevector: object,
    expected_theta: float,
    expected_phi: float,
    *,
    atol: float = 0.1,
) -> None:
    """Assert single-qubit state is close to expected Bloch sphere position.

    Bloch vector: (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    theta in [0, pi]: polar angle (0=|0>, pi=|1>)
    phi in [0, 2pi): azimuthal angle

    Args:
        statevector:    Single-qubit state [alpha, beta] (length-2 array).
        expected_theta: Expected polar angle in radians.
        expected_phi:   Expected azimuthal angle in radians.
        atol:           Tolerance for Bloch vector Euclidean distance
                        (default 0.1).

    Raises:
        AssertionError: If Bloch vector distance > atol.
        ValueError: If statevector is not length 2.
    """
    sv: NDArray[np.complex128] = np.asarray(statevector, dtype=np.complex128).flatten()
    if len(sv) != 2:
        raise ValueError(
            f"assert_bloch_sphere_close requires a single-qubit state "
            f"(length 2), got length {len(sv)}."
        )
    norm = np.linalg.norm(sv)
    if norm < 1e-12:
        raise ValueError("Statevector has zero norm.")
    sv = sv / norm

    # Bloch vector from statevector
    r_x = float(2.0 * np.real(sv[0].conj() * sv[1]))
    r_y = float(2.0 * np.imag(sv[0].conj() * sv[1]))
    r_z = float(abs(sv[0]) ** 2 - abs(sv[1]) ** 2)

    # Expected Bloch vector from (theta, phi)
    ex = float(np.sin(expected_theta) * np.cos(expected_phi))
    ey = float(np.sin(expected_theta) * np.sin(expected_phi))
    ez = float(np.cos(expected_theta))

    dist = float(np.sqrt((r_x - ex) ** 2 + (r_y - ey) ** 2 + (r_z - ez) ** 2))
    if dist > atol:
        # Convert actual Bloch vector to (theta, phi)
        actual_theta = float(np.arccos(np.clip(r_z, -1.0, 1.0)))
        actual_phi = float(np.arctan2(r_y, r_x)) % (2 * float(np.pi))
        raise AssertionError(
            f"Bloch sphere position mismatch (Euclidean distance {dist:.4f} > {atol}).\n"
            f"  Expected (theta, phi) : ({expected_theta:.4f}, {expected_phi:.4f})\n"
            f"  Actual   (theta, phi) : ({actual_theta:.4f}, {actual_phi:.4f})\n"
            f"  Expected Bloch vector : ({ex:.4f}, {ey:.4f}, {ez:.4f})\n"
            f"  Actual   Bloch vector : ({r_x:.4f}, {r_y:.4f}, {r_z:.4f})"
        )


def assert_schmidt_rank_at_most(
    statevector: object,
    partition: list[int],
    max_rank: int,
    *,
    n_qubits: int | None = None,
    tol: float = 1e-10,
) -> None:
    """Assert Schmidt rank of bipartite pure state partition is at most max_rank.

    Schmidt rank = 1 means separable (product state) on this partition.

    Args:
        statevector: Pure state amplitudes (1D complex array of length 2^n).
        partition:   Qubit indices in subsystem A.
        max_rank:    Maximum allowed Schmidt rank.
        n_qubits:    Total qubit count (inferred if not given).
        tol:         Singular value threshold for rank counting (default 1e-10).

    Raises:
        AssertionError: If Schmidt rank > max_rank.
        ValueError: If statevector length is not a power of 2.
    """
    sv: NDArray[np.complex128] = np.asarray(statevector, dtype=np.complex128).flatten()
    n = n_qubits if n_qubits is not None else round(np.log2(len(sv)))
    if 2**n != len(sv):
        raise ValueError(
            f"Statevector length {len(sv)} is not a power of 2 "
            f"(expected 2^{n} = {2**n})."
        )

    dim_a = 2 ** len(partition)
    dim_b = 2 ** (n - len(partition))

    # Permute qubit axes so partition qubits come first, then reshape to matrix
    all_qubits = list(range(n))
    b_qubits = [q for q in all_qubits if q not in partition]
    perm = list(partition) + b_qubits

    sv_tensor = sv.reshape([2] * n)
    sv_permuted = np.transpose(sv_tensor, perm)
    matrix = sv_permuted.reshape(dim_a, dim_b)

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_values > tol))

    if rank > max_rank:
        raise AssertionError(
            f"Schmidt rank {rank} exceeds max_rank {max_rank}.\n"
            f"  Partition (subsystem A) : {partition}\n"
            f"  Subsystem A dim         : {dim_a}\n"
            f"  Subsystem B dim         : {dim_b}\n"
            f"  Singular values         : {singular_values.tolist()}\n"
            f"  (threshold for counting : {tol:.2e})"
        )
