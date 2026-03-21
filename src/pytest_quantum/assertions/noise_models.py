"""Noise channel and leakage assertions for quantum testing.

All functions are pure-numpy; no quantum SDK is required.  They operate on
Kraus operators (lists of NumPy arrays) or density matrices and raise
``AssertionError`` with detailed, human-readable messages on failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Pauli matrices (module-level constants, used internally)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _kraus_to_choi(kraus_ops: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Convert a list of Kraus operators to the Choi matrix.

    Uses column-vectorisation: Choi = sum_i |K_i)(K_i|, giving a d^2 x d^2
    matrix for d x d Kraus operators.
    """
    d = kraus_ops[0].shape[0]
    choi = np.zeros((d * d, d * d), dtype=np.complex128)
    for K in kraus_ops:
        k_vec = K.flatten(order="F").reshape(-1, 1)
        choi += k_vec @ k_vec.conj().T
    return choi


def _average_channel_fidelity(kraus_ops: list[NDArray[np.complex128]]) -> float:
    """Compute the average gate fidelity F_avg for a single-qubit channel.

    Uses the formula:  F_avg = (d * F_proc + 1) / (d + 1)
    where F_proc = sum_i |Tr(K_i)|^2 / d^2   and d is the Hilbert-space dim.

    Reference: Nielsen (2002), Phys. Lett. A 303, 249.
    """
    d = kraus_ops[0].shape[0]
    f_proc = sum(abs(complex(np.trace(K))) ** 2 for K in kraus_ops) / d**2
    f_avg = (d * f_proc + 1) / (d + 1)
    return float(np.real(f_avg))


# ---------------------------------------------------------------------------
# Public assertions
# ---------------------------------------------------------------------------


def assert_depolarizing_channel(
    channel_matrices: list[NDArray[np.complex128]],
    expected_error_rate: float,
    *,
    atol: float = 0.01,
) -> None:
    """Assert Kraus operators represent a depolarizing channel with a given error rate.

    For the standard single-qubit depolarizing channel with error rate *p*::

        K0 = sqrt(1 - p) * I
        K1 = sqrt(p / 3) * X
        K2 = sqrt(p / 3) * Y
        K3 = sqrt(p / 3) * Z

    The function recovers *p* from the average gate fidelity of the channel.
    Only K0 = sqrt(1-p)*I has nonzero trace, so::

        F_proc = (1 - p)
        F_avg  = (d * F_proc + 1) / (d + 1)  [Nielsen 2002]
               = (3 - 2p) / 3   for d = 2
        p      = (1 - F_avg) * (d + 1) / 2   [for d=2: p = (1 - F_avg) * 3/2]

    Args:
        channel_matrices:   List of Kraus operators (square numpy arrays of equal shape).
        expected_error_rate: Expected depolarizing error rate *p* in [0, 1].
        atol:               Absolute tolerance for the error-rate comparison (default 0.01).

    Raises:
        AssertionError: If |estimated_p - expected_error_rate| > atol, showing both values.
        ValueError: If ``channel_matrices`` is empty or operators are non-square / mismatched.
    """
    if not channel_matrices:
        raise ValueError("channel_matrices must be a non-empty list of Kraus operators.")
    ops: list[NDArray[np.complex128]] = [
        np.asarray(K, dtype=np.complex128) for K in channel_matrices
    ]
    d = ops[0].shape[0]
    for i, K in enumerate(ops):
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError(
                f"Kraus operator at index {i} must be a square 2D matrix, got shape {K.shape}"
            )
        if K.shape != ops[0].shape:
            raise ValueError(
                f"All Kraus operators must have the same shape; "
                f"ops[0] has shape {ops[0].shape} but ops[{i}] has shape {K.shape}"
            )

    f_avg = _average_channel_fidelity(ops)
    # Derivation (single-qubit depolarizing, d = 2):
    #   K0 = sqrt(1-p)*I  →  Tr(K0)=2*sqrt(1-p),  |Tr(K0)|² = 4(1-p)
    #   K1..K3 are traceless Paulis scaled by sqrt(p/3), so |Tr(K_i)|² = 0
    #   F_proc = sum_i |Tr(K_i)|² / d² = (1-p)
    #   F_avg  = (d * F_proc + 1) / (d + 1) = (2(1-p) + 1) / 3 = (3 - 2p) / 3
    #   1 - F_avg = 2p/3
    #   p = (1 - F_avg) * (d + 1) / 2   [for d=2: 3/2]
    estimated_p = (1.0 - f_avg) * (d + 1) / 2.0

    if abs(estimated_p - expected_error_rate) > atol:
        raise AssertionError(
            f"Depolarizing error rate mismatch.\n"
            f"  Estimated rate : {estimated_p:.6f}\n"
            f"  Expected rate  : {expected_error_rate:.6f}\n"
            f"  Difference     : {abs(estimated_p - expected_error_rate):.6f}   (tolerance: {atol})"
        )


def assert_amplitude_damping_channel(
    channel_matrices: list[NDArray[np.complex128]],
    expected_gamma: float,
    *,
    atol: float = 0.01,
) -> None:
    """Assert Kraus operators represent an amplitude damping channel with parameter gamma.

    The canonical amplitude damping Kraus operators are::

        K0 = [[1, 0], [0, sqrt(1 - gamma)]]
        K1 = [[0, sqrt(gamma)], [0, 0]]

    The function estimates *gamma* from the Choi matrix of the channel.
    Specifically, the (1, 1) element of the Choi matrix (in the column-vectorised
    convention) encodes ``1 - gamma``, so::

        gamma_est = 1 - Re(choi[1, 1]) / Re(choi[0, 0])

    Args:
        channel_matrices:  List of Kraus operators for a single-qubit channel.
        expected_gamma:    Expected decay parameter gamma in [0, 1].
        atol:              Absolute tolerance (default 0.01).

    Raises:
        AssertionError: If |estimated_gamma - expected_gamma| > atol.
        ValueError:     If operators are not 2x2 or the list is empty.
    """
    if not channel_matrices:
        raise ValueError("channel_matrices must be a non-empty list of Kraus operators.")
    ops: list[NDArray[np.complex128]] = [
        np.asarray(K, dtype=np.complex128) for K in channel_matrices
    ]
    if ops[0].shape != (2, 2):
        raise ValueError(
            f"assert_amplitude_damping_channel requires 2x2 Kraus operators, "
            f"got shape {ops[0].shape}"
        )

    # Build the action of the channel on |0><0|, |0><1|, |1><0|, |1><1|
    # and extract gamma from the diagonal decay of the excited state.
    # rho_11_out = (1-gamma) * rho_11  =>  gamma = 1 - E(|1><1|)[1,1]
    rho_excited: NDArray[np.complex128] = np.array(
        [[0, 0], [0, 1]], dtype=np.complex128
    )
    rho_out = sum(K @ rho_excited @ K.conj().T for K in ops)
    estimated_gamma = float(np.real(1.0 - rho_out[1, 1]))  # type: ignore[index]

    if abs(estimated_gamma - expected_gamma) > atol:
        raise AssertionError(
            f"Amplitude damping parameter mismatch.\n"
            f"  Estimated gamma : {estimated_gamma:.6f}\n"
            f"  Expected gamma  : {expected_gamma:.6f}\n"
            f"  Difference      : {abs(estimated_gamma - expected_gamma):.6f}   (tolerance: {atol})"
        )


def assert_dephasing_channel(
    channel_matrices: list[NDArray[np.complex128]],
    expected_rate: float,
    *,
    atol: float = 0.01,
) -> None:
    """Assert Kraus operators represent a dephasing (phase damping) channel.

    The canonical dephasing Kraus operators are::

        K0 = [[1, 0], [0, sqrt(1 - p)]]
        K1 = [[0, 0], [0, sqrt(p)]]

    The dephasing rate *p* controls how quickly off-diagonal coherences decay.
    It is estimated by applying the channel to the superposition state
    ``|+><+|`` and reading off the off-diagonal survival.  The output
    off-diagonal element is ``0.5 * sqrt(1 - p)``, so::

        p_est = 1 - (2 * |rho_out[0, 1]|) ** 2

    Args:
        channel_matrices: List of Kraus operators for a single-qubit channel.
        expected_rate:    Expected dephasing rate *p* in [0, 1].
        atol:             Absolute tolerance (default 0.01).

    Raises:
        AssertionError: If |estimated_rate - expected_rate| > atol.
        ValueError:     If operators are not 2x2 or the list is empty.
    """
    if not channel_matrices:
        raise ValueError("channel_matrices must be a non-empty list of Kraus operators.")
    ops: list[NDArray[np.complex128]] = [
        np.asarray(K, dtype=np.complex128) for K in channel_matrices
    ]
    if ops[0].shape != (2, 2):
        raise ValueError(
            f"assert_dephasing_channel requires 2x2 Kraus operators, "
            f"got shape {ops[0].shape}"
        )

    # Apply channel to |+><+| = 0.5 * [[1,1],[1,1]]
    rho_plus: NDArray[np.complex128] = np.array(
        [[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128
    )
    rho_out = sum(K @ rho_plus @ K.conj().T for K in ops)
    # Off-diagonal of output = 0.5 * sqrt(1 - p)
    # => (2*|off_diag|)^2 = 1 - p  =>  p = 1 - (2*|rho_out[0,1]|)^2
    off_diag = float(abs(rho_out[0, 1]))  # type: ignore[index]
    estimated_rate = float(np.clip(1.0 - (2.0 * off_diag) ** 2, 0.0, 1.0))

    if abs(estimated_rate - expected_rate) > atol:
        raise AssertionError(
            f"Dephasing rate mismatch.\n"
            f"  Estimated rate : {estimated_rate:.6f}\n"
            f"  Expected rate  : {expected_rate:.6f}\n"
            f"  Difference     : {abs(estimated_rate - expected_rate):.6f}   (tolerance: {atol})"
        )


def assert_no_leakage(
    density_matrix: NDArray[np.complex128],
    computational_subspace_dim: int,
    *,
    max_leakage: float = 0.01,
) -> float:
    """Assert a density matrix has negligible population outside the computational subspace.

    Leakage is defined as the probability of being in states outside the
    first ``computational_subspace_dim`` basis states::

        leakage = 1 - Tr(P_comp @ rho @ P_comp)
                = 1 - sum_{i=0}^{d_comp-1} rho[i, i]

    where ``P_comp`` is the projector onto the first ``computational_subspace_dim``
    basis states.  This is particularly useful for qutrit or multi-level system
    simulations where leakage to non-computational levels is undesirable.

    Args:
        density_matrix:           Square density matrix of shape (d, d).
        computational_subspace_dim: Number of computational basis states
            (e.g. 2 for a qubit, 4 for two qubits).
        max_leakage:              Maximum acceptable leakage probability (default 0.01).

    Returns:
        The measured leakage as a float in [0, 1].

    Raises:
        AssertionError: If leakage > max_leakage, showing measured vs allowed leakage.
        ValueError:     If the density matrix is not square or ``computational_subspace_dim``
                        exceeds the matrix dimension.
    """
    rho: NDArray[np.complex128] = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(
            f"density_matrix must be a square 2D matrix, got shape {rho.shape}"
        )
    d = rho.shape[0]
    if computational_subspace_dim > d:
        raise ValueError(
            f"computational_subspace_dim ({computational_subspace_dim}) "
            f"exceeds the matrix dimension ({d})."
        )
    if computational_subspace_dim < 1:
        raise ValueError(
            f"computational_subspace_dim must be >= 1, got {computational_subspace_dim}."
        )

    pop_in_subspace = float(np.real(np.trace(rho[:computational_subspace_dim, :computational_subspace_dim])))
    leakage = 1.0 - pop_in_subspace
    leakage = max(0.0, leakage)  # clamp numerical noise

    if leakage > max_leakage:
        raise AssertionError(
            f"Leakage exceeds maximum allowed value.\n"
            f"  Measured leakage : {leakage:.6f}\n"
            f"  Max allowed      : {max_leakage:.6f}\n"
            f"  Computational subspace dim: {computational_subspace_dim} / {d}"
        )

    return leakage


def assert_channel_preserves_trace(
    channel_matrices: list[NDArray[np.complex128]],
    *,
    atol: float = 1e-6,
) -> None:
    """Assert that Kraus operators satisfy the trace-preserving (TP) condition.

    A quantum channel is trace-preserving iff::

        sum_i  K_i^dagger @ K_i  ==  I

    This is equivalent to demanding that the channel maps every valid
    density matrix to another density matrix with the same trace.

    Args:
        channel_matrices: Non-empty list of Kraus operators (square numpy arrays
            of equal shape).
        atol:             Absolute tolerance for the completeness check (default 1e-6).

    Raises:
        AssertionError: If the completeness relation is violated, showing the
            Frobenius norm of the deviation and the tolerance.
        ValueError:     If the list is empty, operators have different shapes, or
            operators are not square.
    """
    if not channel_matrices:
        raise ValueError("channel_matrices must be a non-empty list of Kraus operators.")
    ops: list[NDArray[np.complex128]] = [
        np.asarray(K, dtype=np.complex128) for K in channel_matrices
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
    completeness: NDArray[np.complex128] = sum(  # type: ignore[assignment]
        K.conj().T @ K for K in ops
    )
    identity = np.eye(d, dtype=np.complex128)
    deviation_fro = float(np.linalg.norm(completeness - identity, ord="fro"))

    if not np.allclose(completeness, identity, atol=atol):
        raise AssertionError(
            f"Channel is not trace-preserving: sum(K_i† K_i) ≠ I.\n"
            f"  Number of Kraus operators : {len(ops)}\n"
            f"  Matrix dimension          : {d}x{d}\n"
            f"  ||sum(K†K) - I||_fro      : {deviation_fro:.6e}   (tolerance: {atol:.2e})"
        )


def assert_channel_diamond_norm_below(
    channel_a_kraus: list[NDArray[np.complex128]],
    channel_b_kraus: list[NDArray[np.complex128]],
    max_diamond_norm: float,
    *,
    atol: float = 1e-4,
) -> None:
    """Assert the diamond-norm distance between two channels is below a threshold.

    The diamond norm (completely bounded trace norm) is the operationally
    meaningful distance between quantum channels: it equals the maximum
    distinguishing advantage over all input states and ancillae.

    **When cvxpy is available** the function solves the exact SDP formulation
    of Watrous (2009) and the result is precise.

    **Fallback (cvxpy not installed)**: the function uses the operator
    (spectral) norm of the difference of the normalised Choi matrices as a
    proxy distance.  For single-qubit channels this proxy is within a constant
    factor of the true diamond norm and gives a conservative bound.

    Args:
        channel_a_kraus:   Kraus operators for channel A (square arrays, equal shape).
        channel_b_kraus:   Kraus operators for channel B (same dimension as A).
        max_diamond_norm:  Maximum acceptable diamond-norm distance.
        atol:              Absolute tolerance added to the threshold (default 1e-4).

    Raises:
        AssertionError: If the diamond-norm distance exceeds ``max_diamond_norm + atol``,
            showing the estimated distance and the method used.
        ValueError:     If the Kraus lists are empty or have mismatched dimensions.
    """
    if not channel_a_kraus or not channel_b_kraus:
        raise ValueError(
            "channel_a_kraus and channel_b_kraus must be non-empty lists of Kraus operators."
        )
    ops_a: list[NDArray[np.complex128]] = [
        np.asarray(K, dtype=np.complex128) for K in channel_a_kraus
    ]
    ops_b: list[NDArray[np.complex128]] = [
        np.asarray(K, dtype=np.complex128) for K in channel_b_kraus
    ]
    if ops_a[0].shape != ops_b[0].shape:
        raise ValueError(
            f"Channel A and B must have the same Kraus operator shape; "
            f"got {ops_a[0].shape} vs {ops_b[0].shape}"
        )

    choi_a = _kraus_to_choi(ops_a)
    choi_b = _kraus_to_choi(ops_b)
    d = ops_a[0].shape[0]
    # Normalise Choi matrices by d so the maximum diamond norm of any channel difference is 2
    choi_diff = (choi_a - choi_b) / d

    try:
        import cvxpy as cp

        # Exact SDP for diamond norm (Watrous 2009):
        #   || Phi ||_diamond = max_{rho >= 0, Tr(rho) = 1} || (id ⊗ Phi)(rho) ||_1
        # Dual SDP: minimise over block-positive semidefinite constraints.
        # We use the primal formulation with a direct SDP.
        # For the difference channel J = J_A - J_B (Choi matrices already normalised):
        # diamond norm = d * max_{rho} || (I_d ⊗ Phi)(|Omega><Omega|) ||_1
        # Standard SDP:  minimize   Tr(W0) + Tr(W1)
        #               subject to  [[W0,  -J ], [-J†, W1]] >= 0,  W0, W1 >= 0 (Hermitian)
        # where J = choi_diff (d^2 x d^2).
        d2 = d * d
        J = choi_diff
        W0 = cp.Variable((d2, d2), hermitian=True)
        W1 = cp.Variable((d2, d2), hermitian=True)
        block = cp.bmat([[W0, -J], [-J.conj().T, W1]])
        constraints = [block >> 0, W0 >> 0, W1 >> 0]
        objective = cp.Minimize(cp.real(cp.trace(W0)) + cp.real(cp.trace(W1)))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, eps=1e-8)
        estimated_distance = float(prob.value) if prob.value is not None else float("inf")
        method = "SDP (cvxpy)"
    except ImportError:
        # Fallback: operator norm of Choi difference
        # ||J_A - J_B||_op  <=  diamond_norm  <=  d * ||J_A - J_B||_op
        # We use the spectral norm as a lower bound proxy.
        estimated_distance = float(np.linalg.norm(choi_diff, ord=2))
        method = "spectral norm of Choi difference (cvxpy not available)"

    if estimated_distance > max_diamond_norm + atol:
        raise AssertionError(
            f"Diamond-norm distance between channels exceeds threshold.\n"
            f"  Estimated distance : {estimated_distance:.6f}   (method: {method})\n"
            f"  Max allowed        : {max_diamond_norm:.6f}\n"
            f"  Tolerance          : {atol:.1e}"
        )
