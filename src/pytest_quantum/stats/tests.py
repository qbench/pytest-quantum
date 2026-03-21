"""Statistical test primitives for quantum output validation.

All functions are pure numpy/scipy — no quantum SDK required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------


def fidelity(
    psi: NDArray[np.complex128],
    phi: NDArray[np.complex128],
) -> float:
    """Pure-state fidelity :math:`|\\langle\\psi|\\phi\\rangle|^2`.

    Both arrays are flattened and normalised before computation, so minor
    normalisation errors from simulators do not affect the result.

    Args:
        psi: First statevector, any shape.  Will be flattened.
        phi: Second statevector, same number of elements as *psi*.

    Returns:
        Float in ``[0.0, 1.0]``.  ``1.0`` means the states are identical
        (up to global phase).  ``0.0`` means orthogonal.

    Raises:
        ValueError: If *psi* and *phi* have different sizes.

    Example::

        >>> import numpy as np
        >>> zero = np.array([1, 0], dtype=complex)
        >>> plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        >>> fidelity(zero, plus)
        0.5
    """
    psi_flat = np.asarray(psi, dtype=np.complex128).flatten()
    phi_flat = np.asarray(phi, dtype=np.complex128).flatten()
    if psi_flat.size != phi_flat.size:
        raise ValueError(
            f"Statevectors must have the same size. "
            f"Got {psi_flat.size} and {phi_flat.size}."
        )
    norm_psi = np.linalg.norm(psi_flat)
    norm_phi = np.linalg.norm(phi_flat)
    if norm_psi < 1e-12 or norm_phi < 1e-12:
        raise ValueError("Cannot compute fidelity of a zero-norm statevector.")
    psi_flat = psi_flat / norm_psi
    phi_flat = phi_flat / norm_phi
    return float(abs(np.vdot(psi_flat, phi_flat)) ** 2)


# ---------------------------------------------------------------------------
# Total Variation Distance
# ---------------------------------------------------------------------------


def tvd(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> float:
    """Total Variation Distance between two probability distributions.

    .. math::

        \\text{TVD}(p, q) = \\frac{1}{2} \\sum_x |p(x) - q(x)|

    Args:
        p: First probability distribution (1-D array, sums to 1).
        q: Second probability distribution (1-D array, same shape as *p*).

    Returns:
        Float in ``[0.0, 1.0]``.  ``0.0`` means identical distributions;
        ``1.0`` means disjoint support.

    Example::

        >>> import numpy as np
        >>> tvd(np.array([0.5, 0.5]), np.array([0.6, 0.4]))
        0.1
    """
    p_arr = np.asarray(p, dtype=np.float64).flatten()
    q_arr = np.asarray(q, dtype=np.float64).flatten()
    return float(0.5 * np.sum(np.abs(p_arr - q_arr)))


def tvd_from_counts(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
) -> float:
    """Compute TVD between two shot-count dictionaries.

    Normalises each dict to a probability distribution before computing TVD.

    Args:
        counts_a: First counts dict, e.g. ``{"00": 489, "11": 511}``.
        counts_b: Second counts dict, e.g. ``{"00": 501, "11": 499}``.

    Returns:
        Float in ``[0.0, 1.0]``.

    Raises:
        ValueError: If either dict is empty.
    """
    if not counts_a or not counts_b:
        raise ValueError("Both count dicts must be non-empty.")
    keys = sorted(set(counts_a) | set(counts_b))
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    p = np.array([counts_a.get(k, 0) / total_a for k in keys], dtype=np.float64)
    q = np.array([counts_b.get(k, 0) / total_b for k in keys], dtype=np.float64)
    return tvd(p, q)


# ---------------------------------------------------------------------------
# Chi-square test
# ---------------------------------------------------------------------------


def chi_square_test(
    observed: dict[str, int] | NDArray[np.float64],
    expected_probs: dict[str, float] | NDArray[np.float64],
    total_shots: int | None = None,
) -> tuple[float, float]:
    """Chi-square goodness-of-fit test for quantum measurement distributions.

    Tests whether *observed* counts are consistent with *expected_probs*.

    Args:
        observed:       Either a count dict ``{"00": 489, "11": 511}`` or a
                        1-D numpy array of observed counts.
        expected_probs: Either a probability dict ``{"00": 0.5, "11": 0.5}``
                        (must sum to 1) or a 1-D numpy array of expected
                        probabilities.  When using numpy arrays, provide
                        *total_shots* so expected counts can be computed.
        total_shots:    Required when both inputs are numpy arrays.  Ignored
                        when dict inputs are used (total is inferred from
                        *observed*).

    Returns:
        ``(statistic, pvalue)`` — the chi-square statistic and the p-value.
        Reject the null hypothesis (distributions match) when
        ``pvalue < significance``.

    Raises:
        ValueError: If inputs are inconsistent (different keys, missing
            total_shots for array inputs, etc.).

    Example::

        >>> stat, p = chi_square_test({"00": 495, "11": 505}, {"00": 0.5, "11": 0.5})
        >>> p > 0.05  # consistent with Bell state
        True
    """
    from scipy.stats import chisquare  # lazy — scipy is a hard dependency

    if isinstance(observed, dict) and isinstance(expected_probs, dict):
        keys = sorted(set(observed) | set(expected_probs))
        total = sum(observed.values())
        if total == 0:
            raise ValueError("observed counts sum to zero — no shots recorded.")
        f_obs = np.array([observed.get(k, 0) for k in keys], dtype=np.float64)
        f_exp = np.array(
            [expected_probs.get(k, 0.0) * total for k in keys], dtype=np.float64
        )
    else:
        f_obs = np.asarray(observed, dtype=np.float64).flatten()
        f_exp = np.asarray(expected_probs, dtype=np.float64).flatten()
        if total_shots is not None:
            f_exp = f_exp * total_shots
        if f_obs.shape != f_exp.shape:
            raise ValueError(
                f"observed and expected_probs must have the same shape. "
                f"Got {f_obs.shape} and {f_exp.shape}."
            )

    stat, pvalue = chisquare(f_obs=f_obs, f_exp=f_exp)
    return float(stat), float(pvalue)
