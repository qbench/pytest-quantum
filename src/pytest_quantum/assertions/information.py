"""Additional information-theoretic distribution assertions."""

from __future__ import annotations

import numpy as np


def assert_hellinger_close(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    *,
    max_distance: float = 0.1,
) -> None:
    """Assert Hellinger distance H(p,q) <= max_distance.

    H(p,q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2
    Range: [0, 1].  H=0 identical, H=1 disjoint support.

    More symmetric and bounded than KL divergence.

    Args:
        counts_a:     First count dictionary.
        counts_b:     Second count dictionary.
        max_distance: Maximum allowed Hellinger distance (default 0.1).

    Raises:
        AssertionError: If H(p,q) > max_distance, with a per-key table.
        ValueError: If both count dictionaries are empty.
    """
    all_keys = sorted(set(counts_a) | set(counts_b))
    if not all_keys:
        raise ValueError("Both count dictionaries are empty.")

    total_a = sum(counts_a.values()) or 1
    total_b = sum(counts_b.values()) or 1

    p = np.array([counts_a.get(k, 0) / total_a for k in all_keys], dtype=np.float64)
    q = np.array([counts_b.get(k, 0) / total_b for k in all_keys], dtype=np.float64)

    hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    if hellinger > max_distance:
        # Build per-key table for the error message
        rows = [
            f"  {'key':<12} {'p(key)':>10} {'q(key)':>10} {'|√p-√q|²':>12}",
            f"  {'-' * 48}",
        ]
        for k, pi, qi in zip(all_keys, p, q, strict=True):
            sq_diff = (float(np.sqrt(pi)) - float(np.sqrt(qi))) ** 2
            rows.append(f"  {k:<12} {pi:>10.4f} {qi:>10.4f} {sq_diff:>12.4e}")
        table = "\n".join(rows)
        raise AssertionError(
            f"Hellinger distance H = {hellinger:.6f} exceeds max_distance "
            f"{max_distance}.\n"
            f"  H = (1/√2) ||√p - √q||₂ ∈ [0, 1]\n"
            f"{table}"
        )


def assert_kl_divergence_below(
    counts: dict[str, int],
    expected_probs: dict[str, float],
    *,
    max_kl: float = 0.1,
) -> None:
    """Assert KL divergence D_KL(observed || expected) <= max_kl.

    D_KL(P||Q) = sum_x P(x) * log2(P(x) / Q(x))

    Note: KL is asymmetric and infinite if Q(x)=0 but P(x)>0.
    Raises ValueError if expected_probs has zero probability for any
    observed outcome.

    Args:
        counts:         Observed count dictionary.
        expected_probs: Expected probability dictionary.
        max_kl:         Maximum allowed KL divergence in bits (default 0.1).

    Raises:
        AssertionError: If D_KL > max_kl.
        ValueError: If any outcome with non-zero observed count has zero
            expected probability (KL would be infinite).
    """
    if not counts:
        return  # vacuously true
    total = sum(counts.values())
    if total == 0:
        return

    kl = 0.0
    for outcome, count in counts.items():
        p = count / total
        if p == 0.0:
            continue
        q = expected_probs.get(outcome, 0.0)
        if q == 0.0:
            raise ValueError(
                f"KL divergence is infinite: outcome {outcome!r} has "
                f"non-zero observed probability ({p:.4f}) but expected_probs "
                f"assigns it zero probability."
            )
        kl += p * float(np.log2(p / q))

    if kl > max_kl:
        raise AssertionError(
            f"KL divergence D_KL(observed||expected) = {kl:.6f} bits "
            f"exceeds max_kl = {max_kl}.\n"
            f"  D_KL = ∑ P(x) log₂(P(x)/Q(x))  (lower = more similar)"
        )


def assert_cross_entropy_below(
    counts: dict[str, int],
    expected_probs: dict[str, float],
    *,
    max_ce: float = 1.0,
) -> None:
    """Assert cross-entropy H(P, Q) = -sum_x P(x) log2 Q(x) <= max_ce.

    Used in quantum supremacy experiments as XEB (cross-entropy benchmarking).

    Args:
        counts:         Observed count dictionary.
        expected_probs: Expected probability dictionary Q.
        max_ce:         Maximum allowed cross-entropy in bits (default 1.0).

    Raises:
        AssertionError: If H(P, Q) > max_ce.
        ValueError: If any observed outcome has zero expected probability.
    """
    if not counts:
        return
    total = sum(counts.values())
    if total == 0:
        return

    cross_entropy = 0.0
    for outcome, count in counts.items():
        p = count / total
        if p == 0.0:
            continue
        q = expected_probs.get(outcome, 0.0)
        if q <= 0.0:
            raise ValueError(
                f"Cross-entropy is undefined: outcome {outcome!r} has "
                f"non-zero observed probability ({p:.4f}) but expected_probs "
                f"assigns it zero/negative probability ({q})."
            )
        cross_entropy += -p * float(np.log2(q))

    if cross_entropy > max_ce:
        raise AssertionError(
            f"Cross-entropy H(P, Q) = {cross_entropy:.6f} bits exceeds "
            f"max_ce = {max_ce}.\n"
            f"  H(P,Q) = -∑ P(x) log₂ Q(x)  (lower = more similar)"
        )
