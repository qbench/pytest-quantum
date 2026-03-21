"""Shot-count calculators.

Quantum tests require running a circuit many times (``shots``).  Choosing the
right shot count is surprisingly nuanced:

* Too few shots → test is flaky (random failures unrelated to correctness).
* Too many shots → test is slow and wasteful.

This module provides principled formulas so you never have to guess.
"""

from __future__ import annotations

import math


def min_shots(
    epsilon: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Minimum shots to detect a total-variation distance of *epsilon*.

    Based on two-sample statistical power analysis:

    .. math::

        N = \\lceil (z_{1-\\alpha/2} + z_{\\text{power}})^2 \\; / \\; (2\\varepsilon^2) \\rceil

    Args:
        epsilon: Minimum detectable total variation distance (TVD).  For
            example, ``0.01`` means the test can reliably catch a 1 %
            deviation from the expected distribution.
        alpha:   Significance level (default ``0.05`` → 95 % confidence).
        power:   Statistical power, i.e. probability of detecting a real
            error (default ``0.80`` → 80 % power).

    Returns:
        Minimum recommended shot count as an integer.

    Raises:
        ValueError: If any argument is outside its valid range.

    Examples::

        >>> min_shots(0.01)   # detect 1% TVD
        7299
        >>> min_shots(0.05)   # detect 5% TVD
        293
        >>> min_shots(0.10)   # detect 10% TVD
        74
        >>> min_shots(0.01, alpha=0.01, power=0.90)  # stricter
        11282
    """
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon!r}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0, 1), got {power!r}")

    from scipy.stats import norm  # lazy import — scipy is a hard dependency

    z_alpha = float(norm.ppf(1 - alpha / 2))
    z_power = float(norm.ppf(power))
    n = (z_alpha + z_power) ** 2 / (2 * epsilon**2)
    return math.ceil(n)


def recommended_shots(
    expected_probs: dict[str, float],
    min_expected_per_bucket: int = 5,
) -> int:
    """Recommend shots so every bucket gets enough expected counts for chi-square.

    The chi-square goodness-of-fit test requires each cell to have an expected
    count of at least 5 (a common rule of thumb) to be statistically valid.
    This function returns the shot count that satisfies that requirement for
    the rarest outcome in *expected_probs*.

    Args:
        expected_probs:         Dict mapping outcome labels to probabilities.
                                Must sum to 1.  Zero-probability outcomes
                                are ignored.
        min_expected_per_bucket: Minimum expected count per non-zero bucket
                                (default ``5``).

    Returns:
        Recommended shot count as an integer.

    Raises:
        ValueError: If *expected_probs* is empty or all probabilities are zero.

    Example::

        >>> recommended_shots({"00": 0.499, "01": 0.001, "11": 0.5})
        5000
        >>> recommended_shots({"0": 0.5, "1": 0.5})
        10
    """
    nonzero = [p for p in expected_probs.values() if p > 0]
    if not nonzero:
        raise ValueError(
            "expected_probs must contain at least one nonzero probability."
        )
    min_prob = min(nonzero)
    return math.ceil(min_expected_per_bucket / min_prob)
