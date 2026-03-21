"""Shot-distribution assertions for quantum tests.

These assertions test the *statistical* output of a circuit — the probability
distribution over measurement outcomes — using principled chi-square tests
rather than ad-hoc tolerances.
"""

from __future__ import annotations

import warnings

import numpy as np

from pytest_quantum.stats.tests import chi_square_test, tvd_from_counts


def assert_measurement_distribution(
    counts: dict[str, int],
    expected_probs: dict[str, float],
    *,
    significance: float = 0.05,
    min_expected_per_bucket: int = 5,
) -> None:
    """Assert that measured counts match the expected probability distribution.

    Uses a chi-square goodness-of-fit test — the standard statistical tool for
    this exact problem.  The test fails only when the deviation is statistically
    significant (p < *significance*), so occasional random fluctuations do not
    cause false failures.

    Args:
        counts: Measured counts dict, e.g. ``{"00": 489, "11": 511}``.
            Keys are bitstring labels; values are integer counts.
        expected_probs: Expected probability dict, e.g. ``{"00": 0.5, "11": 0.5}``.
            Must sum to 1.0 (within 1e-6).  Outcomes not present
            are assumed to have zero expected probability.
        significance: P-value threshold below which the test fails
            (default ``0.05``).
        min_expected_per_bucket: Chi-square requires expected count >= 5 per
            non-zero cell for valid results.  A ``UserWarning`` is raised
            (but the test does not fail) if this is violated; consider
            increasing shots.

    Raises:
        AssertionError: If ``p_value < significance``, with a per-state
            breakdown of observed vs expected probabilities.
        ValueError: If *expected_probs* does not sum to 1.0, or *counts* is
            empty.

    Example::

        def test_bell_distribution(aer_simulator):
            from qiskit import QuantumCircuit, transpile
            from pytest_quantum import assert_measurement_distribution

            qc = QuantumCircuit(2)
            qc.h(0); qc.cx(0, 1); qc.measure_all()
            qc_t = transpile(qc, aer_simulator)
            counts = aer_simulator.run(qc_t, shots=2000).result().get_counts()

            assert_measurement_distribution(
                counts,
                expected_probs={"00": 0.5, "11": 0.5},
            )
    """
    if not expected_probs:
        raise ValueError("expected_probs cannot be empty.")

    prob_sum = sum(expected_probs.values())
    if not np.isclose(prob_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"expected_probs must sum to 1.0, got {prob_sum:.8f}. "
            "Normalise your expected probabilities."
        )

    total_shots = sum(counts.values())
    if total_shots == 0:
        raise AssertionError("counts dict is empty — no shots were recorded.")

    # Warn if any non-zero bucket has too few expected counts for chi-square
    low_buckets = [
        k
        for k, p in expected_probs.items()
        if 0 < p * total_shots < min_expected_per_bucket
    ]
    if low_buckets:
        warnings.warn(
            f"Some buckets have expected count < {min_expected_per_bucket}: "
            f"{low_buckets}. "
            f"Chi-square may be unreliable — consider increasing shots or "
            f"merging low-probability outcomes.",
            UserWarning,
            stacklevel=2,
        )

    stat, pvalue = chi_square_test(counts, expected_probs)

    if pvalue < significance:
        all_keys = sorted(set(counts) | set(expected_probs))
        rows = []
        for k in all_keys:
            obs_p = counts.get(k, 0) / total_shots
            exp_p = expected_probs.get(k, 0.0)
            diff = obs_p - exp_p
            rows.append(f"    {k:>12s}:  observed {obs_p:.4f}  expected {exp_p:.4f}  diff {diff:+.4f}")
        table = "\n".join(rows)
        raise AssertionError(
            f"Measurement distribution mismatch (chi-square test failed).\n"
            f"  χ² statistic  : {stat:.4f}\n"
            f"  p-value       : {pvalue:.6f}   (threshold: {significance})\n"
            f"  total shots   : {total_shots}\n"
            f"\n"
            f"  Per-state breakdown:\n{table}\n"
            f"\n"
            f"  Hint: if this test is inherently probabilistic and you see "
            f"occasional failures, use @pytest.mark.quantum_slow and increase "
            f"shot count with min_shots(epsilon=0.02) to reduce flakiness."
        )


def assert_counts_close(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    *,
    max_tvd: float = 0.05,
) -> None:
    """Assert that two count dictionaries are statistically close.

    Computes the Total Variation Distance (TVD) between the normalised
    distributions and fails if it exceeds *max_tvd*.

    Useful for comparing two backends, or checking that transpilation has
    not changed a circuit's output distribution.

    Args:
        counts_a: First counts dict.
        counts_b: Second counts dict.
        max_tvd:  Maximum acceptable TVD (default ``0.05``).  TVD of 0 means
            identical distributions; 1 means disjoint support.

    Raises:
        AssertionError: If TVD exceeds *max_tvd*.

    Example::

        def test_transpile_preserves_distribution(aer_simulator):
            from qiskit import QuantumCircuit, transpile
            qc = QuantumCircuit(2)
            qc.h(0); qc.cx(0, 1); qc.measure_all()

            # ideal vs noise-free transpiled
            qc_t = transpile(qc, aer_simulator, optimization_level=3)
            counts_ideal     = aer_simulator.run(qc,   shots=2000).result().get_counts()
            counts_transpiled = aer_simulator.run(qc_t, shots=2000).result().get_counts()
            assert_counts_close(counts_ideal, counts_transpiled, max_tvd=0.05)
    """
    distance = tvd_from_counts(counts_a, counts_b)
    if distance > max_tvd:
        raise AssertionError(
            f"Count distributions differ beyond allowed TVD.\n"
            f"  Total Variation Distance : {distance:.4f}\n"
            f"  Maximum allowed TVD      : {max_tvd}\n"
            f"  Excess                   : {distance - max_tvd:.4f}"
        )
