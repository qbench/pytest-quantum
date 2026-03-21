"""Assertions for Qiskit Primitives (StatevectorSampler / StatevectorEstimator)."""

from __future__ import annotations

from typing import Any

import numpy as np


def assert_sampler_distribution(
    result: Any,
    expected_probs: dict[str, float],
    *,
    pub_idx: int = 0,
    significance: float = 0.05,
) -> None:
    """Assert a Qiskit Sampler result matches expected probability distribution.

    Uses chi-square goodness-of-fit (same as assert_measurement_distribution).

    Args:
        result:         PrimitiveResult from StatevectorSampler.run().
        expected_probs: Expected probability distribution e.g.
                        {"00": 0.5, "11": 0.5}.
        pub_idx:        Which pub result to check (default 0).
        significance:   p-value threshold (default 0.05).

    Raises:
        AssertionError: If distribution doesn't match.

    Example::

        def test_sampler_bell(qiskit_sampler):
            from qiskit.circuit import QuantumCircuit
            from pytest_quantum import assert_sampler_distribution

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            result = qiskit_sampler.run([(qc,)]).result()
            assert_sampler_distribution(result, {"00": 0.5, "11": 0.5})
    """
    from pytest_quantum.assertions.distributions import assert_measurement_distribution

    counts = _extract_sampler_counts(result, pub_idx)
    assert_measurement_distribution(counts, expected_probs, significance=significance)


def assert_estimator_close(
    result: Any,
    expected: float,
    *,
    atol: float = 0.1,
    pub_idx: int = 0,
) -> None:
    """Assert a Qiskit Estimator result is close to the expected value.

    Args:
        result:   PrimitiveResult from StatevectorEstimator.run().
        expected: Expected expectation value.
        atol:     Absolute tolerance (default 0.1).
        pub_idx:  Which pub result to check (default 0).

    Raises:
        AssertionError: If |actual - expected| > atol.

    Example::

        def test_estimator_z(qiskit_estimator):
            from qiskit.circuit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp
            from pytest_quantum import assert_estimator_close

            qc = QuantumCircuit(1)  # |0>, <Z> = 1.0
            obs = SparsePauliOp("Z")
            result = qiskit_estimator.run([(qc, obs)]).result()
            assert_estimator_close(result, expected=1.0, atol=0.01)
    """
    from pytest_quantum.assertions.observables import assert_expectation_value_close

    actual = _extract_estimator_value(result, pub_idx)
    assert_expectation_value_close(actual, expected, atol=atol)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_sampler_counts(result: Any, pub_idx: int) -> dict[str, int]:
    try:
        pub_result = result[pub_idx]
        data = pub_result.data
        creg_name = next(k for k in vars(data) if not k.startswith("_"))
        bit_array = getattr(data, creg_name)
        return dict(bit_array.get_counts())
    except Exception:
        pass
    if hasattr(result, "quasi_dists"):
        qd = result.quasi_dists[pub_idx]
        meta = result.metadata[pub_idx] if hasattr(result, "metadata") else {}
        shots = meta.get("shots", 1024)
        n = meta.get(
            "num_qubits",
            max(len(format(k, "b")) for k in qd if k >= 0) if qd else 1,
        )
        return {
            format(k, f"0{n}b"): int(v * shots)
            for k, v in qd.items()
            if v > 0 and k >= 0
        }
    raise TypeError(
        f"Cannot extract counts from {type(result).__qualname__!r}. "
        "Expected PrimitiveResult from StatevectorSampler."
    )


def _extract_estimator_value(result: Any, pub_idx: int) -> float:
    try:
        pub_result = result[pub_idx]
        return float(np.asarray(pub_result.data.evs).flat[0])
    except Exception:
        pass
    if hasattr(result, "values"):
        return float(np.asarray(result.values)[pub_idx])
    raise TypeError(
        f"Cannot extract expectation value from {type(result).__qualname__!r}. "
        "Expected PrimitiveResult from StatevectorEstimator."
    )
