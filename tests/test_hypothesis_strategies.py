"""Tests for Hypothesis strategies (requires hypothesis to be installed)."""

from __future__ import annotations

import importlib.util
import sys
import unittest.mock as mock

import numpy as np
import pytest

HAS_HYPOTHESIS = importlib.util.find_spec("hypothesis") is not None


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestStatevectorStrategy:
    def test_statevectors_are_normalized(self) -> None:
        from hypothesis import given, settings

        from pytest_quantum import assert_normalized
        from pytest_quantum.hypothesis_strategies import statevectors

        @given(statevectors(n_qubits=1))
        @settings(max_examples=20)
        def _inner(sv: np.ndarray) -> None:
            assert_normalized(sv)

        _inner()

    def test_statevectors_correct_shape(self) -> None:
        from hypothesis import given, settings

        from pytest_quantum.hypothesis_strategies import statevectors

        @given(statevectors(n_qubits=2))
        @settings(max_examples=10)
        def _inner(sv: np.ndarray) -> None:
            assert sv.shape == (4,)
            assert sv.dtype == np.complex128

        _inner()


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestDensityMatrixStrategy:
    def test_density_matrices_are_hermitian(self) -> None:
        from hypothesis import given, settings

        from pytest_quantum import assert_hermitian
        from pytest_quantum.hypothesis_strategies import density_matrices

        @given(density_matrices(n_qubits=1))
        @settings(max_examples=20)
        def _inner(rho: np.ndarray) -> None:
            assert_hermitian(rho)

        _inner()

    def test_density_matrices_have_trace_one(self) -> None:
        from hypothesis import given, settings

        from pytest_quantum.hypothesis_strategies import density_matrices

        @given(density_matrices(n_qubits=1))
        @settings(max_examples=20)
        def _inner(rho: np.ndarray) -> None:
            assert abs(np.trace(rho) - 1.0) < 1e-10

        _inner()


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestUnitaryStrategy:
    def test_unitaries_are_unitary(self) -> None:
        from hypothesis import given, settings

        from pytest_quantum.hypothesis_strategies import unitary_matrices

        @given(unitary_matrices(n_qubits=1))
        @settings(max_examples=20)
        def _inner(U: np.ndarray) -> None:
            product = U.conj().T @ U
            assert np.allclose(product, np.eye(2), atol=1e-10)

        _inner()


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestCountDistributionStrategy:
    def test_counts_sum_to_total(self) -> None:
        from hypothesis import given, settings

        from pytest_quantum.hypothesis_strategies import count_distributions

        @given(count_distributions(n_qubits=2, total_shots=500))
        @settings(max_examples=20)
        def _inner(counts: dict[str, int]) -> None:
            assert sum(counts.values()) == 500

        _inner()


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestKrausChannelStrategy:
    def test_kraus_channels_are_cptp(self) -> None:
        from hypothesis import given, settings

        from pytest_quantum import assert_channel_is_cptp
        from pytest_quantum.hypothesis_strategies import kraus_channels

        @given(kraus_channels(n_qubits=1, n_kraus=4))
        @settings(max_examples=10)
        def _inner(kraus: list[np.ndarray]) -> None:
            assert_channel_is_cptp(kraus)

        _inner()


def test_hypothesis_not_installed_raises() -> None:
    """Without hypothesis, strategies raise ImportError."""
    import importlib

    import pytest_quantum.hypothesis_strategies as hs

    with mock.patch.dict(
        sys.modules,
        {
            "hypothesis": None,
            "hypothesis.strategies": None,
            "hypothesis.extra.numpy": None,
        },
    ):
        importlib.reload(hs)
        with pytest.raises(ImportError, match="hypothesis"):
            hs.statevectors(n_qubits=1)

    # Reload with real hypothesis to restore state
    importlib.reload(hs)
