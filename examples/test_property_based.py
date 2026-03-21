"""
Example: property-based testing with random state generators.

Uses pytest_quantum.random to generate random quantum states and verify
mathematical properties that must always hold.

Run with:
    pip install "pytest-quantum"
    pytest examples/test_property_based.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from pytest_quantum import (
    assert_channel_is_cptp,
    assert_hermitian,
    assert_normalized,
    assert_positive_semidefinite,
)
from pytest_quantum.random import (
    depolarizing_kraus,
    random_density_matrix,
    random_kraus_channel,
    random_statevector,
    random_unitary,
)


class TestRandomStateProperties:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_statevectors_are_normalized(self, seed):
        """Haar random statevectors must have unit norm."""
        sv = random_statevector(2, seed=seed)
        assert_normalized(sv)

    @pytest.mark.parametrize("n_qubits", [1, 2])
    def test_random_density_matrices_are_valid(self, n_qubits):
        """Random density matrices must be Hermitian, PSD, trace=1."""
        rho = random_density_matrix(n_qubits, seed=42)
        assert_hermitian(rho)
        assert_positive_semidefinite(rho)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    @pytest.mark.parametrize("seed", range(3))
    def test_random_unitaries_are_unitary(self, seed):
        """Random unitaries must satisfy U†U = I."""
        U = random_unitary(2, seed=seed)
        product = U.conj().T @ U
        assert np.allclose(product, np.eye(4), atol=1e-10)

    def test_depolarizing_kraus_is_cptp(self):
        """Depolarizing channel must be CPTP for any valid error rate."""
        for p in [0.0, 0.01, 0.1, 0.5, 1.0]:
            kraus = depolarizing_kraus(n_qubits=1, error_rate=p)
            assert_channel_is_cptp(kraus)

    @pytest.mark.parametrize("seed", range(3))
    def test_random_kraus_channel_is_cptp(self, seed):
        """Random Kraus channels must satisfy CPTP completeness relation."""
        kraus = random_kraus_channel(n_qubits=1, n_kraus=4, seed=seed)
        assert_channel_is_cptp(kraus)
