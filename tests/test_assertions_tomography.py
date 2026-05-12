import numpy as np
import pytest

from pytest_quantum.assertions.tomography import (
    assert_process_tomography_close,
    assert_state_tomography_close,
)


class TestAssertStateTomographyClose:
    def test_ideal_zero_state(self):
        measurements = {"X": 0.0, "Y": 0.0, "Z": 1.0}
        expected = np.array([1, 0], dtype=np.complex128)
        assert_state_tomography_close(measurements, expected)

    def test_ideal_plus_state(self):
        measurements = {"X": 1.0, "Y": 0.0, "Z": 0.0}
        expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        assert_state_tomography_close(measurements, expected)

    def test_ideal_y_plus_state(self):
        measurements = {"X": 0.0, "Y": 1.0, "Z": 0.0}
        expected = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
        assert_state_tomography_close(measurements, expected)

    def test_noisy_data_fails(self):
        measurements = {"X": 0.5, "Y": 0.5, "Z": 0.5}
        expected = np.array([1, 0], dtype=np.complex128)
        with pytest.raises(AssertionError, match="trace distance"):
            assert_state_tomography_close(measurements, expected, atol=0.01)

    def test_density_matrix_input(self):
        measurements = {"X": 0.0, "Y": 0.0, "Z": 1.0}
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert_state_tomography_close(measurements, expected)


class TestAssertProcessTomographyClose:
    def test_identity_process(self):
        chi = np.eye(4, dtype=np.complex128) / 4
        assert_process_tomography_close(chi, np.eye(4) / 4)

    def test_non_hermitian_fails(self):
        np.array([[1, 1j], [-1j, 1]], dtype=np.complex128)
        # This is actually Hermitian, so make it non-Hermitian
        chi_bad = np.array([[1, 2j], [1j, 1]], dtype=np.complex128)
        with pytest.raises(AssertionError, match="not Hermitian"):
            assert_process_tomography_close(chi_bad, np.eye(2))

    def test_not_positive_semidefinite_fails(self):
        chi = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        with pytest.raises(AssertionError, match="not positive semidefinite"):
            assert_process_tomography_close(chi, np.eye(2))

    def test_distance_exceeds_tolerance(self):
        chi = np.eye(4, dtype=np.complex128) / 4
        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = 1.0
        with pytest.raises(AssertionError, match="Frobenius distance"):
            assert_process_tomography_close(chi, expected, atol=0.01)
