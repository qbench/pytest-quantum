"""Integration tests for Pytket support.

These tests require pytket to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytket = pytest.importorskip("pytket", reason="pytket not installed")

from pytket.circuit import Circuit

from pytest_quantum import (
    assert_circuit_depth,
    assert_circuit_is_clifford,
    assert_circuit_width,
    assert_circuits_equivalent,
    assert_gate_count,
    assert_unitary,
)
from pytest_quantum.converters.to_unitary import to_unitary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pytket_h():
    c = Circuit(1)
    c.H(0)
    return c


@pytest.fixture(scope="module")
def pytket_bell():
    c = Circuit(2)
    c.H(0)
    c.CX(0, 1)
    return c


# ---------------------------------------------------------------------------
# to_unitary
# ---------------------------------------------------------------------------


def test_pytket_to_unitary_h(pytket_h):
    U = to_unitary(pytket_h)
    expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    assert U.shape == (2, 2)
    assert np.allclose(np.abs(U), np.abs(expected), atol=1e-6)


def test_pytket_to_unitary_identity():
    c = Circuit(1)
    U = to_unitary(c)
    assert np.allclose(U, np.eye(2, dtype=complex), atol=1e-6)


def test_pytket_to_unitary_bell(pytket_bell):
    U = to_unitary(pytket_bell)
    assert U.shape == (4, 4)
    # Unitary should be valid: U†U ≈ I
    assert np.allclose(U.conj().T @ U, np.eye(4, dtype=complex), atol=1e-6)


# ---------------------------------------------------------------------------
# assert_unitary
# ---------------------------------------------------------------------------


def test_pytket_assert_unitary_h(pytket_h):
    expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    assert_unitary(pytket_h, expected, allow_global_phase=True)


# ---------------------------------------------------------------------------
# assert_circuits_equivalent (cross-framework)
# ---------------------------------------------------------------------------


def test_pytket_equivalent_qiskit_h(pytket_h):
    pytest.importorskip("qiskit", reason="qiskit not installed")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    assert_circuits_equivalent(pytket_h, qc)


# ---------------------------------------------------------------------------
# Structure assertions
# ---------------------------------------------------------------------------


def test_pytket_circuit_depth(pytket_h):
    assert_circuit_depth(pytket_h, max_depth=1)


def test_pytket_circuit_width(pytket_h):
    assert_circuit_width(pytket_h, expected_qubits=1)


def test_pytket_circuit_width_bell(pytket_bell):
    assert_circuit_width(pytket_bell, expected_qubits=2)


def test_pytket_gate_count_h(pytket_h):
    assert_gate_count(pytket_h, "H", 1)


def test_pytket_gate_count_cx(pytket_bell):
    assert_gate_count(pytket_bell, "CX", 1)


# ---------------------------------------------------------------------------
# Clifford assertions
# ---------------------------------------------------------------------------


def test_pytket_clifford_h_cx(pytket_bell):
    assert_circuit_is_clifford(pytket_bell)


def test_pytket_non_clifford_t_gate():
    c = Circuit(1)
    c.T(0)
    with pytest.raises((AssertionError, Exception)):
        assert_circuit_is_clifford(c)
