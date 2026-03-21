"""Integration tests for Amazon Braket support.

These tests require amazon-braket-sdk to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("braket", reason="amazon-braket-sdk not installed")

from braket.circuits import Circuit

from pytest_quantum import (
    assert_circuit_depth,
    assert_circuit_is_clifford,
    assert_circuit_width,
    assert_gate_count,
    assert_unitary,
)


@pytest.fixture(scope="module")
def h_circuit():
    return Circuit().h(0)


@pytest.fixture(scope="module")
def bell_circuit():
    return Circuit().h(0).cnot(0, 1)


# ---------------------------------------------------------------------------
# Basic structural assertions
# ---------------------------------------------------------------------------


def test_braket_circuit_depth_h(h_circuit):
    assert_circuit_depth(h_circuit, max_depth=1)


def test_braket_circuit_depth_bell(bell_circuit):
    assert_circuit_depth(bell_circuit, max_depth=2)


def test_braket_circuit_width_single_qubit(h_circuit):
    assert_circuit_width(h_circuit, expected_qubits=1)


def test_braket_circuit_width_bell(bell_circuit):
    assert_circuit_width(bell_circuit, expected_qubits=2)


# ---------------------------------------------------------------------------
# Gate count assertions
# ---------------------------------------------------------------------------


def test_braket_gate_count_h(h_circuit):
    assert_gate_count(h_circuit, "H", 1)


def test_braket_gate_count_cnot(bell_circuit):
    assert_gate_count(bell_circuit, "CNot", 1)


def test_braket_gate_count_h_in_bell(bell_circuit):
    assert_gate_count(bell_circuit, "H", 1)


# ---------------------------------------------------------------------------
# Unitary assertions
# ---------------------------------------------------------------------------


def test_braket_h_gate_unitary(h_circuit):
    expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    assert_unitary(h_circuit, expected)


# ---------------------------------------------------------------------------
# Clifford assertions
# ---------------------------------------------------------------------------


def test_braket_clifford_h_cnot(bell_circuit):
    assert_circuit_is_clifford(bell_circuit)


def test_braket_clifford_s_gate():
    c = Circuit().s(0)
    assert_circuit_is_clifford(c)


def test_braket_non_clifford_t_gate():
    c = Circuit().t(0)
    with pytest.raises(AssertionError, match="non-Clifford"):
        assert_circuit_is_clifford(c)
