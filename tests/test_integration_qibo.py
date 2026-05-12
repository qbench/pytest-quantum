"""Integration tests for Qibo adapter.

All tests are skipped if qibo is not installed.
"""

import numpy as np
import pytest

qibo = pytest.importorskip("qibo")
from qibo import Circuit, gates


class TestQiboAdapter:
    def test_detect(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(1)
        c.add(gates.H(0))
        assert adapter.detect(c)

    def test_detect_rejects_non_qibo(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        assert not adapter.detect("not a circuit")

    def test_to_unitary_h_gate(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(1)
        c.add(gates.H(0))
        u = adapter.to_unitary(c)
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_allclose(u, expected, atol=1e-10)

    def test_to_unitary_bell_state(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(2)
        c.add(gates.H(0))
        c.add(gates.CNOT(0, 1))
        u = adapter.to_unitary(c)
        assert u.shape == (4, 4)
        # Check unitarity
        np.testing.assert_allclose(u @ u.conj().T, np.eye(4), atol=1e-10)

    def test_get_depth(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(2)
        c.add(gates.H(0))
        c.add(gates.CNOT(0, 1))
        depth = adapter.get_depth(c)
        assert depth >= 1

    def test_get_width(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(3)
        assert adapter.get_width(c) == 3

    def test_count_gates(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(2)
        c.add(gates.H(0))
        c.add(gates.H(1))
        c.add(gates.CNOT(0, 1))
        counts = adapter.count_gates(c)
        assert counts.get("H", 0) == 2
        assert counts.get("CNOT", 0) == 1

    def test_gate_names(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(2)
        c.add(gates.H(0))
        c.add(gates.CNOT(0, 1))
        names = adapter.gate_names(c)
        assert "H" in names
        assert "CNOT" in names

    def test_is_clifford_true(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(2)
        c.add(gates.H(0))
        c.add(gates.CNOT(0, 1))
        assert adapter.is_clifford(c)

    def test_is_clifford_false(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        adapter = QiboAdapter()
        c = Circuit(1)
        c.add(gates.T(0))
        assert not adapter.is_clifford(c)

    def test_framework_name(self):
        from pytest_quantum.adapters.qibo import QiboAdapter

        assert QiboAdapter.framework_name == "qibo"


class TestQiboAssertions:
    """Test that existing assertion functions work with Qibo circuits."""

    def test_assert_circuit_depth(self):
        from pytest_quantum.assertions.structure import assert_circuit_depth

        c = Circuit(2)
        c.add(gates.H(0))
        c.add(gates.CNOT(0, 1))
        assert_circuit_depth(c, max_depth=10)

    def test_assert_circuit_width(self):
        from pytest_quantum.assertions.structure import assert_circuit_width

        c = Circuit(3)
        c.add(gates.H(0))
        assert_circuit_width(c, expected_qubits=3)

    def test_assert_gate_count(self):
        from pytest_quantum.assertions.structure import assert_gate_count

        c = Circuit(2)
        c.add(gates.H(0))
        c.add(gates.CNOT(0, 1))
        assert_gate_count(c, "H", 1)
