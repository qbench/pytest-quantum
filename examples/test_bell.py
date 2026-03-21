"""
Complete example: testing a Bell state circuit.

Demonstrates:
- assert_measurement_distribution (chi-square test for shot distributions)
- assert_circuit_depth and assert_circuit_width (structure assertions)
- assert_unitary (exact unitary matrix comparison)
- assert_circuits_equivalent (cross-framework comparison)

Run with:
    pip install "pytest-quantum[qiskit,cirq]"
    pytest examples/test_bell.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pytest_quantum import (
    assert_circuit_depth,
    assert_circuit_width,
    assert_circuits_equivalent,
    assert_measurement_distribution,
    assert_unitary,
)

HAS_QISKIT = False
HAS_CIRQ = False
try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    pass
try:
    import cirq  # noqa: F401

    HAS_CIRQ = True
except ImportError:
    pass


BELL_UNITARY = np.array(
    [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, -1, 0],
        [1, 0, 0, -1],
    ],
    dtype=complex,
) / math.sqrt(2)


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
class TestBellQiskit:
    def test_bell_measurement_distribution(self, aer_simulator):
        """Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 should measure 50/50."""
        from qiskit import QuantumCircuit, transpile

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        counts = (
            aer_simulator.run(transpile(qc, aer_simulator), shots=2000)
            .result()
            .get_counts()
        )

        assert_measurement_distribution(
            counts,
            expected_probs={"00": 0.5, "11": 0.5},
        )

    def test_bell_circuit_structure(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        assert_circuit_width(qc, expected_qubits=2)
        assert_circuit_depth(qc, max_depth=3)

    def test_bell_unitary(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        assert_unitary(qc, BELL_UNITARY)


@pytest.mark.skipif(not (HAS_QISKIT and HAS_CIRQ), reason="qiskit and cirq required")
def test_bell_cross_framework_equivalent():
    """Same Bell circuit in Qiskit and Cirq should be equivalent."""
    import cirq
    from qiskit import QuantumCircuit

    qk = QuantumCircuit(2)
    qk.h(0)
    qk.cx(0, 1)

    q = cirq.LineQubit.range(2)
    cc = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))

    assert_circuits_equivalent(qk, cc)
