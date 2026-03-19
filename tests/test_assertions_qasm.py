"""Tests for assert_qasm_roundtrip and assert_transpilation_preserves_semantics."""

from __future__ import annotations

import pytest

from pytest_quantum import (
    assert_qasm_roundtrip,
    assert_transpilation_preserves_semantics,
)

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
from qiskit import QuantumCircuit

cirq = pytest.importorskip("cirq", reason="cirq not installed")

# Qiskit QASM3 round-trips require qiskit-qasm3-import
try:
    import qiskit_qasm3_import as _qasm3_import  # noqa: F401

    _has_qasm3_import = True
except ImportError:
    _has_qasm3_import = False

skip_no_qasm3 = pytest.mark.skipif(
    not _has_qasm3_import,
    reason="qiskit-qasm3-import not installed",
)


# ---------------------------------------------------------------------------
# Qiskit QASM3 round-trips
# ---------------------------------------------------------------------------


@skip_no_qasm3
def test_qasm_roundtrip_qiskit_h_gate():
    """Single-qubit H gate survives QASM3 round-trip."""
    qc = QuantumCircuit(1)
    qc.h(0)
    assert_qasm_roundtrip(qc)


@skip_no_qasm3
def test_qasm_roundtrip_qiskit_bell_state():
    """Bell-state prep circuit survives QASM3 round-trip."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert_qasm_roundtrip(qc)


@skip_no_qasm3
def test_qasm_roundtrip_qiskit_multi_gate():
    """Multi-gate circuit without parameters survives QASM3 round-trip."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.z(0)
    assert_qasm_roundtrip(qc)


# ---------------------------------------------------------------------------
# Cirq JSON round-trips
# ---------------------------------------------------------------------------


def test_qasm_roundtrip_cirq_h_gate():
    """Cirq H gate survives JSON round-trip."""
    q = cirq.LineQubit.range(1)
    cc = cirq.Circuit(cirq.H(q[0]))
    assert_qasm_roundtrip(cc)


def test_qasm_roundtrip_cirq_bell_state():
    """Cirq Bell-state circuit survives JSON round-trip."""
    q = cirq.LineQubit.range(2)
    cc = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))
    assert_qasm_roundtrip(cc)


def test_qasm_roundtrip_cirq_two_qubit():
    """Cirq CZ gate circuit survives JSON round-trip."""
    q = cirq.LineQubit.range(2)
    cc = cirq.Circuit(cirq.H(q[0]), cirq.CZ(q[0], q[1]))
    assert_qasm_roundtrip(cc)


# ---------------------------------------------------------------------------
# Unsupported framework
# ---------------------------------------------------------------------------


def test_qasm_roundtrip_unsupported_raises():
    """Passing an unsupported object raises NotImplementedError."""

    class FakeCircuit:
        pass

    with pytest.raises(NotImplementedError, match="supports Qiskit and Cirq"):
        assert_qasm_roundtrip(FakeCircuit())


# ---------------------------------------------------------------------------
# assert_transpilation_preserves_semantics
# ---------------------------------------------------------------------------


def test_transpilation_preserves_semantics_bell():
    """Transpiled Bell-state circuit keeps the same unitary."""
    pytest.importorskip(
        "qiskit.providers.fake_provider",
        reason="qiskit fake_provider not available",
    )
    from qiskit.providers.fake_provider import GenericBackendV2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    backend = GenericBackendV2(num_qubits=2)
    assert_transpilation_preserves_semantics(qc, backend)


def test_transpilation_preserves_semantics_non_qiskit_raises():
    """Non-Qiskit input raises NotImplementedError."""

    class FakeCircuit:
        pass

    with pytest.raises(NotImplementedError, match="only supports Qiskit"):
        assert_transpilation_preserves_semantics(FakeCircuit(), object())
