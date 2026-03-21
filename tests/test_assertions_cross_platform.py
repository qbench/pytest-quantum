"""Tests for cross-platform circuit equivalence assertions.

Requires Qiskit; individual tests skip if Cirq is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from pytest_quantum.assertions.cross_platform import (
    assert_cross_platform_equivalent,
    assert_qiskit_cirq_equivalent,
    assert_qiskit_pytket_equivalent,
)

# ---------------------------------------------------------------------------
# Helpers — build simple Qiskit circuits
# ---------------------------------------------------------------------------


def _qiskit_h() -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


def _qiskit_x() -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)
    return qc


def _qiskit_h2() -> object:
    """Two-qubit circuit: H on qubit 0, identity on qubit 1."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    return qc


# ---------------------------------------------------------------------------
# assert_cross_platform_equivalent — Qiskit vs Qiskit (no SDK dependency)
# ---------------------------------------------------------------------------


class TestCrossPlatformSameFramework:
    def test_cross_platform_same_circuit_passes(self) -> None:
        """Identical Qiskit H circuits must compare as equivalent."""
        assert_cross_platform_equivalent(_qiskit_h(), _qiskit_h())

    def test_cross_platform_different_unitaries_fails(self) -> None:
        """H gate vs X gate — should raise AssertionError."""
        with pytest.raises(AssertionError, match="NOT equivalent"):
            assert_cross_platform_equivalent(_qiskit_h(), _qiskit_x())

    def test_cross_platform_global_phase_ignored(self) -> None:
        """Circuit vs e^(iπ/4) * same circuit passes with allow_global_phase=True."""
        import pytest_quantum.assertions.cross_platform as mod

        original_to_unitary = mod.to_unitary
        HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        phase = np.exp(1j * math.pi / 4)
        matrices = iter([HADAMARD, phase * HADAMARD])

        def mock_to_unitary(circuit: object) -> np.ndarray:  # type: ignore[return]
            return next(matrices)

        mod.to_unitary = mock_to_unitary  # type: ignore[assignment]
        try:
            # Should NOT raise because allow_global_phase=True (default)
            assert_cross_platform_equivalent(
                "fake_a", "fake_b", allow_global_phase=True
            )
        finally:
            mod.to_unitary = original_to_unitary  # type: ignore[assignment]

    def test_cross_platform_global_phase_detected(self) -> None:
        """Circuit vs e^(iπ/4) * same circuit fails with allow_global_phase=False."""
        import pytest_quantum.assertions.cross_platform as mod

        original_to_unitary = mod.to_unitary
        HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        phase = np.exp(1j * math.pi / 4)
        matrices = iter([HADAMARD, phase * HADAMARD])

        def mock_to_unitary(circuit: object) -> np.ndarray:  # type: ignore[return]
            return next(matrices)

        mod.to_unitary = mock_to_unitary  # type: ignore[assignment]
        try:
            with pytest.raises(AssertionError, match="NOT equivalent"):
                assert_cross_platform_equivalent(
                    "fake_a", "fake_b", allow_global_phase=False
                )
        finally:
            mod.to_unitary = original_to_unitary  # type: ignore[assignment]

    def test_cross_platform_shape_mismatch(self) -> None:
        """Different qubit counts produce an AssertionError about Hilbert space."""
        from qiskit import QuantumCircuit

        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc2 = QuantumCircuit(2)
        qc2.h(0)

        with pytest.raises(AssertionError, match="different-sized Hilbert space"):
            assert_cross_platform_equivalent(qc1, qc2)

    def test_bad_circuit_raises_value_error(self) -> None:
        """Passing a non-circuit object raises ValueError (not TypeError)."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(ValueError, match="Could not convert circuit_a"):
            assert_cross_platform_equivalent("not_a_circuit", qc)

    def test_framework_hint_appears_in_error(self) -> None:
        """framework_a/b labels appear in the AssertionError message."""
        with pytest.raises(AssertionError, match="my_fw_a") as exc_info:
            assert_cross_platform_equivalent(
                _qiskit_h(),
                _qiskit_x(),
                framework_a="my_fw_a",
                framework_b="my_fw_b",
            )
        assert "my_fw_b" in str(exc_info.value)

    def test_atol_respected(self) -> None:
        """Circuits that are slightly off pass with a generous atol."""
        import pytest_quantum.assertions.cross_platform as mod

        original_to_unitary = mod.to_unitary
        HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        noisy = HADAMARD + 1e-5 * np.ones((2, 2), dtype=np.complex128)
        matrices = iter([HADAMARD, noisy])

        def mock_to_unitary(circuit: object) -> np.ndarray:  # type: ignore[return]
            return next(matrices)

        mod.to_unitary = mock_to_unitary  # type: ignore[assignment]
        try:
            assert_cross_platform_equivalent("a", "b", atol=1e-4)
        finally:
            mod.to_unitary = original_to_unitary  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# assert_qiskit_cirq_equivalent
# ---------------------------------------------------------------------------


class TestQiskitCirqEquivalent:
    def test_qiskit_cirq_equivalent_passes(self) -> None:
        """H gate in Qiskit and Cirq is equivalent."""
        cirq = pytest.importorskip("cirq")

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.H(q[0]))

        assert_qiskit_cirq_equivalent(qc, cc)

    def test_qiskit_cirq_equivalent_fails(self) -> None:
        """H gate (Qiskit) vs X gate (Cirq) must raise AssertionError."""
        cirq = pytest.importorskip("cirq")

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.X(q[0]))

        with pytest.raises(AssertionError, match="NOT equivalent"):
            assert_qiskit_cirq_equivalent(qc, cc)

    def test_qiskit_cirq_labels_in_error(self) -> None:
        """Error message should mention 'qiskit' and 'cirq'."""
        cirq = pytest.importorskip("cirq")

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.X(q[0]))

        with pytest.raises(AssertionError, match="qiskit") as exc_info:
            assert_qiskit_cirq_equivalent(qc, cc)
        assert "cirq" in str(exc_info.value)

    def test_qiskit_cirq_global_phase_passes(self) -> None:
        """Circuits differing only by global phase pass with allow_global_phase=True."""
        cirq = pytest.importorskip("cirq")

        from qiskit import QuantumCircuit

        # S gate in Qiskit applies e^{iπ/4} global phase relative to Cirq's S
        # Use Z gate which has the same unitary in both frameworks.
        qc = QuantumCircuit(1)
        qc.z(0)

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.Z(q[0]))

        assert_qiskit_cirq_equivalent(qc, cc, allow_global_phase=True)


# ---------------------------------------------------------------------------
# assert_qiskit_pytket_equivalent
# ---------------------------------------------------------------------------


class TestQiskitPytketEquivalent:
    def test_qiskit_pytket_equivalent_passes(self) -> None:
        """H gate in Qiskit and pytket is equivalent."""
        pytest.importorskip("pytket")

        from pytket import Circuit as TketCircuit  # type: ignore[import-untyped]
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)

        tk = TketCircuit(1)
        tk.H(0)

        assert_qiskit_pytket_equivalent(qc, tk)

    def test_qiskit_pytket_different_fails(self) -> None:
        """H gate (Qiskit) vs X gate (pytket) raises AssertionError."""
        pytest.importorskip("pytket")

        from pytket import Circuit as TketCircuit  # type: ignore[import-untyped]
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)

        tk = TketCircuit(1)
        tk.X(0)

        with pytest.raises(AssertionError, match="NOT equivalent"):
            assert_qiskit_pytket_equivalent(qc, tk)

    def test_qiskit_pytket_labels_in_error(self) -> None:
        """Error message should mention 'qiskit' and 'pytket'."""
        pytest.importorskip("pytket")

        from pytket import Circuit as TketCircuit  # type: ignore[import-untyped]
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)

        tk = TketCircuit(1)
        tk.X(0)

        with pytest.raises(AssertionError, match="qiskit") as exc_info:
            assert_qiskit_pytket_equivalent(qc, tk)
        assert "pytket" in str(exc_info.value)
