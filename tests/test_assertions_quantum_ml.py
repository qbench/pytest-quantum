"""Tests for quantum ML assertions: XEB, expressibility, entanglement capability, barren plateau."""

from __future__ import annotations

import pytest

from pytest_quantum.assertions.quantum_ml import (
    assert_entanglement_capability_above,
    assert_expressibility_above,
    assert_no_barren_plateau,
    assert_xeb_fidelity_above,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")


def _rx_ansatz(params):
    """Simple 1-qubit Rx ansatz."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.rx(float(params[0]), 0)
    return qc


def _bell_ansatz(params):
    """2-qubit entangling ansatz: Ry(θ0) ⊗ Ry(θ1) + CX."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.ry(float(params[0]), 0)
    qc.ry(float(params[1]), 1)
    qc.cx(0, 1)
    qc.ry(float(params[2]), 0)
    qc.ry(float(params[3]), 1)
    return qc


def _product_ansatz(params):
    """2-qubit product state (no entanglement)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.ry(float(params[0]), 0)
    qc.ry(float(params[1]), 1)
    return qc


# ---------------------------------------------------------------------------
# XEB tests
# ---------------------------------------------------------------------------


class TestXEBFidelity:
    def test_ideal_simulator_positive_fidelity(self):
        """AerSimulator (noiseless) should return positive XEB fidelity.

        For Haar-random circuits the theoretical expected F_XEB on a noiseless
        backend is positive (≈0.3–0.8 for small circuits).  We just assert it
        runs without raising when target=0.
        """
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        fidelity = assert_xeb_fidelity_above(
            backend,
            num_qubits=2,
            target_fidelity=0.0,
            num_circuits=5,
            depth=3,
            shots=512,
        )
        assert isinstance(fidelity, float)

    def test_returns_float(self):
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        fidelity = assert_xeb_fidelity_above(
            backend,
            num_qubits=1,
            target_fidelity=0.0,
            num_circuits=3,
            depth=2,
            shots=256,
        )
        assert isinstance(fidelity, float)

    def test_fails_below_threshold(self):
        """Fail gracefully when target is impossibly high."""
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        with pytest.raises(AssertionError, match="XEB fidelity below target"):
            assert_xeb_fidelity_above(
                backend,
                num_qubits=2,
                target_fidelity=999.0,  # impossible
                num_circuits=2,
                depth=2,
                shots=64,
            )

    def test_import_error_without_qiskit(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "qiskit":
                raise ImportError("no qiskit")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="qiskit is required"):
            assert_xeb_fidelity_above(object(), num_qubits=2, target_fidelity=0.9)


# ---------------------------------------------------------------------------
# Expressibility tests
# ---------------------------------------------------------------------------


class TestExpressibility:
    def test_entangling_ansatz_is_expressive(self):
        """A hardware-efficient ansatz with entangling layers should be expressive."""
        score = assert_expressibility_above(
            _bell_ansatz,
            num_qubits=2,
            num_params=4,
            target_expressibility=0.1,
            num_samples=50,
        )
        assert score >= 0.1

    def test_returns_float_in_range(self):
        score = assert_expressibility_above(
            _rx_ansatz,
            num_qubits=1,
            num_params=1,
            target_expressibility=0.0,
            num_samples=30,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fails_below_threshold(self):
        with pytest.raises(AssertionError, match="expressibility below target"):
            assert_expressibility_above(
                _rx_ansatz,
                num_qubits=1,
                num_params=1,
                target_expressibility=0.999,  # impossibly high
                num_samples=20,
            )

    def test_import_error_without_qiskit(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "qiskit.quantum_info" in name or name == "qiskit":
                raise ImportError("no qiskit")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="qiskit is required"):
            assert_expressibility_above(
                lambda p: None, num_qubits=1, num_params=1, target_expressibility=0.5
            )


# ---------------------------------------------------------------------------
# Entanglement capability tests
# ---------------------------------------------------------------------------


class TestEntanglementCapability:
    def test_entangling_circuit_has_capability(self):
        """Bell ansatz should have nonzero entanglement capability."""
        q = assert_entanglement_capability_above(
            _bell_ansatz,
            num_qubits=2,
            num_params=4,
            target_capability=0.1,
            num_samples=30,
        )
        assert q >= 0.1

    def test_product_state_low_entanglement(self):
        """Product ansatz (no CNOT) should have near-zero entanglement capability."""
        q = assert_entanglement_capability_above(
            _product_ansatz,
            num_qubits=2,
            num_params=2,
            target_capability=0.0,  # just verify it doesn't raise
            num_samples=20,
        )
        assert isinstance(q, float)
        assert 0.0 <= q <= 1.0

    def test_fails_below_threshold(self):
        with pytest.raises(AssertionError, match="Meyer-Wallach"):
            assert_entanglement_capability_above(
                _product_ansatz,
                num_qubits=2,
                num_params=2,
                target_capability=0.99,  # product states can't reach this
                num_samples=20,
            )


# ---------------------------------------------------------------------------
# Barren plateau tests
# ---------------------------------------------------------------------------


class TestNoBerrenPlateau:
    def test_simple_ansatz_has_gradients(self):
        """A shallow ansatz should have detectable gradients."""
        var = assert_no_barren_plateau(
            _bell_ansatz,
            num_qubits=2,
            num_params=4,
            num_samples=30,
            min_gradient_variance=1e-10,  # very permissive
        )
        assert isinstance(var, float)
        assert var >= 0.0

    def test_fails_on_impossibly_high_variance(self):
        with pytest.raises(AssertionError, match=r"[Bb]arren plateau"):
            assert_no_barren_plateau(
                _bell_ansatz,
                num_qubits=2,
                num_params=4,
                num_samples=20,
                min_gradient_variance=1e10,  # impossible
            )

    def test_custom_observable(self):
        from qiskit.quantum_info import SparsePauliOp

        obs = SparsePauliOp("ZZ")
        var = assert_no_barren_plateau(
            _bell_ansatz,
            num_qubits=2,
            num_params=4,
            observable=obs,
            num_samples=20,
            min_gradient_variance=0.0,
        )
        assert isinstance(var, float)
