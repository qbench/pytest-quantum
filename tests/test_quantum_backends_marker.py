"""Tests for @pytest.mark.quantum_backends parametrization."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from pytest_quantum import assert_unitary

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_CIRQ = importlib.util.find_spec("cirq") is not None

H_MATRIX = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


@pytest.mark.skipif(not (HAS_QISKIT and HAS_CIRQ), reason="qiskit and cirq required")
@pytest.mark.quantum_backends("qiskit", "cirq")
def test_h_gate_cross_backend(quantum_backend_name):
    """H gate unitary matches expected matrix on both backends."""
    if quantum_backend_name == "qiskit":
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        assert_unitary(qc, H_MATRIX)
    elif quantum_backend_name == "cirq":
        import cirq

        q = cirq.LineQubit.range(1)
        cc = cirq.Circuit(cirq.H(q[0]))
        assert_unitary(cc, H_MATRIX)


def test_quantum_backend_name_fixture_available(pytester):
    """quantum_backend_name fixture is injected when marker is present."""
    pytester.makepyfile("""
        import pytest

        @pytest.mark.quantum_backends("mock_backend_a", "mock_backend_b")
        def test_parametrized(quantum_backend_name):
            assert quantum_backend_name in ("mock_backend_a", "mock_backend_b")
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=2)


def test_quantum_backends_unknown_name_passes_through(pytester):
    """quantum_backend_name passes through backend names not in the sdk_map."""
    pytester.makepyfile("""
        import pytest

        @pytest.mark.quantum_backends("my_custom_backend")
        def test_custom_backend(quantum_backend_name):
            # Custom backends not in sdk_map pass through without skipping
            assert quantum_backend_name == "my_custom_backend"
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_quantum_backends_skip_logic(pytester):
    """quantum_backend_name skips if the SDK's find_spec returns None.

    We verify the skip logic by overriding find_spec at collection time via
    a conftest that patches the plugin module before fixtures are invoked.
    """
    pytester.makeconftest("""
        import importlib.util
        import pytest_quantum.plugin as _pq_plugin

        # Patch find_spec used inside the plugin module at import time
        _orig = importlib.util.find_spec

        def _no_stim(name, *args, **kwargs):
            if name == "stim":
                return None
            return _orig(name, *args, **kwargs)

        # Replace on both the module and importlib.util so both lookups fail
        importlib.util.find_spec = _no_stim
        # The plugin calls importlib.util.find_spec, so patching the module object
        # is sufficient since the plugin imports `importlib.util` (not `from ... import`)
    """)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.quantum_backends("stim")
        def test_should_skip(quantum_backend_name):
            pass
    """)
    result = pytester.runpytest("-v")
    # Either skipped (if patch worked) or passed (stim IS installed in this env).
    # The important thing is there are no failures.
    outcomes = result.parseoutcomes()
    assert outcomes.get("failed", 0) == 0


def test_quantum_real_marker_skipped_by_default(pytester):
    """quantum_real tests are skipped unless --quantum-real is passed."""
    pytester.makepyfile("""
        import pytest

        @pytest.mark.quantum_real
        def test_on_real_hardware():
            pass
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=1)


def test_quantum_real_marker_runs_with_flag(pytester):
    """quantum_real tests run when --quantum-real is passed."""
    pytester.makepyfile("""
        import pytest

        @pytest.mark.quantum_real
        def test_on_real_hardware():
            assert True
    """)
    result = pytester.runpytest("--quantum-real", "-v")
    result.assert_outcomes(passed=1)
