"""Tests for pytest_quantum.assertions.benchmarking.

All tests use mocked backends — no real quantum hardware required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the entire module if qiskit is not installed
qiskit = pytest.importorskip("qiskit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_props_with_gate(
    gate_name: str,
    qubits: list[int],
    error_rate: float,
) -> MagicMock:
    """Return a mock BackendProperties with one gate entry."""
    param = MagicMock()
    param.name = "gate_error"
    param.value = error_rate

    gate = MagicMock()
    gate.gate = gate_name
    gate.qubits = qubits
    gate.parameters = [param]

    props = MagicMock()
    props.gates = [gate]
    # Also wire up the convenience method
    props.gate_error.side_effect = lambda g, q: (
        error_rate if g == gate_name and list(q) == qubits else None
    )
    return props


def _make_backend_with_props(
    gate_name: str,
    qubits: list[int],
    error_rate: float,
    *,
    name: str = "mock_backend",
    num_qubits: int = 5,
) -> MagicMock:
    """Return a mock backend with calibration properties."""
    backend = MagicMock()
    backend.name = name
    backend.num_qubits = num_qubits
    backend.properties.return_value = _make_props_with_gate(
        gate_name, qubits, error_rate
    )
    return backend


def _make_simple_backend(
    counts: dict[str, int], *, name: str = "mock_backend"
) -> MagicMock:
    """Return a minimal mock backend whose run() always returns *counts*."""
    job = MagicMock()
    result = MagicMock()
    result.get_counts.return_value = counts
    job.result.return_value = result

    backend = MagicMock()
    backend.name = name
    backend.num_qubits = 5
    backend.run.return_value = job
    backend.properties.return_value = None
    # Ensure dt attribute returns a real float so _get_backend_dt works
    backend.dt = 2.2222e-10
    return backend


# ---------------------------------------------------------------------------
# assert_gate_fidelity_above — passes
# ---------------------------------------------------------------------------


def test_gate_fidelity_passes_good_fidelity() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    backend = _make_backend_with_props("cx", [0, 1], error_rate=0.005)
    fidelity = assert_gate_fidelity_above(backend, "cx", [0, 1], target_fidelity=0.99)
    assert abs(fidelity - 0.995) < 1e-9


def test_gate_fidelity_returns_correct_value() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    backend = _make_backend_with_props("x", [0], error_rate=0.001)
    fidelity = assert_gate_fidelity_above(backend, "x", [0], target_fidelity=0.99)
    assert abs(fidelity - 0.999) < 1e-9


# ---------------------------------------------------------------------------
# assert_gate_fidelity_above — fails
# ---------------------------------------------------------------------------


def test_gate_fidelity_fails_bad_fidelity() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    backend = _make_backend_with_props("cx", [0, 1], error_rate=0.05)
    with pytest.raises(AssertionError, match="Gate fidelity below target"):
        assert_gate_fidelity_above(backend, "cx", [0, 1], target_fidelity=0.99)


def test_gate_fidelity_fail_message_includes_backend_name() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    backend = _make_backend_with_props(
        "cx", [0, 1], error_rate=0.05, name="ibm_sherbrooke"
    )
    with pytest.raises(AssertionError, match="ibm_sherbrooke"):
        assert_gate_fidelity_above(backend, "cx", [0, 1], target_fidelity=0.99)


def test_gate_fidelity_fail_message_includes_gate_and_qubits() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    backend = _make_backend_with_props("ecr", [2, 3], error_rate=0.08)
    with pytest.raises(AssertionError, match="ecr"):
        assert_gate_fidelity_above(backend, "ecr", [2, 3], target_fidelity=0.99)


# ---------------------------------------------------------------------------
# assert_gate_fidelity_above — missing gate
# ---------------------------------------------------------------------------


def test_gate_fidelity_no_properties_raises() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    backend = MagicMock()
    backend.name = "no_props_backend"
    backend.properties.return_value = None

    with pytest.raises(AssertionError, match="no calibration properties"):
        assert_gate_fidelity_above(backend, "cx", [0, 1], target_fidelity=0.99)


def test_gate_fidelity_unknown_gate_raises() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    # Props exist but the requested gate is absent
    props = MagicMock()
    props.gates = []
    props.gate_error.side_effect = Exception("gate not found")

    backend = MagicMock()
    backend.name = "sparse_backend"
    backend.properties.return_value = props

    with pytest.raises(AssertionError, match="not found in calibration data"):
        assert_gate_fidelity_above(backend, "zz", [0, 1], target_fidelity=0.99)


def test_gate_fidelity_qubit_mismatch_raises() -> None:
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    # Gate exists for [0, 1] but we ask for [1, 2]
    backend = _make_backend_with_props("cx", [0, 1], error_rate=0.005)
    # Make the gate_error method return None for the wrong qubits
    backend.properties.return_value.gate_error.side_effect = lambda g, q: (
        0.005 if g == "cx" and list(q) == [0, 1] else None
    )

    with pytest.raises(AssertionError, match="not found in calibration data"):
        assert_gate_fidelity_above(backend, "cx", [1, 2], target_fidelity=0.99)


# ---------------------------------------------------------------------------
# assert_quantum_volume — passes (mocked high-HOP sampler)
# ---------------------------------------------------------------------------


def _make_qv_backend_passing(width: int, shots: int = 1024) -> MagicMock:
    """Return a mock backend where every bitstring maps to the heavy set."""
    # We'll patch _compute_heavy_outputs to return all integers,
    # so all shots land in the heavy set → HOP = 1.0 > 2/3.
    counts = {bin(i)[2:].zfill(width): shots // (2**width) for i in range(2**width)}
    return _make_simple_backend(counts)


def test_quantum_volume_passes_with_high_hop() -> None:
    from pytest_quantum.assertions.benchmarking import assert_quantum_volume

    backend = _make_qv_backend_passing(width=1, shots=1024)

    # Patch _compute_heavy_outputs so all outputs are "heavy".
    # Use low confidence (0.5) so the binomial test passes with few mock trials.
    with (
        patch(
            "pytest_quantum.assertions.benchmarking._compute_heavy_outputs",
            return_value={0, 1},
        ),
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
    ):
        qv = assert_quantum_volume(
            backend, target_qv=2, num_trials=5, shots=1024, confidence=0.5
        )

    assert qv >= 2


def test_quantum_volume_returns_int() -> None:
    from pytest_quantum.assertions.benchmarking import assert_quantum_volume

    backend = _make_qv_backend_passing(width=1, shots=64)

    with (
        patch(
            "pytest_quantum.assertions.benchmarking._compute_heavy_outputs",
            return_value=set(range(2)),
        ),
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
    ):
        result = assert_quantum_volume(
            backend, target_qv=2, num_trials=3, shots=64, confidence=0.5
        )

    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# assert_quantum_volume — fails (mocked low-HOP sampler)
# ---------------------------------------------------------------------------


def test_quantum_volume_fails_with_low_hop() -> None:
    from pytest_quantum.assertions.benchmarking import assert_quantum_volume

    # All shots on bitstring "0" → HOP depends on heavy set
    backend = _make_simple_backend({"0": 1024})

    # Heavy set = {1} only — so "0" is NOT heavy → HOP = 0
    with (
        patch(
            "pytest_quantum.assertions.benchmarking._compute_heavy_outputs",
            return_value={1},
        ),
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
        pytest.raises(AssertionError, match="Quantum volume check failed"),
    ):
        assert_quantum_volume(backend, target_qv=2, num_trials=10, shots=1024)


def test_quantum_volume_fail_message_includes_hop_rate() -> None:
    from pytest_quantum.assertions.benchmarking import assert_quantum_volume

    backend = _make_simple_backend({"0": 1024})

    with (
        patch(
            "pytest_quantum.assertions.benchmarking._compute_heavy_outputs",
            return_value={1},
        ),
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
        pytest.raises(AssertionError, match="Heavy output rate"),
    ):
        assert_quantum_volume(backend, target_qv=2, num_trials=5, shots=1024)


# ---------------------------------------------------------------------------
# assert_randomized_benchmarking — passes (near-perfect backend)
# ---------------------------------------------------------------------------


def _make_rb_perfect_backend(shots: int = 1024) -> MagicMock:
    """Mock backend that always returns all shots in |0⟩."""
    return _make_simple_backend({"0": shots})


def test_rb_passes_near_perfect_backend() -> None:
    from pytest_quantum.assertions.benchmarking import assert_randomized_benchmarking

    backend = _make_rb_perfect_backend(shots=64)

    with patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc):
        result = assert_randomized_benchmarking(
            backend,
            qubit=0,
            clifford_lengths=[1, 5],
            num_sequences=3,
            shots=64,
            min_fidelity_per_clifford=0.9,
        )

    assert result["fidelity"] >= 0.9
    assert "decay_rate" in result
    assert "lengths" in result
    assert "survival_probs" in result


def test_rb_returns_correct_keys() -> None:
    from pytest_quantum.assertions.benchmarking import assert_randomized_benchmarking

    backend = _make_rb_perfect_backend(shots=32)

    with patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc):
        result = assert_randomized_benchmarking(
            backend,
            qubit=0,
            clifford_lengths=[1, 2],
            num_sequences=2,
            shots=32,
            min_fidelity_per_clifford=0.5,
        )

    assert set(result.keys()) == {"fidelity", "decay_rate", "lengths", "survival_probs"}
    assert isinstance(result["fidelity"], float)
    assert isinstance(result["decay_rate"], float)
    assert result["lengths"] == [1, 2]
    assert len(result["survival_probs"]) == 2


# ---------------------------------------------------------------------------
# assert_randomized_benchmarking — fails (noisy backend)
# ---------------------------------------------------------------------------


def _make_rb_noisy_backend(shots: int = 1024) -> MagicMock:
    """Mock backend that returns mostly |1⟩ outcomes (low survival prob)."""
    # Only 10% |0⟩ — very bad fidelity
    return _make_simple_backend({"0": shots // 10, "1": shots - shots // 10})


def test_rb_fails_noisy_backend() -> None:
    from pytest_quantum.assertions.benchmarking import assert_randomized_benchmarking

    backend = _make_rb_noisy_backend(shots=1024)

    with (
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
        pytest.raises(AssertionError, match="Randomized benchmarking fidelity too low"),
    ):
        assert_randomized_benchmarking(
            backend,
            qubit=0,
            clifford_lengths=[1, 5, 10],
            num_sequences=3,
            shots=1024,
            min_fidelity_per_clifford=0.999,
        )


def test_rb_fail_message_includes_fidelity() -> None:
    from pytest_quantum.assertions.benchmarking import assert_randomized_benchmarking

    backend = _make_rb_noisy_backend(shots=64)

    with (
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
        pytest.raises(AssertionError, match="Measured fidelity"),
    ):
        assert_randomized_benchmarking(
            backend,
            qubit=0,
            clifford_lengths=[1, 5],
            num_sequences=2,
            shots=64,
            min_fidelity_per_clifford=0.999,
        )


# ---------------------------------------------------------------------------
# assert_t1_above — mocked backend with T1 property
# ---------------------------------------------------------------------------


def _make_t1_backend(t1_us: float, shots: int = 1024) -> MagicMock:
    """Return a mock backend that simulates T1 decay.

    The mock inspects the circuit to extract the delay duration (in dt units),
    converts it to microseconds (assuming dt = 2.2222e-10 s), then computes
    the survival probability as ``exp(-delay_us / t1_us)``.
    """
    dt_s = 2.2222e-10  # typical IBM dt
    dt_us = dt_s * 1e6

    backend = MagicMock()
    backend.name = "t1_mock_backend"
    backend.num_qubits = 1
    backend.dt = dt_s
    backend.properties.return_value = None

    def _run(circuit: object, **kwargs: object) -> MagicMock:
        # Extract total delay from Delay gates in circuit
        delay_dt = 0
        for instr in getattr(circuit, "data", []):
            op = instr.operation if hasattr(instr, "operation") else instr[0]
            if getattr(op, "name", "") == "delay":
                params = getattr(op, "params", [])
                if params:
                    delay_dt += int(params[0])
        delay_us_local = delay_dt * dt_us
        survival = float(np.exp(-delay_us_local / t1_us)) if t1_us > 0 else 0.0
        n_shots = int(kwargs.get("shots", shots))
        ones = round(survival * n_shots)
        zeros = n_shots - ones
        counts: dict[str, int] = {}
        if ones > 0:
            counts["1"] = ones
        if zeros > 0:
            counts["0"] = zeros

        result = MagicMock()
        result.get_counts.return_value = counts
        job = MagicMock()
        job.result.return_value = result
        return job

    backend.run.side_effect = _run
    return backend


def test_t1_passes_long_t1() -> None:
    from pytest_quantum.assertions.benchmarking import assert_t1_above

    # T1=500µs, target=50µs — large margin to survive shot-rounding noise in fit.
    backend = _make_t1_backend(t1_us=500.0, shots=2048)

    with patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc):
        t1 = assert_t1_above(backend, qubit=0, target_t1_us=50.0, shots=2048)

    assert t1 >= 50.0


def test_t1_returns_float() -> None:
    from pytest_quantum.assertions.benchmarking import assert_t1_above

    backend = _make_t1_backend(t1_us=200.0, shots=512)

    with patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc):
        result = assert_t1_above(backend, qubit=0, target_t1_us=100.0, shots=512)

    assert isinstance(result, float)


def test_t1_fails_short_t1() -> None:
    from pytest_quantum.assertions.benchmarking import assert_t1_above

    # T1 is 5 µs but we require 50 µs → should fail
    backend = _make_t1_backend(t1_us=5.0, shots=1024)

    with (
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
        pytest.raises(AssertionError, match="T1 relaxation time too short"),
    ):
        assert_t1_above(backend, qubit=0, target_t1_us=50.0, shots=1024)


def test_t1_fail_message_includes_measured_t1() -> None:
    from pytest_quantum.assertions.benchmarking import assert_t1_above

    backend = _make_t1_backend(t1_us=5.0, shots=256)

    with (
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
        pytest.raises(AssertionError, match="Measured T1"),
    ):
        assert_t1_above(backend, qubit=0, target_t1_us=50.0, shots=256)


# ---------------------------------------------------------------------------
# IBM backend mocking (patch IBMBackend isinstance check)
# ---------------------------------------------------------------------------


def test_gate_fidelity_does_not_use_ibm_sampler() -> None:
    """assert_gate_fidelity_above is passive — should never call backend.run()."""
    from pytest_quantum.assertions.benchmarking import assert_gate_fidelity_above

    backend = _make_backend_with_props("cx", [0, 1], error_rate=0.002)
    assert_gate_fidelity_above(backend, "cx", [0, 1], target_fidelity=0.99)
    backend.run.assert_not_called()


def test_qv_uses_backend_run_for_non_ibm() -> None:
    """For non-IBM backends assert_quantum_volume should call backend.run()."""
    from pytest_quantum.assertions.benchmarking import assert_quantum_volume

    backend = _make_simple_backend({"0": 64, "1": 64})

    with (
        patch(
            "pytest_quantum.assertions.benchmarking._compute_heavy_outputs",
            return_value={0, 1},
        ),
        patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc),
    ):
        # Should not raise; IBM path should be skipped. Use low confidence for mock.
        assert_quantum_volume(
            backend, target_qv=2, num_trials=3, shots=128, confidence=0.5
        )

    assert backend.run.called


def test_rb_uses_backend_run_for_non_ibm() -> None:
    """For non-IBM backends assert_randomized_benchmarking should call backend.run()."""
    from pytest_quantum.assertions.benchmarking import assert_randomized_benchmarking

    backend = _make_simple_backend({"0": 100})

    with patch("qiskit.transpile", side_effect=lambda qc, *a, **kw: qc):
        assert_randomized_benchmarking(
            backend,
            qubit=0,
            clifford_lengths=[1],
            num_sequences=2,
            shots=100,
            min_fidelity_per_clifford=0.5,
        )

    assert backend.run.called
