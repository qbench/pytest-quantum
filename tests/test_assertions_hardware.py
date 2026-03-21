"""Tests for pytest_quantum.assertions.hardware.

All tests use mocked backends and jobs — no real quantum hardware required.
Real hardware integration tests should be run with --quantum-real and valid
IBM_QUANTUM_TOKEN credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_job(
    counts: dict[str, int], *, fail: bool = False, cancelled: bool = False
) -> MagicMock:
    """Return a mock Qiskit job that returns *counts* on result()."""
    job = MagicMock()
    job.job_id.return_value = "mock-job-123"

    if fail:
        try:
            from qiskit.providers import JobStatus

            job.status.return_value = JobStatus.ERROR
        except ImportError:
            job.status.return_value = MagicMock(name="ERROR")
    elif cancelled:
        try:
            from qiskit.providers import JobStatus

            job.status.return_value = JobStatus.CANCELLED
        except ImportError:
            job.status.return_value = MagicMock(name="CANCELLED")
    else:
        try:
            from qiskit.providers import JobStatus

            job.status.return_value = JobStatus.DONE
        except ImportError:
            job.status.return_value = MagicMock(name="DONE")

    result_mock = MagicMock()
    result_mock.get_counts.return_value = counts
    job.result.return_value = result_mock
    return job


def _make_backend(
    counts: dict[str, int], *, name: str = "mock_backend", num_qubits: int = 5
) -> MagicMock:
    """Return a mock Qiskit backend."""
    backend = MagicMock()
    backend.name = name
    backend.num_qubits = num_qubits
    backend.run.return_value = _make_job(counts)
    return backend


# ---------------------------------------------------------------------------
# _wait_for_job
# ---------------------------------------------------------------------------


def test_wait_for_job_done_immediately() -> None:
    from pytest_quantum.assertions.hardware import _wait_for_job

    job = _make_job({"0": 500, "1": 500})
    counts = _wait_for_job(job, timeout=10.0)
    assert counts == {"0": 500, "1": 500}


def test_wait_for_job_timeout() -> None:
    from pytest_quantum.assertions.hardware import _wait_for_job

    # Make status always return RUNNING so it times out
    job = MagicMock()
    job.job_id.return_value = "job-timeout"
    try:
        from qiskit.providers import JobStatus

        job.status.return_value = JobStatus.RUNNING
    except ImportError:
        pytest.skip("qiskit not installed")

    with pytest.raises(AssertionError, match="timed out"):
        _wait_for_job(job, timeout=0.01)


def test_wait_for_job_error_status() -> None:
    from pytest_quantum.assertions.hardware import _wait_for_job

    job = _make_job({}, fail=True)
    try:
        from qiskit.providers import JobStatus  # noqa: F401
    except ImportError:
        pytest.skip("qiskit not installed")

    with pytest.raises(AssertionError, match="ERROR"):
        _wait_for_job(job, timeout=10.0)


def test_wait_for_job_cancelled_status() -> None:
    from pytest_quantum.assertions.hardware import _wait_for_job

    job = _make_job({}, cancelled=True)
    try:
        from qiskit.providers import JobStatus  # noqa: F401
    except ImportError:
        pytest.skip("qiskit not installed")

    with pytest.raises(AssertionError, match="CANCELLED"):
        _wait_for_job(job, timeout=10.0)


# ---------------------------------------------------------------------------
# _backend_name / _job_id helpers
# ---------------------------------------------------------------------------


def test_backend_name_string_attr() -> None:
    from pytest_quantum.assertions.hardware import _backend_name

    backend = MagicMock()
    backend.name = "ibm_brisbane"
    assert _backend_name(backend) == "ibm_brisbane"


def test_backend_name_callable_attr() -> None:
    from pytest_quantum.assertions.hardware import _backend_name

    backend = MagicMock()
    backend.name = lambda: "ibm_kyoto"
    assert _backend_name(backend) == "ibm_kyoto"


def test_job_id_callable() -> None:
    from pytest_quantum.assertions.hardware import _job_id

    job = MagicMock()
    job.job_id.return_value = "abc-123"
    assert _job_id(job) == "abc-123"


def test_job_id_no_method() -> None:
    from pytest_quantum.assertions.hardware import _job_id

    class NoIdJob:
        pass

    assert _job_id(NoIdJob()) == "<unknown>"


# ---------------------------------------------------------------------------
# assert_backend_executes
# ---------------------------------------------------------------------------

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

# Patch target: qiskit.transpile as imported inside hardware.py
_TRANSPILE_PATH = "qiskit.transpile"


def _identity_transpile(circuit: object, backend: object, **kwargs: object) -> object:
    """Return circuit unchanged — used to avoid real transpilation in tests."""
    return circuit


def test_backend_executes_returns_counts() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_backend_executes

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    backend = _make_backend({"0": 600, "1": 400})
    with patch(_TRANSPILE_PATH, side_effect=_identity_transpile):
        counts = assert_backend_executes(qc, backend, shots=1000)
    assert counts == {"0": 600, "1": 400}


def test_backend_executes_calls_run_with_shots() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_backend_executes

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    backend = _make_backend({"0": 512, "1": 512})
    assert_backend_executes(qc, backend, shots=1024, transpile=False)
    backend.run.assert_called_once()
    _, kwargs = backend.run.call_args
    assert kwargs.get("shots") == 1024


def test_backend_executes_without_transpile() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_backend_executes

    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)

    backend = _make_backend({"1": 1000})
    counts = assert_backend_executes(qc, backend, shots=1000, transpile=False)
    assert counts["1"] == 1000


# ---------------------------------------------------------------------------
# assert_real_counts_close
# ---------------------------------------------------------------------------


def test_real_counts_close_passes_within_tvd() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_real_counts_close

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    backend = _make_backend({"0": 520, "1": 480})
    with patch(_TRANSPILE_PATH, side_effect=_identity_transpile):
        assert_real_counts_close(
            qc, backend, {"0": 0.5, "1": 0.5}, shots=1000, max_tvd=0.1
        )


def test_real_counts_close_fails_when_tvd_exceeded() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_real_counts_close

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    backend = _make_backend({"0": 1000, "1": 0})
    with (
        patch(_TRANSPILE_PATH, side_effect=_identity_transpile),
        pytest.raises(AssertionError, match="TVD"),
    ):
        assert_real_counts_close(
            qc, backend, {"0": 0.5, "1": 0.5}, shots=1000, max_tvd=0.1
        )


def test_real_counts_close_error_includes_backend_name() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_real_counts_close

    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    backend = _make_backend({"0": 1000}, name="ibm_sherbrooke")
    with (
        patch(_TRANSPILE_PATH, side_effect=_identity_transpile),
        pytest.raises(AssertionError, match="ibm_sherbrooke"),
    ):
        assert_real_counts_close(qc, backend, {"1": 1.0}, shots=1000, max_tvd=0.05)


def test_real_counts_close_returns_counts() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_real_counts_close

    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    backend = _make_backend({"0": 1000})
    with patch(_TRANSPILE_PATH, side_effect=_identity_transpile):
        result = assert_real_counts_close(
            qc, backend, {"0": 1.0}, shots=1000, max_tvd=0.05
        )
    assert result == {"0": 1000}


# ---------------------------------------------------------------------------
# assert_mirror_fidelity
# ---------------------------------------------------------------------------


def test_mirror_fidelity_passes_high_fidelity() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_mirror_fidelity

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    backend = _make_backend({"00": 950, "01": 30, "10": 20})
    with patch(_TRANSPILE_PATH, side_effect=_identity_transpile):
        fidelity = assert_mirror_fidelity(qc, backend, shots=1000, min_fidelity=0.8)
    assert abs(fidelity - 0.95) < 1e-6


def test_mirror_fidelity_fails_low_fidelity() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_mirror_fidelity

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    backend = _make_backend({"00": 200, "01": 400, "10": 300, "11": 100})
    with (
        patch(_TRANSPILE_PATH, side_effect=_identity_transpile),
        pytest.raises(AssertionError, match="Mirror fidelity"),
    ):
        assert_mirror_fidelity(qc, backend, shots=1000, min_fidelity=0.8)


def test_mirror_fidelity_returns_float() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_mirror_fidelity

    qc = QuantumCircuit(1)
    qc.x(0)

    backend = _make_backend({"0": 990, "1": 10})
    with patch(_TRANSPILE_PATH, side_effect=_identity_transpile):
        result = assert_mirror_fidelity(qc, backend, shots=1000, min_fidelity=0.5)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# assert_backend_calibration
# ---------------------------------------------------------------------------


def _make_calibrated_backend(
    readout_errors: list[float],
    gate_errors: list[float],
    *,
    name: str = "mock_backend",
) -> MagicMock:
    backend = MagicMock()
    backend.name = name
    backend.num_qubits = len(readout_errors)

    props = MagicMock()
    props.readout_error.side_effect = lambda q: (
        readout_errors[q] if q < len(readout_errors) else None
    )

    gates = []
    for err in gate_errors:
        gate = MagicMock()
        gate.qubits = [0, 1]
        param = MagicMock()
        param.name = "gate_error"
        param.value = err
        gate.parameters = [param]
        gates.append(gate)
    props.gates = gates

    backend.properties.return_value = props
    return backend


def test_backend_calibration_passes_good_device() -> None:
    from pytest_quantum.assertions.hardware import assert_backend_calibration

    backend = _make_calibrated_backend(
        readout_errors=[0.01, 0.02, 0.01],
        gate_errors=[0.005, 0.007],
    )
    assert_backend_calibration(backend, max_gate_error=0.01, max_readout_error=0.05)


def test_backend_calibration_fails_high_readout_error() -> None:
    from pytest_quantum.assertions.hardware import assert_backend_calibration

    backend = _make_calibrated_backend(
        readout_errors=[0.01, 0.10, 0.01],  # qubit 1 is bad
        gate_errors=[0.005],
    )
    with pytest.raises(AssertionError, match="readout error"):
        assert_backend_calibration(backend, max_readout_error=0.05)


def test_backend_calibration_fails_high_gate_error() -> None:
    from pytest_quantum.assertions.hardware import assert_backend_calibration

    backend = _make_calibrated_backend(
        readout_errors=[0.01, 0.01],
        gate_errors=[0.02],  # exceeds default 0.01
    )
    with pytest.raises(AssertionError, match="gate error"):
        assert_backend_calibration(backend, max_gate_error=0.01)


def test_backend_calibration_no_properties_raises() -> None:
    from pytest_quantum.assertions.hardware import assert_backend_calibration

    backend = MagicMock()
    backend.name = "no_props_backend"
    backend.properties.return_value = None

    with pytest.raises(AssertionError, match="no calibration properties"):
        assert_backend_calibration(backend)


# ---------------------------------------------------------------------------
# assert_circuit_fits_backend
# ---------------------------------------------------------------------------


def test_circuit_fits_backend_passes() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_circuit_fits_backend

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    backend = _make_backend({"00": 500, "11": 500})
    with patch(_TRANSPILE_PATH, side_effect=_identity_transpile):
        info = assert_circuit_fits_backend(qc, backend, max_depth=100, max_2q_gates=10)
    assert "depth" in info
    assert "num_2q_gates" in info
    assert isinstance(info["depth"], int)


def test_circuit_fits_backend_fails_depth() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_circuit_fits_backend

    qc = QuantumCircuit(1)
    for _ in range(20):
        qc.h(0)

    backend = _make_backend({})
    with (
        patch(_TRANSPILE_PATH, side_effect=_identity_transpile),
        pytest.raises(AssertionError, match="depth"),
    ):
        assert_circuit_fits_backend(qc, backend, max_depth=1)


def test_circuit_fits_backend_fails_2q_gates() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_circuit_fits_backend

    qc = QuantumCircuit(2)
    for _ in range(5):
        qc.cx(0, 1)

    backend = _make_backend({})
    with (
        patch(_TRANSPILE_PATH, side_effect=_identity_transpile),
        pytest.raises(AssertionError, match="2Q gates"),
    ):
        assert_circuit_fits_backend(qc, backend, max_2q_gates=2)


def test_circuit_fits_backend_returns_info_dict() -> None:
    from qiskit import QuantumCircuit

    from pytest_quantum.assertions.hardware import assert_circuit_fits_backend

    qc = QuantumCircuit(1)
    qc.h(0)

    backend = _make_backend({})
    with patch(_TRANSPILE_PATH, side_effect=_identity_transpile):
        info = assert_circuit_fits_backend(qc, backend)
    assert set(info.keys()) == {"depth", "num_2q_gates", "num_qubits", "ops"}
