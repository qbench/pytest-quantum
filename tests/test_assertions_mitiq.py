"""Tests for pytest_quantum.assertions.mitiq_assertions.

Tests are designed to work WITHOUT mitiq installed (ImportError path),
and with mitiq mocked (via unittest.mock) to test assertion logic.

Real mitiq integration tests are skipped unless mitiq is installed.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_zne(
    unmitigated_value: float = 0.3,
    mitigated_value: float = 0.05,
) -> MagicMock:
    """Return a mock mitiq.zne module."""
    mock_zne = MagicMock()
    mock_zne.RichardsonFactory.return_value = MagicMock()
    mock_zne.LinearFactory.return_value = MagicMock()
    mock_zne.PolyFactory.return_value = MagicMock()
    mock_zne.execute_with_zne.return_value = mitigated_value
    return mock_zne


def _make_mock_mitiq(
    unmitigated_value: float = 0.3,
    mitigated_value: float = 0.05,
) -> MagicMock:
    """Return a mock mitiq top-level module."""
    mock_mitiq = MagicMock()
    mock_zne = _make_mock_zne(unmitigated_value, mitigated_value)
    mock_mitiq.zne = mock_zne
    return mock_mitiq


def _simple_executor(circuit: object) -> float:
    """Executor that always returns 0.3 (noisy)."""
    return 0.3


def _ideal_executor(circuit: object) -> float:
    """Executor that always returns 0.0 (ideal)."""
    return 0.0


# ---------------------------------------------------------------------------
# ImportError path — mitiq not installed
# ---------------------------------------------------------------------------


def test_zne_reduces_error_raises_import_error_without_mitiq() -> None:
    """assert_zne_reduces_error raises ImportError when mitiq is not installed."""
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    with (
        patch.dict(sys.modules, {"mitiq": None, "mitiq.zne": None}),
        pytest.raises(ImportError, match="mitiq"),
    ):
        assert_zne_reduces_error(object(), _simple_executor)


def test_zne_expectation_close_raises_import_error_without_mitiq() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_expectation_close

    with (
        patch.dict(sys.modules, {"mitiq": None, "mitiq.zne": None}),
        pytest.raises(ImportError, match="mitiq"),
    ):
        assert_zne_expectation_close(object(), _simple_executor, expected=0.0)


def test_cdr_reduces_error_raises_import_error_without_mitiq() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_cdr_reduces_error

    with (
        patch.dict(sys.modules, {"mitiq": None, "mitiq.cdr": None}),
        pytest.raises(ImportError, match="mitiq"),
    ):
        assert_cdr_reduces_error(object(), _simple_executor, _ideal_executor)


# ---------------------------------------------------------------------------
# assert_zne_reduces_error — mocked mitiq
# ---------------------------------------------------------------------------


def test_zne_reduces_error_returns_tuple() -> None:
    """assert_zne_reduces_error should return (unmitigated, mitigated)."""
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    mock_mitiq = _make_mock_mitiq(unmitigated_value=0.3, mitigated_value=0.05)

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        result = assert_zne_reduces_error(
            circuit=object(),
            executor=_simple_executor,
        )

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_zne_reduces_error_values_are_floats() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    mock_mitiq = _make_mock_mitiq(unmitigated_value=0.3, mitigated_value=0.05)

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        unmitigated, mitigated = assert_zne_reduces_error(object(), _simple_executor)

    assert isinstance(unmitigated, float)
    assert isinstance(mitigated, float)


def test_zne_reduces_error_default_scale_factors() -> None:
    """With default scale_factors=[1.0, 2.0, 3.0], RichardsonFactory is called."""
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    mock_mitiq = _make_mock_mitiq()

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        assert_zne_reduces_error(object(), _simple_executor)

    mock_mitiq.zne.RichardsonFactory.assert_called_once_with([1.0, 2.0, 3.0])


def test_zne_reduces_error_custom_scale_factors() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    mock_mitiq = _make_mock_mitiq()

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        assert_zne_reduces_error(
            object(), _simple_executor, scale_factors=[1.0, 1.5, 2.0]
        )

    mock_mitiq.zne.RichardsonFactory.assert_called_once_with([1.0, 1.5, 2.0])


def test_zne_reduces_error_linear_factory() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    mock_mitiq = _make_mock_mitiq()

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        assert_zne_reduces_error(
            object(), _simple_executor, noise_extrapolation="linear"
        )

    mock_mitiq.zne.LinearFactory.assert_called_once()


def test_zne_reduces_error_poly2_factory() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    mock_mitiq = _make_mock_mitiq()

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        assert_zne_reduces_error(
            object(), _simple_executor, noise_extrapolation="poly2"
        )

    mock_mitiq.zne.PolyFactory.assert_called_once()


def test_zne_reduces_error_unknown_factory_falls_back_to_richardson() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

    mock_mitiq = _make_mock_mitiq()

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        assert_zne_reduces_error(
            object(),
            _simple_executor,
            noise_extrapolation="nonexistent_method",
        )
    # Falls back to RichardsonFactory
    assert mock_mitiq.zne.RichardsonFactory.call_count >= 1


# ---------------------------------------------------------------------------
# assert_zne_expectation_close — mocked mitiq
# ---------------------------------------------------------------------------


def test_zne_expectation_close_passes_when_within_atol() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_expectation_close

    mock_mitiq = _make_mock_mitiq()
    mock_mitiq.zne.execute_with_zne.return_value = 0.95  # close to expected=1.0

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        # Should not raise
        assert_zne_expectation_close(object(), _simple_executor, expected=1.0, atol=0.1)


def test_zne_expectation_close_passes_exactly_on_boundary() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_expectation_close

    mock_mitiq = _make_mock_mitiq()
    mock_mitiq.zne.execute_with_zne.return_value = 0.9  # exactly atol=0.1 from 1.0

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        assert_zne_expectation_close(object(), _simple_executor, expected=1.0, atol=0.1)


def test_zne_expectation_close_fails_when_outside_atol() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_expectation_close

    mock_mitiq = _make_mock_mitiq()
    mock_mitiq.zne.execute_with_zne.return_value = 0.5  # 0.5 away from 1.0

    with (
        patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}),
        pytest.raises(AssertionError, match="ZNE-mitigated"),
    ):
        assert_zne_expectation_close(object(), _simple_executor, expected=1.0, atol=0.1)


def test_zne_expectation_close_error_message_shows_values() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_expectation_close

    mock_mitiq = _make_mock_mitiq()
    mock_mitiq.zne.execute_with_zne.return_value = 0.5

    with (
        patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}),
        pytest.raises(AssertionError) as exc_info,
    ):
        assert_zne_expectation_close(object(), _simple_executor, expected=1.0, atol=0.1)

    err = str(exc_info.value)
    assert "Unmitigated" in err
    assert "Mitigated" in err
    assert "Expected" in err


def test_zne_expectation_close_default_atol() -> None:
    """Default atol is 0.1."""
    from pytest_quantum.assertions.mitiq_assertions import assert_zne_expectation_close

    mock_mitiq = _make_mock_mitiq()
    mock_mitiq.zne.execute_with_zne.return_value = 0.95

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        assert_zne_expectation_close(object(), _simple_executor, expected=1.0)


# ---------------------------------------------------------------------------
# assert_cdr_reduces_error — mocked mitiq
# ---------------------------------------------------------------------------


def _make_mock_cdr(mitigated_value: float = 0.05) -> MagicMock:
    mock_cdr = MagicMock()
    mock_cdr.execute_with_cdr.return_value = mitigated_value
    return mock_cdr


def test_cdr_reduces_error_returns_tuple() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_cdr_reduces_error

    mock_mitiq = MagicMock()
    mock_cdr = _make_mock_cdr(mitigated_value=0.02)
    mock_mitiq.cdr = mock_cdr

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.cdr": mock_cdr}):
        result = assert_cdr_reduces_error(object(), _simple_executor, _ideal_executor)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_cdr_reduces_error_values_are_floats() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_cdr_reduces_error

    mock_mitiq = MagicMock()
    mock_cdr = _make_mock_cdr(mitigated_value=0.02)
    mock_mitiq.cdr = mock_cdr

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.cdr": mock_cdr}):
        unmitigated, mitigated = assert_cdr_reduces_error(
            object(), _simple_executor, _ideal_executor
        )

    assert isinstance(unmitigated, float)
    assert isinstance(mitigated, float)


def test_cdr_reduces_error_passes_without_atol() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_cdr_reduces_error

    mock_mitiq = MagicMock()
    mock_cdr = _make_mock_cdr(mitigated_value=100.0)  # any mitigated value
    mock_mitiq.cdr = mock_cdr

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.cdr": mock_cdr}):
        # No atol — no assertion on result quality
        assert_cdr_reduces_error(object(), _simple_executor, _ideal_executor)


def test_cdr_reduces_error_passes_within_atol() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_cdr_reduces_error

    mock_mitiq = MagicMock()
    mock_cdr = _make_mock_cdr(mitigated_value=0.05)
    mock_mitiq.cdr = mock_cdr

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.cdr": mock_cdr}):
        # ideal_executor returns 0.0; mitigated=0.05; |0.05 - 0.0| <= 0.1
        assert_cdr_reduces_error(object(), _simple_executor, _ideal_executor, atol=0.1)


def test_cdr_reduces_error_fails_outside_atol() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_cdr_reduces_error

    mock_mitiq = MagicMock()
    mock_cdr = _make_mock_cdr(mitigated_value=0.5)
    mock_mitiq.cdr = mock_cdr

    # ideal_executor returns 0.0; mitigated=0.5; |0.5| > 0.1
    with (
        patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.cdr": mock_cdr}),
        pytest.raises(AssertionError, match="CDR-mitigated"),
    ):
        assert_cdr_reduces_error(object(), _simple_executor, _ideal_executor, atol=0.1)


def test_cdr_reduces_error_calls_execute_with_cdr() -> None:
    from pytest_quantum.assertions.mitiq_assertions import assert_cdr_reduces_error

    mock_mitiq = MagicMock()
    mock_cdr = _make_mock_cdr()
    mock_mitiq.cdr = mock_cdr

    circuit = object()
    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.cdr": mock_cdr}):
        assert_cdr_reduces_error(
            circuit, _simple_executor, _ideal_executor, num_training_circuits=5
        )

    mock_cdr.execute_with_cdr.assert_called_once_with(
        circuit,
        _simple_executor,
        _ideal_executor,
        num_training_circuits=5,
    )


# ---------------------------------------------------------------------------
# assert_mitigation_improves_fidelity — pure numpy (no mitiq needed)
# ---------------------------------------------------------------------------


def _qubit0_dm() -> np.ndarray:
    """Density matrix for |0>."""
    return np.array([[1, 0], [0, 0]], dtype=np.complex128)


def _qubit1_dm() -> np.ndarray:
    """Density matrix for |1>."""
    return np.array([[0, 0], [0, 1]], dtype=np.complex128)


def _mixed_dm(p: float = 0.1) -> np.ndarray:
    """Density matrix (1-p)|0><0| + p|1><1|."""
    return np.array([[1 - p, 0], [0, p]], dtype=np.complex128)


def test_mitigation_improves_fidelity_passes() -> None:
    """Mitigated state closer to ideal than noisy state."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_mitigation_improves_fidelity,
    )

    ideal = _qubit0_dm()
    noisy = _mixed_dm(p=0.2)  # 20% noise: trace distance = 0.2
    mitigated = _mixed_dm(p=0.05)  # 5% noise: trace distance = 0.05

    # mitigated is closer to ideal than noisy — should pass
    assert_mitigation_improves_fidelity(noisy, mitigated, ideal)


def test_mitigation_improves_fidelity_with_statevector_ideal() -> None:
    """ideal_state can be a statevector (1D array)."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_mitigation_improves_fidelity,
    )

    ideal_sv = np.array([1, 0], dtype=np.complex128)  # |0>
    noisy = _mixed_dm(p=0.3)
    mitigated = _mixed_dm(p=0.05)

    assert_mitigation_improves_fidelity(noisy, mitigated, ideal_sv)


def test_mitigation_improves_fidelity_fails_when_worse() -> None:
    """Mitigated state further from ideal than noisy must fail."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_mitigation_improves_fidelity,
    )

    ideal = _qubit0_dm()
    noisy = _mixed_dm(p=0.05)  # 5% noise: close to ideal
    mitigated = _mixed_dm(p=0.2)  # 20% noise: further from ideal

    with pytest.raises(AssertionError, match="did not improve"):
        assert_mitigation_improves_fidelity(noisy, mitigated, ideal)


def test_mitigation_improves_fidelity_with_atol() -> None:
    """atol > 0 requires mitigated to be at least atol better."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_mitigation_improves_fidelity,
    )

    ideal = _qubit0_dm()
    noisy = _mixed_dm(p=0.2)
    mitigated = _mixed_dm(p=0.05)

    # mitigated is 0.15 better — passes with atol=0.1
    assert_mitigation_improves_fidelity(noisy, mitigated, ideal, atol=0.1)


def test_mitigation_improves_fidelity_fails_for_insufficient_improvement() -> None:
    """If improvement is less than atol, should fail."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_mitigation_improves_fidelity,
    )

    ideal = _qubit0_dm()
    noisy = _mixed_dm(p=0.2)
    mitigated = _mixed_dm(p=0.18)  # very small improvement

    with pytest.raises(AssertionError, match="did not improve"):
        assert_mitigation_improves_fidelity(noisy, mitigated, ideal, atol=0.1)


def test_mitigation_improves_fidelity_error_message_shows_trace_distances() -> None:
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_mitigation_improves_fidelity,
    )

    ideal = _qubit0_dm()
    noisy = _mixed_dm(p=0.05)
    mitigated = _mixed_dm(p=0.2)

    with pytest.raises(AssertionError) as exc_info:
        assert_mitigation_improves_fidelity(noisy, mitigated, ideal)

    err = str(exc_info.value)
    assert "T(noisy" in err
    assert "T(mitigated" in err


def test_mitigation_improves_fidelity_same_state_passes() -> None:
    """When noisy == ideal, mitigated can be the same too (0 <= 0)."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_mitigation_improves_fidelity,
    )

    ideal = _qubit0_dm()
    assert_mitigation_improves_fidelity(ideal, ideal, ideal)


# ---------------------------------------------------------------------------
# Real mitiq integration tests (skipped if mitiq not installed)
# ---------------------------------------------------------------------------


_mitiq_available = importlib.util.find_spec("mitiq") is not None


@pytest.mark.skipif(not _mitiq_available, reason="mitiq not installed")
class TestRealMitiq:
    """Integration tests that exercise real mitiq when installed."""

    def test_zne_reduces_error_real(self) -> None:
        pytest.importorskip("mitiq", reason="mitiq not installed")
        pytest.importorskip("qiskit", reason="qiskit not installed")

        from qiskit import QuantumCircuit

        from pytest_quantum.assertions.mitiq_assertions import assert_zne_reduces_error

        qc = QuantumCircuit(1)
        qc.x(0)  # Simple X gate

        call_count = 0

        def counting_executor(circuit: Any) -> float:
            nonlocal call_count
            call_count += 1
            return 0.5  # constant noisy value

        unmitigated, mitigated = assert_zne_reduces_error(
            qc,
            counting_executor,
            scale_factors=[1.0, 2.0, 3.0],
        )
        assert isinstance(unmitigated, float)
        assert isinstance(mitigated, float)
        assert call_count > 0  # executor was called

    def test_zne_expectation_close_real(self) -> None:
        pytest.importorskip("mitiq", reason="mitiq not installed")
        pytest.importorskip("qiskit", reason="qiskit not installed")

        from qiskit import QuantumCircuit

        from pytest_quantum.assertions.mitiq_assertions import (
            assert_zne_expectation_close,
        )

        qc = QuantumCircuit(1)
        qc.id(0)

        def constant_executor(circuit: Any) -> float:
            return 1.0

        # ZNE of a constant function is itself — extrapolation returns ~1.0
        assert_zne_expectation_close(qc, constant_executor, expected=1.0, atol=0.5)


# ---------------------------------------------------------------------------
# assert_pec_reduces_error — mocked mitiq
# ---------------------------------------------------------------------------


def _make_mock_pec(mitigated_value: float = 0.05) -> MagicMock:
    """Return a mock mitiq.pec module."""
    mock_pec = MagicMock()
    mock_pec.execute_with_pec.return_value = mitigated_value
    return mock_pec


def test_pec_reduces_error_returns_tuple() -> None:
    """assert_pec_reduces_error should return (unmitigated, mitigated) as floats."""
    from pytest_quantum.assertions.mitiq_assertions import assert_pec_reduces_error

    mock_mitiq = MagicMock()
    mock_pec = _make_mock_pec(mitigated_value=0.02)
    mock_mitiq.pec = mock_pec

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.pec": mock_pec}):
        result = assert_pec_reduces_error(
            circuit=object(),
            executor=_simple_executor,
            noise_model=object(),
        )

    assert isinstance(result, tuple)
    assert len(result) == 2
    unmitigated, mitigated = result
    assert isinstance(unmitigated, float)
    assert isinstance(mitigated, float)


def test_pec_reduces_error_raises_import_error_without_mitiq() -> None:
    """assert_pec_reduces_error raises ImportError when mitiq is not installed."""
    from pytest_quantum.assertions.mitiq_assertions import assert_pec_reduces_error

    with (
        patch.dict(sys.modules, {"mitiq": None, "mitiq.pec": None}),
        pytest.raises(ImportError, match="mitiq"),
    ):
        assert_pec_reduces_error(object(), _simple_executor, object())


# ---------------------------------------------------------------------------
# assert_pec_expectation_close — mocked mitiq
# ---------------------------------------------------------------------------


def test_pec_expectation_close_passes() -> None:
    """assert_pec_expectation_close passes when mitigated is within atol."""
    from pytest_quantum.assertions.mitiq_assertions import assert_pec_expectation_close

    mock_mitiq = MagicMock()
    mock_pec = _make_mock_pec(mitigated_value=0.95)  # close to expected=1.0
    mock_mitiq.pec = mock_pec

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.pec": mock_pec}):
        # Should not raise; |0.95 - 1.0| = 0.05 <= atol=0.1
        assert_pec_expectation_close(
            object(), _simple_executor, object(), expected=1.0, atol=0.1
        )


def test_pec_expectation_close_fails() -> None:
    """assert_pec_expectation_close raises AssertionError when outside atol."""
    from pytest_quantum.assertions.mitiq_assertions import assert_pec_expectation_close

    mock_mitiq = MagicMock()
    mock_pec = _make_mock_pec(mitigated_value=0.5)  # 0.5 away from expected=1.0
    mock_mitiq.pec = mock_pec

    with (
        patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.pec": mock_pec}),
        pytest.raises(AssertionError) as exc_info,
    ):
        assert_pec_expectation_close(
            object(), _simple_executor, object(), expected=1.0, atol=0.1
        )

    err = str(exc_info.value)
    assert "PEC-mitigated" in err
    assert "Mitigated" in err
    assert "Expected" in err
    assert "Unmitigated" in err


# ---------------------------------------------------------------------------
# assert_error_mitigation_benchmark — mocked mitiq
# ---------------------------------------------------------------------------


def test_error_mitigation_benchmark_all_pass() -> None:
    """All methods within atol — returns dict with all method results."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_error_mitigation_benchmark,
    )

    mock_mitiq = _make_mock_mitiq(mitigated_value=0.05)
    # ideal_executor returns 0.0; mitigated=0.05; |0.05 - 0.0| = 0.05 <= atol=0.1
    mock_mitiq.zne.execute_with_zne.return_value = 0.05

    with patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}):
        results = assert_error_mitigation_benchmark(
            circuit=object(),
            ideal_executor=_ideal_executor,
            noisy_executor=_simple_executor,
            methods=["zne_richardson", "zne_linear"],
            atol=0.1,
        )

    assert isinstance(results, dict)
    assert "zne_richardson" in results
    assert "zne_linear" in results
    for val in results.values():
        assert isinstance(val, float)


def test_error_mitigation_benchmark_some_fail() -> None:
    """When a method exceeds atol, AssertionError with helpful message is raised."""
    from pytest_quantum.assertions.mitiq_assertions import (
        assert_error_mitigation_benchmark,
    )

    mock_mitiq = _make_mock_mitiq()
    # ideal_executor returns 0.0; mitigated=0.5; |0.5 - 0.0| = 0.5 > atol=0.1
    mock_mitiq.zne.execute_with_zne.return_value = 0.5

    with (
        patch.dict(sys.modules, {"mitiq": mock_mitiq, "mitiq.zne": mock_mitiq.zne}),
        pytest.raises(AssertionError) as exc_info,
    ):
        assert_error_mitigation_benchmark(
            circuit=object(),
            ideal_executor=_ideal_executor,
            noisy_executor=_simple_executor,
            methods=["zne_richardson", "zne_linear"],
            atol=0.1,
        )

    err = str(exc_info.value)
    assert "benchmark failed" in err
    # Both methods should be reported as failing
    assert "zne_richardson" in err
    assert "zne_linear" in err
