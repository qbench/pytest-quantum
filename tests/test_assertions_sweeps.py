"""Tests for pytest_quantum.assertions.sweeps.

Requires qiskit for circuit construction (skips gracefully if not installed).
Pure-numpy path tests do not require any quantum SDK.
"""

import numpy as np
import pytest

# All tests in this file require qiskit for circuit construction.
qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")


def _qk_rx_circuit(theta: float) -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    return qc


def _qk_rx_expected(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _qk_rz_circuit(phi: float) -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.rz(phi, 0)
    return qc


def _qk_rz_expected(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=np.complex128
    )


def _qk_h_circuit(theta: float) -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


def _qk_h_expected(theta: float) -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def _qk_wrong_expected(theta: float) -> np.ndarray:
    """Deliberately wrong expected unitary."""
    return np.eye(2, dtype=np.complex128)


# ---------------------------------------------------------------------------
# assert_circuit_sweep — happy path
# ---------------------------------------------------------------------------


def test_circuit_sweep_rx_single_angle() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    assert_circuit_sweep(
        _qk_rx_circuit,
        {"theta": [np.pi / 2]},
        _qk_rx_expected,
    )


def test_circuit_sweep_rx_multiple_angles() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    assert_circuit_sweep(
        _qk_rx_circuit,
        {"theta": np.linspace(0, 2 * np.pi, 6)},
        _qk_rx_expected,
    )


def test_circuit_sweep_rz_multiple_angles() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    assert_circuit_sweep(
        _qk_rz_circuit,
        {"phi": [0.0, np.pi / 4, np.pi / 2, np.pi]},
        _qk_rz_expected,
    )


def test_circuit_sweep_h_constant() -> None:
    """H gate is independent of theta — still should pass for any theta values."""
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    assert_circuit_sweep(
        _qk_h_circuit,
        {"theta": [0.0, 1.0, 2.0]},
        _qk_h_expected,
        allow_global_phase=True,
    )


def test_circuit_sweep_global_phase_ignored() -> None:
    """assert_circuit_sweep must pass when circuits differ only by global phase."""
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    # Wrap Rx with an overall e^{i*pi/4} global phase in expected
    def expected_with_phase(theta: float) -> np.ndarray:
        return np.exp(1j * np.pi / 4) * _qk_rx_expected(theta)

    assert_circuit_sweep(
        _qk_rx_circuit,
        {"theta": [np.pi / 3, np.pi]},
        expected_with_phase,
        allow_global_phase=True,
    )


# ---------------------------------------------------------------------------
# assert_circuit_sweep — failure path
# ---------------------------------------------------------------------------


def test_circuit_sweep_fails_for_wrong_unitary() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    with pytest.raises(AssertionError, match="assert_circuit_sweep failed"):
        assert_circuit_sweep(
            _qk_rx_circuit,
            {"theta": [np.pi / 2]},
            _qk_wrong_expected,
        )


def test_circuit_sweep_reports_all_failures() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    with pytest.raises(AssertionError, match="3 /") as exc_info:
        assert_circuit_sweep(
            _qk_rx_circuit,
            {"theta": [0.1, 0.5, 1.0]},
            _qk_wrong_expected,
        )
    # Each failure should mention the param value
    assert "theta" in str(exc_info.value)


def test_circuit_sweep_fails_global_phase_disabled() -> None:
    """When allow_global_phase=False, global-phase-only difference must fail."""
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    def expected_with_phase(theta: float) -> np.ndarray:
        return np.exp(1j * np.pi / 3) * _qk_rx_expected(theta)

    with pytest.raises(AssertionError, match="assert_circuit_sweep failed"):
        assert_circuit_sweep(
            _qk_rx_circuit,
            {"theta": [np.pi / 2]},
            expected_with_phase,
            allow_global_phase=False,
        )


# ---------------------------------------------------------------------------
# assert_circuit_sweep — cartesian product
# ---------------------------------------------------------------------------


def _qk_u_circuit(theta: float, phi: float, lam: float) -> object:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.u(theta, phi, lam, 0)
    return qc


def test_circuit_sweep_cartesian_product() -> None:
    """3 * 2 = 6 combinations should all pass."""
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep

    def expected_u(theta: float, phi: float, lam: float) -> np.ndarray:
        from qiskit.quantum_info import Operator

        qc = _qk_u_circuit(theta, phi, lam)
        return np.asarray(Operator(qc).data, dtype=np.complex128)

    assert_circuit_sweep(
        _qk_u_circuit,
        {"theta": [0.0, np.pi / 2], "phi": [0.0, np.pi], "lam": [0.0]},
        expected_u,
    )


# ---------------------------------------------------------------------------
# assert_circuit_sweep_states — happy path
# ---------------------------------------------------------------------------


def test_circuit_sweep_states_rx() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep_states

    psi0 = np.array([1, 0], dtype=np.complex128)

    def expected_state(theta: float) -> np.ndarray:
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([c, -1j * s], dtype=np.complex128)

    assert_circuit_sweep_states(
        _qk_rx_circuit,
        psi0,
        {"theta": [0.0, np.pi / 4, np.pi / 2, np.pi]},
        expected_state,
        min_fidelity=0.999,
    )


def test_circuit_sweep_states_rz() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep_states

    psi0 = np.array([1, 0], dtype=np.complex128)

    def expected_state(phi: float) -> np.ndarray:
        return np.array([np.exp(-1j * phi / 2), 0], dtype=np.complex128)

    assert_circuit_sweep_states(
        _qk_rz_circuit,
        psi0,
        {"phi": [0.0, np.pi / 2, np.pi, 2 * np.pi]},
        expected_state,
        min_fidelity=0.999,
    )


# ---------------------------------------------------------------------------
# assert_circuit_sweep_states — failure path
# ---------------------------------------------------------------------------


def test_circuit_sweep_states_fails_for_wrong_expected() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep_states

    psi0 = np.array([1, 0], dtype=np.complex128)

    def wrong_expected(theta: float) -> np.ndarray:
        return np.array([0, 1], dtype=np.complex128)  # always |1>, never right

    with pytest.raises(AssertionError, match="assert_circuit_sweep_states failed"):
        assert_circuit_sweep_states(
            _qk_rx_circuit,
            psi0,
            {"theta": [np.pi / 4, np.pi / 2]},
            wrong_expected,
            min_fidelity=0.999,
        )


def test_circuit_sweep_states_reports_fidelity() -> None:
    from pytest_quantum.assertions.sweeps import assert_circuit_sweep_states

    psi0 = np.array([1, 0], dtype=np.complex128)

    def wrong_expected(theta: float) -> np.ndarray:
        return np.array([0, 1], dtype=np.complex128)

    with pytest.raises(AssertionError, match="fidelity="):
        assert_circuit_sweep_states(
            _qk_rx_circuit,
            psi0,
            {"theta": [np.pi / 2]},
            wrong_expected,
            min_fidelity=0.99,
        )


# ---------------------------------------------------------------------------
# assert_parametrized_unitary_continuous — happy path
# ---------------------------------------------------------------------------


def test_parametrized_unitary_continuous_rx() -> None:
    from pytest_quantum.assertions.sweeps import assert_parametrized_unitary_continuous

    assert_parametrized_unitary_continuous(
        _qk_rx_circuit,
        "theta",
        (0, 2 * np.pi),
        n_samples=20,
        max_jump=1.0,
    )


def test_parametrized_unitary_continuous_rz() -> None:
    from pytest_quantum.assertions.sweeps import assert_parametrized_unitary_continuous

    assert_parametrized_unitary_continuous(
        _qk_rz_circuit,
        "phi",
        (0, 2 * np.pi),
        n_samples=16,
        max_jump=1.0,
    )


def test_parametrized_unitary_continuous_constant_h() -> None:
    """H is constant — consecutive jumps are zero."""
    from pytest_quantum.assertions.sweeps import assert_parametrized_unitary_continuous

    assert_parametrized_unitary_continuous(
        _qk_h_circuit,
        "theta",
        (0, np.pi),
        n_samples=5,
        max_jump=0.0,  # zero jump since unitary is constant
    )


# ---------------------------------------------------------------------------
# assert_parametrized_unitary_continuous — failure path
# ---------------------------------------------------------------------------


def test_parametrized_unitary_continuous_detects_jump() -> None:
    """Use a very tight max_jump to force a detection of a valid change."""
    from pytest_quantum.assertions.sweeps import assert_parametrized_unitary_continuous

    with pytest.raises(AssertionError, match="Unitary not continuous"):
        assert_parametrized_unitary_continuous(
            _qk_rx_circuit,
            "theta",
            (0, 2 * np.pi),
            n_samples=5,
            max_jump=0.001,  # extremely tight — will catch normal rotation changes
        )


# ---------------------------------------------------------------------------
# _fmt_params / _count_combos helpers
# ---------------------------------------------------------------------------


def test_fmt_params() -> None:
    from pytest_quantum.assertions.sweeps import _fmt_params

    result = _fmt_params({"theta": 1.5708, "phi": 3.1416})
    assert "theta" in result
    assert "phi" in result


def test_count_combos_single() -> None:
    from pytest_quantum.assertions.sweeps import _count_combos

    assert _count_combos({"theta": [1.0, 2.0, 3.0]}) == 3


def test_count_combos_product() -> None:
    from pytest_quantum.assertions.sweeps import _count_combos

    assert _count_combos({"a": [1.0, 2.0], "b": [3.0, 4.0, 5.0]}) == 6


def test_count_combos_empty() -> None:
    from pytest_quantum.assertions.sweeps import _count_combos

    assert _count_combos({}) == 1
