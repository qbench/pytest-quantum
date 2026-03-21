"""pytest-quantum plugin entry point.

Discovered automatically by pytest via the ``pytest11`` entry-point group
declared in ``pyproject.toml``.  Registers markers, CLI options, hooks, and
all framework fixtures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest import Config, Item, Parser


# ---------------------------------------------------------------------------
# Hooks — configuration
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Parser) -> None:
    """Add pytest-quantum command-line options."""
    group = parser.getgroup("quantum", "pytest-quantum options")
    group.addoption(
        "--quantum-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.quantum_slow (skipped by default).",
    )
    group.addoption(
        "--quantum-shots",
        action="store",
        dest="quantum_shots",
        default=None,
        type=int,
        metavar="N",
        help="Override default shot count for all sampling-based tests.",
    )
    group.addoption(
        "--quantum-significance",
        action="store",
        dest="quantum_significance",
        default=None,
        type=float,
        metavar="P",
        help="Override default p-value threshold for statistical assertion tests.",
    )


def pytest_configure(config: Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "quantum: mark test as a quantum test (runs by default with normal suite).",
    )
    config.addinivalue_line(
        "markers",
        "quantum_slow: mark test as requiring many shots — skipped unless "
        "--quantum-slow is passed.",
    )
    config.addinivalue_line(
        "markers",
        "shots(n): override the default shot count for this individual test.",
    )
    config.addinivalue_line(
        "markers",
        "significance(p): override the p-value threshold for this individual test.",
    )


# ---------------------------------------------------------------------------
# Hooks — collection
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Skip quantum_slow tests unless --quantum-slow is supplied."""
    if config.getoption("--quantum-slow", default=False):
        return
    skip_marker = pytest.mark.skip(
        reason="Skipping quantum_slow test — pass --quantum-slow to run."
    )
    for item in items:
        if "quantum_slow" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Hooks — reporting
# ---------------------------------------------------------------------------


def pytest_assertrepr_compare(
    config: Config,
    op: str,
    left: object,
    right: object,
) -> list[str] | None:
    """Improved failure messages for numpy array comparisons in quantum tests."""
    try:
        import numpy as np

        if (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and op == "=="
        ):
            l_flat = left.flatten().astype(complex)
            r_flat = right.flatten().astype(complex)
            if l_flat.shape == r_flat.shape:
                f = float(abs(np.vdot(l_flat, r_flat)) ** 2)
                max_diff = float(np.max(np.abs(l_flat - r_flat)))
                return [
                    "Quantum statevector comparison failed:",
                    f"  shapes    : {left.shape} vs {right.shape}",
                    f"  fidelity  : |⟨left|right⟩|² = {f:.6f}  (1.0 = identical)",
                    f"  max |diff|: {max_diff:.2e}",
                ]
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Fixtures — Qiskit / Aer
# ---------------------------------------------------------------------------


def _require(package: str, extra: str) -> None:
    """Raise a helpful ImportError if *package* is not installed."""
    import importlib.util

    if importlib.util.find_spec(package) is None:
        pytest.skip(
            f"{package!r} is not installed. "
            f"Install it with: pip install pytest-quantum[{extra}]"
        )


@pytest.fixture(scope="session")
def aer_simulator() -> object:
    """Session-scoped AerSimulator for shot-based Qiskit tests.

    Example::

        def test_bell(aer_simulator):
            from qiskit import QuantumCircuit, transpile
            qc = QuantumCircuit(2)
            qc.h(0); qc.cx(0, 1); qc.measure_all()
            counts = aer_simulator.run(transpile(qc, aer_simulator), shots=1024) \\
                .result().get_counts()
            assert "00" in counts
    """
    _require("qiskit_aer", "qiskit")
    from qiskit_aer import AerSimulator

    return AerSimulator()


@pytest.fixture(scope="session")
def aer_statevector_simulator() -> object:
    """Session-scoped AerSimulator configured for statevector simulation."""
    _require("qiskit_aer", "qiskit")
    from qiskit_aer import AerSimulator

    return AerSimulator(method="statevector")


@pytest.fixture
def aer_noise_simulator() -> object:
    """Function-scoped AerSimulator with configurable depolarizing noise.

    Returns a factory ``make_simulator(error_rate)`` — call it with the
    single-qubit depolarizing error probability you want to simulate.

    Scope is *function* (not session) because the noise model is
    parameterised per-test.

    Example::

        def test_noisy_bell(aer_noise_simulator):
            from qiskit import QuantumCircuit, transpile
            from pytest_quantum import assert_measurement_distribution

            sim = aer_noise_simulator(error_rate=0.01)
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            counts = sim.run(transpile(qc, sim), shots=2000).result().get_counts()

            # With 1% noise, distribution is still close to Bell — use wider tolerance
            assert_measurement_distribution(
                counts,
                expected_probs={"00": 0.5, "11": 0.5},
                significance=0.001,
            )
    """
    _require("qiskit_aer", "qiskit")

    def make_simulator(error_rate: float = 0.01) -> object:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise_model = NoiseModel()
        # Single-qubit gate error
        single_qubit_error = depolarizing_error(error_rate, 1)
        # Two-qubit gate error (typically ~10x higher)
        two_qubit_error = depolarizing_error(min(error_rate * 10, 1.0), 2)
        noise_model.add_all_qubit_quantum_error(
            single_qubit_error, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "u"]
        )
        noise_model.add_all_qubit_quantum_error(
            two_qubit_error, ["cx", "cz", "cy", "swap"]
        )
        return AerSimulator(noise_model=noise_model)

    return make_simulator


# ---------------------------------------------------------------------------
# Fixtures — Cirq
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cirq_simulator() -> object:
    """Session-scoped cirq.Simulator.

    Example::

        def test_cirq(cirq_simulator):
            import cirq

            q = cirq.LineQubit.range(1)
            circuit = cirq.Circuit(cirq.H(q[0]))
            sv = cirq_simulator.simulate(circuit).final_state_vector
            assert sv.shape == (2,)
    """
    _require("cirq", "cirq")
    import cirq

    return cirq.Simulator()


# ---------------------------------------------------------------------------
# Fixtures — Amazon Braket
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def braket_simulator() -> object:
    """Session-scoped Braket LocalSimulator.

    Example::

        def test_braket(braket_simulator):
            from braket.circuits import Circuit

            circ = Circuit().h(0).cnot(0, 1)
            circ.measure_all()
            counts = braket_simulator.run(circ, shots=1000).result().measurement_counts
            assert "00" in counts
    """
    _require("braket", "braket")
    from braket.devices import LocalSimulator

    return LocalSimulator()


# ---------------------------------------------------------------------------
# Fixtures — Graphix
# ---------------------------------------------------------------------------


class _GraphixBackend:
    """Thin wrapper that runs a graphix ``Pattern`` and returns the statevector."""

    def run_pattern(
        self,
        pattern: object,
        input_state: object | None = None,
    ) -> object:
        """Run *pattern* and return the output statevector as a numpy array.

        Args:
            pattern: A ``graphix.pattern.Pattern`` instance.
            input_state: Optional input state (``graphix.states.BasicStates`` value).
                If ``None``, the default input state is used.

        Returns:
            A 1-D complex numpy array of shape ``(2**n_output_qubits,)``.
        """
        from graphix.simulator import PatternSimulator

        sim: Any = PatternSimulator(pattern=pattern, backend="statevector")  # type: ignore[arg-type]
        if input_state is not None:
            sim.run(input_state=input_state)
        else:
            sim.run()
        return sim.backend.state.flatten()


@pytest.fixture(scope="session")
def graphix_backend() -> _GraphixBackend:
    """Session-scoped backend for running graphix measurement patterns.

    Example::

        def test_graphix(graphix_backend):
            from graphix.transpiler import Circuit

            circuit = Circuit(1)
            circuit.h(0)
            pattern = circuit.transpile().pattern
            state = graphix_backend.run_pattern(pattern)
            assert state.shape == (2,)
    """
    _require("graphix", "graphix")
    return _GraphixBackend()


# ---------------------------------------------------------------------------
# Fixtures — PennyLane
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pennylane_device() -> Callable[..., Any]:
    """Session-scoped factory that creates ``pennylane.device`` instances.

    Returns a callable ``make_device(wires, shots=None)`` so each test can
    request a device with the right number of wires.

    Example::

        def test_pl(pennylane_device):
            import pennylane as qml

            dev = pennylane_device(wires=2)

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(0)
                qml.CNOT([0, 1])
                return qml.state()

            state = circuit()
            assert state.shape == (4,)
    """
    _require("pennylane", "pennylane")
    import pennylane as qml

    def make_device(wires: int, shots: int | None = None) -> object:
        return qml.device("default.qubit", wires=wires, shots=shots)

    return make_device
