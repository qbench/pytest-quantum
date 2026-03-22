"""pytest-quantum plugin entry point.

Discovered automatically by pytest via the ``pytest11`` entry-point group
declared in ``pyproject.toml``.  Registers markers, CLI options, hooks, and
all framework fixtures.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from pytest import Config, Item, Metafunc, Parser


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
    group.addoption(
        "--quantum-update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate all pytest-quantum snapshot golden files.",
    )
    group.addoption(
        "--quantum-real",
        action="store_true",
        default=False,
        help="Enable real quantum hardware tests (requires cloud credentials). "
        "Tests marked @pytest.mark.quantum_real are skipped unless this flag is passed.",
    )


def pytest_configure(config: Config) -> None:
    """Register custom markers and handle snapshot update flag."""
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
    config.addinivalue_line(
        "markers",
        "quantum_snapshot: marks snapshot tests (for selective update).",
    )
    config.addinivalue_line(
        "markers",
        "quantum_backends(backends): run test on specified backends.",
    )
    config.addinivalue_line(
        "markers",
        "quantum_real: mark test as requiring real quantum hardware "
        "(run with --quantum-real; skipped by default).",
    )
    # Set env var so snapshot helpers can detect the update flag without
    # needing access to the pytest config object.
    try:
        if config.getoption("--quantum-update-snapshots", default=False):
            os.environ["PYTEST_QUANTUM_UPDATE_SNAPSHOTS"] = "1"
    except (ValueError, AttributeError):
        # getoption raises ValueError if the option is not registered yet
        pass


# ---------------------------------------------------------------------------
# Hooks — collection
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Skip quantum_slow and quantum_real tests unless appropriate flags are supplied."""
    if not config.getoption("--quantum-slow", default=False):
        skip_slow = pytest.mark.skip(
            reason="Skipping quantum_slow test — pass --quantum-slow to run."
        )
        for item in items:
            if "quantum_slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--quantum-real", default=False):
        skip_real = pytest.mark.skip(
            reason="real hardware tests skipped (pass --quantum-real)"
        )
        for item in items:
            if item.get_closest_marker("quantum_real"):
                item.add_marker(skip_real)


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Parametrize tests with @pytest.mark.quantum_backends across backend names."""
    marker = metafunc.definition.get_closest_marker("quantum_backends")
    if marker and "quantum_backend_name" in metafunc.fixturenames:
        backends = list(marker.args)
        if not backends:
            backends = ["qiskit", "cirq", "pennylane", "braket"]
        metafunc.parametrize("quantum_backend_name", backends, scope="function")


# ---------------------------------------------------------------------------
# Hooks — reporting
# ---------------------------------------------------------------------------


def pytest_assertrepr_compare(
    config: Config,
    op: str,
    left: object,
    right: object,
) -> list[str] | None:
    """Enhanced assertion messages for quantum objects."""
    # Qiskit QuantumCircuit comparison
    try:
        from qiskit import QuantumCircuit

        if isinstance(left, QuantumCircuit) and isinstance(right, QuantumCircuit):
            lines: list[str] = [f"QuantumCircuit {op} QuantumCircuit"]
            lines.append("Left circuit:")
            lines.extend(f"  {ln}" for ln in str(left.draw("text")).splitlines())
            lines.append("Right circuit:")
            lines.extend(f"  {ln}" for ln in str(right.draw("text")).splitlines())
            return lines
    except ImportError:
        pass

    # Count dict comparison
    if isinstance(left, dict) and isinstance(right, dict):
        combined: dict[str, object] = {**left, **right}
        if all(isinstance(v, (int, float)) for v in combined.values()):
            from pytest_quantum.stats.tests import tvd_from_counts

            try:
                result_lines: list[str] = ["Quantum count distributions differ:"]
                all_keys = sorted(set(left) | set(right))
                total_l = sum(left.values()) or 1
                total_r = sum(right.values()) or 1
                result_lines.append(
                    f"  {'key':<10} {'left':>10} {'right':>10} {'diff':>10}"
                )
                result_lines.append(f"  {'-' * 42}")
                for k in all_keys:
                    l_count = left.get(k, 0)
                    r_count = right.get(k, 0)
                    diff = l_count / total_l - r_count / total_r
                    result_lines.append(
                        f"  {k:<10} {l_count / total_l:>10.4f} "
                        f"{r_count / total_r:>10.4f} {diff:>+10.4f}"
                    )
                distance = tvd_from_counts(
                    {k: int(v) for k, v in left.items()},
                    {k: int(v) for k, v in right.items()},
                )
                result_lines.append(f"  TVD = {distance:.4f}")
                return result_lines
            except Exception:
                pass

    # Numpy array comparison (original behaviour)
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


@pytest.fixture(scope="session")
def quantum_shots(request: pytest.FixtureRequest) -> int | None:
    """Returns the --quantum-shots override value, or None if not set."""
    return request.config.getoption("quantum_shots", default=None)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def quantum_significance(request: pytest.FixtureRequest) -> float | None:
    """Returns the --quantum-significance override value, or None if not set."""
    return request.config.getoption("quantum_significance", default=None)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def qiskit_sampler() -> object:
    """Session-scoped Qiskit StatevectorSampler (Qiskit 1.0+ primitives).

    Returns a StatevectorSampler instance. Use with assert_sampler_distribution.

    Example::

        def test_bell_sampler(qiskit_sampler):
            from qiskit.circuit import QuantumCircuit
            from pytest_quantum import assert_sampler_distribution

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            result = qiskit_sampler.run([(qc,)]).result()
            assert_sampler_distribution(result, {"00": 0.5, "11": 0.5})
    """
    _require("qiskit", "qiskit")
    from qiskit.primitives import StatevectorSampler

    return StatevectorSampler()


@pytest.fixture(scope="session")
def qiskit_estimator() -> object:
    """Session-scoped Qiskit StatevectorEstimator (Qiskit 1.0+ primitives).

    Returns a StatevectorEstimator instance. Use with assert_estimator_close.

    Example::

        def test_z_expectation(qiskit_estimator):
            from qiskit.circuit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp
            from pytest_quantum import assert_estimator_close

            qc = QuantumCircuit(1)  # |0> state, <Z> = 1.0
            obs = SparsePauliOp("Z")
            result = qiskit_estimator.run([(qc, obs)]).result()
            assert_estimator_close(result, expected=1.0, atol=0.01)
    """
    _require("qiskit", "qiskit")
    from qiskit.primitives import StatevectorEstimator

    return StatevectorEstimator()


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


@pytest.fixture(scope="session")
def cirq_sampler() -> object:
    """Session-scoped Cirq sampler for shot-based simulation.

    Returns a callable ``run(circuit, repetitions=1024)`` that executes
    a Cirq circuit with measurements and returns a count dict.

    The circuit must contain measurement gates (cirq.measure).

    Example::

        def test_cirq_bell(cirq_sampler):
            import cirq

            q = cirq.LineQubit.range(2)
            circuit = cirq.Circuit(
                cirq.H(q[0]),
                cirq.CNOT(q[0], q[1]),
                cirq.measure(q[0], q[1], key="result"),
            )
            counts = cirq_sampler(circuit, repetitions=2000)
            assert "00" in counts
    """
    _require("cirq", "cirq")
    import cirq
    import numpy as np

    def run(circuit: object, repetitions: int = 1024) -> dict[str, int]:
        result = cirq.Simulator().run(
            circuit,  # type: ignore[arg-type]
            repetitions=repetitions,
        )
        # Collect all measurement keys and concatenate bits
        all_bits = []
        for key in sorted(result.measurements.keys()):
            all_bits.append(result.measurements[key])
        if not all_bits:
            raise ValueError(
                "Circuit has no measurement gates. Add cirq.measure() to the circuit."
            )
        combined = np.concatenate(all_bits, axis=1)
        counts: dict[str, int] = {}
        for row in combined:
            key = "".join(str(b) for b in row)
            counts[key] = counts.get(key, 0) + 1
        return counts

    return run


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


# ---------------------------------------------------------------------------
# Fixtures — Pytket
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pytket_circuit_factory() -> Any:
    """Returns pytket Circuit class for building circuits in tests.

    Usage::

        def test_h(pytket_circuit_factory):
            Circuit = pytket_circuit_factory
            c = Circuit(1)
            c.H(0)
            assert c.n_qubits == 1
    """
    _require("pytket", "pip install pytket")
    from pytket.circuit import Circuit

    return Circuit


# ---------------------------------------------------------------------------
# Fixtures — Stim
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def stim_sampler() -> Any:
    """Stim stabilizer circuit sampler.

    Returns a callable ``sample(circuit, shots=1024)`` that runs a stim
    circuit and returns a count dict ``{"00": 503, "11": 497, ...}``.

    Usage::

        def test_bell(stim_sampler):
            import stim

            c = stim.Circuit('''
                H 0
                CNOT 0 1
                M 0 1
            ''')
            counts = stim_sampler(c, shots=1000)
            assert "00" in counts
    """
    _require("stim", "pip install stim")

    def _sample(circuit: Any, *, shots: int = 1024) -> dict[str, int]:
        sampler = circuit.compile_sampler()
        batch = sampler.sample(shots)  # shape (shots, n_measurements)
        counts: dict[str, int] = {}
        for row in batch:
            key = "".join("1" if b else "0" for b in row)
            counts[key] = counts.get(key, 0) + 1
        return counts

    return _sample


# ---------------------------------------------------------------------------
# Fixtures — Benchmarking
# ---------------------------------------------------------------------------


@pytest.fixture
def quantum_benchmark(request: pytest.FixtureRequest) -> Any:
    """Quantum-aware benchmark fixture.

    Wraps pytest-benchmark if installed, otherwise provides a simple timing
    wrapper.

    Usage::

        def test_circuit_speed(quantum_benchmark):
            result = quantum_benchmark(run_circuit, my_circuit, shots=1024)
    """
    try:
        benchmark = request.getfixturevalue("benchmark")

        def _run_with_benchmark(
            fn: Any, *args: Any, n_qubits: int | None = None, **kwargs: Any
        ) -> Any:
            result = benchmark.pedantic(
                fn, args=args, kwargs=kwargs, iterations=1, rounds=5
            )
            if n_qubits is not None:
                benchmark.extra_info["n_qubits"] = n_qubits
            return result

        return _run_with_benchmark
    except pytest.FixtureLookupError:
        import time

        def _run_simple(
            fn: Any, *args: Any, n_qubits: int | None = None, **kwargs: Any
        ) -> Any:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"\n  quantum_benchmark: {elapsed * 1000:.1f}ms")
            return result

        return _run_simple


# ---------------------------------------------------------------------------
# Fixtures — Shot Budget
# ---------------------------------------------------------------------------


@pytest.fixture
def shot_budget() -> Any:
    """Track total shots used across assertions in a test.

    Returns the ``ShotBudget`` class — instantiate with ``shot_budget(max_shots=N)``.

    Usage::

        def test_bell(aer_simulator, shot_budget):
            budget = shot_budget(max_shots=10_000)
            shots = budget.allocate(2000)
            # ... run circuit with shots ...
            assert budget.remaining == 8000
    """

    class ShotBudget:
        """Tracks shot usage within a single test."""

        def __init__(self, max_shots: int = 100_000) -> None:
            self.max_shots = max_shots
            self.used = 0

        def allocate(self, n: int) -> int:
            """Return *n* if within budget, else raise AssertionError."""
            if self.used + n > self.max_shots:
                raise AssertionError(
                    f"Shot budget exceeded: requesting {n} more shots but only "
                    f"{self.max_shots - self.used} remaining of "
                    f"{self.max_shots} total."
                )
            self.used += n
            return n

        @property
        def remaining(self) -> int:
            """Number of shots remaining in the budget."""
            return self.max_shots - self.used

        def __repr__(self) -> str:
            return f"ShotBudget(used={self.used}/{self.max_shots})"

    return ShotBudget


# ---------------------------------------------------------------------------
# Fixtures — quantum_backends marker support
# ---------------------------------------------------------------------------


@pytest.fixture
def quantum_backend_name(request: pytest.FixtureRequest) -> str:
    """Parametrized backend name from @pytest.mark.quantum_backends.

    Auto-skips if the backend's SDK is not installed.

    Usage::

        @pytest.mark.quantum_backends("qiskit", "cirq", "pennylane")
        def test_h_gate(quantum_backend_name):
            if quantum_backend_name == "qiskit":
                from qiskit import QuantumCircuit

                qc = QuantumCircuit(1)
                qc.h(0)
                assert_unitary(qc, H_MATRIX)
            elif quantum_backend_name == "cirq":
                ...
    """
    name: str = request.param
    sdk_map: dict[str, str] = {
        "qiskit": "qiskit",
        "cirq": "cirq",
        "pennylane": "pennylane",
        "braket": "braket",
        "pytket": "pytket",
        "stim": "stim",
        "qutip": "qutip",
    }
    if name in sdk_map and importlib.util.find_spec(sdk_map[name]) is None:
        pytest.skip(f"{name!r} SDK not installed (pip install pytest-quantum[{name}])")
    return name


@pytest.fixture
def quantum_backend(
    request: pytest.FixtureRequest, quantum_backend_name: str
) -> object:
    """Returns the primary simulator/runner for the current quantum_backend_name.

    Maps backend names to their corresponding pytest-quantum fixtures:
    - "qiskit"    -> aer_simulator
    - "cirq"      -> cirq_simulator
    - "pennylane" -> pennylane_device (factory)
    - "braket"    -> braket_simulator
    - "pytket"    -> pytket_circuit_factory
    - "stim"      -> stim_sampler

    Usage::

        @pytest.mark.quantum_backends("qiskit", "cirq")
        def test_something(quantum_backend, quantum_backend_name):
            # quantum_backend is the simulator for the current backend
            ...
    """
    fixture_map: dict[str, str] = {
        "qiskit": "aer_simulator",
        "cirq": "cirq_simulator",
        "pennylane": "pennylane_device",
        "braket": "braket_simulator",
        "pytket": "pytket_circuit_factory",
        "stim": "stim_sampler",
    }
    fixture_name = fixture_map.get(quantum_backend_name)
    if fixture_name is None:
        pytest.skip(f"No default fixture for backend {quantum_backend_name!r}")
    return request.getfixturevalue(fixture_name)


# ---------------------------------------------------------------------------
# Fixtures — IBM Quantum real hardware
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ibm_backend(request: pytest.FixtureRequest) -> Any:
    """Real IBM Quantum backend via QiskitRuntimeService.

    Requires:
        - --quantum-real CLI flag
        - IBM_QUANTUM_TOKEN environment variable (or ~/.qiskit/qiskit-ibm.json)
        - Optional: IBM_QUANTUM_INSTANCE env var (e.g. "ibm-q/open/main")
        - Optional: IBM_QUANTUM_BACKEND env var (e.g. "ibm_brisbane") -- defaults to least_busy

    Example::

        @pytest.mark.quantum_real
        def test_on_real_hardware(ibm_backend):
            from qiskit import QuantumCircuit
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from pytest_quantum import assert_measurement_distribution

            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)

            sampler = Sampler(ibm_backend)
            job = sampler.run([qc], shots=1024)
            result = job.result()[0]
            counts = dict(result.data.c.get_counts())
            assert_measurement_distribution(counts, {"0": 0.5, "1": 0.5})
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip("--quantum-real not set")

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
    backend_name = os.environ.get("IBM_QUANTUM_BACKEND", "")
    # Valid channels in qiskit-ibm-runtime >= 0.20:
    #   ibm_quantum_platform  — token from quantum.ibm.com (recommended)
    #   ibm_cloud             — IBM Cloud API key + CRN instance
    channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
    min_qubits = int(os.environ.get("IBM_QUANTUM_MIN_QUBITS", "5"))

    if not token:
        pytest.skip(
            "IBM_QUANTUM_TOKEN not set. Export your token:\n"
            "  export IBM_QUANTUM_TOKEN=<your-token>\n"
            "Get one at https://quantum.ibm.com"
        )

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        pytest.skip("qiskit-ibm-runtime not installed: pip install qiskit-ibm-runtime")

    # Try all known valid channel names in order
    channels_to_try = list(
        dict.fromkeys([channel, "ibm_quantum_platform", "ibm_cloud"])
    )
    service = None
    last_error = ""
    for ch in channels_to_try:
        try:
            service = QiskitRuntimeService(channel=ch, token=token, instance=instance)
            break
        except Exception as exc:
            last_error = str(exc)
            continue

    if service is None:
        pytest.skip(
            f"IBM Quantum connection failed: {last_error}\n"
            "  Check IBM_QUANTUM_TOKEN is valid (get it from quantum.ibm.com)."
        )

    try:
        if backend_name:
            return service.backend(backend_name)
        return service.least_busy(
            min_num_qubits=min_qubits, simulator=False, operational=True
        )
    except Exception as exc:
        pytest.skip(f"IBM Quantum backend selection failed: {exc}")


@pytest.fixture(scope="session")
def ionq_backend(request: pytest.FixtureRequest) -> Any:
    """Real IonQ quantum backend via Azure Quantum or IonQ cloud.

    Requires:
        - --quantum-real CLI flag
        - IONQ_API_KEY environment variable
        - Optional: IONQ_BACKEND env var ("simulator" or "qpu.aria-1", "qpu.forte-1", default "simulator")

    Example::

        def test_on_ionq(ionq_backend):
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            counts = assert_backend_executes(qc, ionq_backend, shots=1024)
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip("--quantum-real not set")

    api_key = os.environ.get("IONQ_API_KEY", "")
    if not api_key:
        pytest.skip(
            "IONQ_API_KEY not set. Export your IonQ API key:\n"
            "  export IONQ_API_KEY=<your-api-key>\n"
            "Get one at https://cloud.ionq.com"
        )

    ionq_backend_name = os.environ.get("IONQ_BACKEND", "simulator")

    try:
        from qiskit_ionq import IonQProvider
    except ImportError:
        pytest.skip("qiskit-ionq not installed: pip install qiskit-ionq")

    try:
        provider = IonQProvider(api_key)
        return provider.get_backend(ionq_backend_name)
    except Exception as exc:
        pytest.skip(f"IonQ connection failed: {exc}")


@pytest.fixture(scope="session")
def quantinuum_backend(request: pytest.FixtureRequest) -> Any:
    """Real Quantinuum quantum backend via pytket-quantinuum.

    Requires:
        - --quantum-real CLI flag
        - QUANTINUUM_USERNAME and QUANTINUUM_PASSWORD environment variables
        - Optional: QUANTINUUM_DEVICE env var (default "H1-1E" emulator)

    Example::

        def test_on_quantinuum(quantinuum_backend): ...
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip("--quantum-real not set")

    username = os.environ.get("QUANTINUUM_USERNAME", "")
    password = os.environ.get("QUANTINUUM_PASSWORD", "")
    if not username or not password:
        pytest.skip(
            "QUANTINUUM_USERNAME and QUANTINUUM_PASSWORD must both be set.\n"
            "  export QUANTINUUM_USERNAME=<your-email>\n"
            "  export QUANTINUUM_PASSWORD=<your-password>\n"
            "Register at https://um.qapi.quantinuum.com"
        )

    device_name = os.environ.get("QUANTINUUM_DEVICE", "H1-1E")

    try:
        from pytket.extensions.quantinuum import QuantinuumBackend
    except ImportError:
        pytest.skip("pytket-quantinuum not installed: pip install pytket-quantinuum")

    try:
        backend = QuantinuumBackend(
            device_name=device_name,
            username=username,
            password=password,
        )
        return backend
    except Exception as exc:
        pytest.skip(f"Quantinuum connection failed: {exc}")


@pytest.fixture(scope="session")
def quantum_hardware_info(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Returns info about configured real hardware backends.

    Checks environment variables to determine which cloud backends have
    credentials configured.  Does **not** attempt a live connection.

    Returns a dict with keys:
        - ``"ibm_available"``        — True if IBM_QUANTUM_TOKEN is set
        - ``"ionq_available"``       — True if IONQ_API_KEY is set
        - ``"quantinuum_available"`` — True if both QUANTINUUM_USERNAME and
          QUANTINUUM_PASSWORD are set
        - ``"aws_available"``        — True if BRAKET_DEVICE_ARN is set

    Example::

        def test_info(quantum_hardware_info):
            info = quantum_hardware_info
            assert isinstance(info["ibm_available"], bool)
    """
    return {
        "ibm_available": bool(os.environ.get("IBM_QUANTUM_TOKEN", "")),
        "ionq_available": bool(os.environ.get("IONQ_API_KEY", "")),
        "quantinuum_available": (
            bool(os.environ.get("QUANTINUUM_USERNAME", ""))
            and bool(os.environ.get("QUANTINUUM_PASSWORD", ""))
        ),
        "aws_available": bool(os.environ.get("BRAKET_DEVICE_ARN", "")),
    }


@pytest.fixture(scope="session")
def braket_cloud_device(request: pytest.FixtureRequest) -> Any:
    """AWS Braket cloud quantum device (session-scoped).

    Auto-skips unless ``--quantum-real`` is passed.
    Requires:

    - AWS credentials configured (``aws configure`` or env vars)
    - ``BRAKET_DEVICE_ARN`` environment variable set to the device ARN,
      e.g. ``arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice``

    Usage::

        @pytest.mark.quantum_real
        def test_bell_on_ionq(braket_cloud_device):
            from braket.circuits import Circuit

            circuit = Circuit().h(0).cnot(0, 1).measure_all()
            task = braket_cloud_device.run(circuit, shots=100)
            counts = {str(k): v for k, v in task.result().measurement_counts.items()}
            assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
    """
    if not request.config.getoption("--quantum-real", default=False):
        pytest.skip(
            "AWS Braket cloud test skipped. Pass --quantum-real to enable.\n"
            "Also requires AWS credentials and BRAKET_DEVICE_ARN env var."
        )

    device_arn = os.environ.get("BRAKET_DEVICE_ARN")
    if not device_arn:
        pytest.skip(
            "BRAKET_DEVICE_ARN not set. Set it to the device ARN, e.g.:\n"
            "  export BRAKET_DEVICE_ARN=arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice"
        )

    _require("braket", "pip install pytest-quantum[braket]")

    try:
        from braket.aws import AwsDevice

        return AwsDevice(device_arn)
    except Exception as exc:
        pytest.skip(f"AWS Braket device unavailable: {exc}")


# ---------------------------------------------------------------------------
# Fixtures — QuTiP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qutip_solver() -> Any:
    """QuTiP Lindblad master equation solver fixture.

    Returns a callable ``solve(H, psi0, tlist, c_ops=None)`` that runs
    ``qutip.mesolve`` and returns the final density matrix as a numpy array.

    Requires: pip install pytest-quantum[qutip]

    Usage::

        def test_qubit_decay(qutip_solver):
            import qutip
            import numpy as np

            H = qutip.sigmaz() / 2  # Hamiltonian
            psi0 = qutip.basis(2, 0)  # |0> initial state
            gamma = 0.1
            c_ops = [np.sqrt(gamma) * qutip.sigmam()]  # decay operator
            tlist = np.linspace(0, 10, 100)

            rho_final = qutip_solver(H, psi0, tlist, c_ops=c_ops)
            assert_purity_above(rho_final, min_purity=0.0)  # mixed, but valid
    """
    _require("qutip", "pip install qutip")

    def _solve(
        H: Any,
        psi0: Any,
        tlist: Any,
        c_ops: list[Any] | None = None,
    ) -> NDArray[np.complex128]:
        import qutip

        result = qutip.mesolve(H, psi0, tlist, c_ops or [], [])
        final_state = result.states[-1]
        # Convert to density matrix if pure state
        rho = qutip.ket2dm(final_state) if final_state.type == "ket" else final_state
        return np.asarray(rho.full(), dtype=np.complex128)

    return _solve


# ---------------------------------------------------------------------------
# Fixtures — Tequila
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_backend_runner() -> Any:
    """Run the same quantum circuit on multiple backends and compare results.

    Returns a ``MultiBackendRunner`` instance with a ``.run_all(circuit_fn, shots)``
    method that executes a circuit on every installed backend and returns a dict
    mapping backend name → counts.

    Backends included: all installed from {qiskit/aer, cirq, pennylane, braket}.
    Backends that are not installed are silently skipped.

    Usage::

        def test_cross_backend(multi_backend_runner):
            def make_bell():
                from qiskit import QuantumCircuit

                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure_all()
                return qc

            results = multi_backend_runner.run_all(make_bell, shots=1024)
            for backend_name, counts in results.items():
                assert "00" in counts or "11" in counts, (
                    f"Bad distribution on {backend_name}"
                )
    """

    class MultiBackendRunner:
        """Runs circuits on all installed backends in parallel (thread-based)."""

        def run_all(
            self,
            circuit_fn: Any,
            shots: int = 1024,
            backends: list[str] | None = None,
        ) -> dict[str, dict[str, int]]:
            """Run circuit_fn() on each available backend.

            Args:
                circuit_fn: Callable that returns a Qiskit QuantumCircuit.
                shots:      Shots per backend.
                backends:   Backend names to run on. Default: all installed.

            Returns:
                Dict mapping backend name to counts dict.
            """
            import concurrent.futures

            available = backends or ["qiskit_aer", "cirq", "pennylane"]
            results: dict[str, dict[str, int]] = {}

            def _run_qiskit_aer(s: int) -> dict[str, int]:
                from qiskit import transpile
                from qiskit_aer import AerSimulator

                backend_obj = AerSimulator()
                qc = circuit_fn()
                t = transpile(qc, backend_obj, optimization_level=0)
                job = backend_obj.run(t, shots=s)
                counts: dict[str, int] = job.result().get_counts()
                return counts

            def _run_cirq(s: int) -> dict[str, int]:
                import cirq

                qc = circuit_fn()
                # Use cirq.Circuit from the Qiskit circuit via qasm2
                try:
                    from cirq.contrib.qasm_import import (
                        circuit_from_qasm,  # type: ignore[import-not-found]
                    )
                    from qiskit import qasm2

                    qasm_str = qasm2.dumps(qc)
                    cirq_circuit = circuit_from_qasm(qasm_str)
                except Exception:
                    # Fallback: return empty
                    return {}
                simulator = cirq.Simulator()
                result = simulator.run(cirq_circuit, repetitions=s)
                counts: dict[str, int] = {}
                for key in sorted(result.measurements.keys()):
                    bits = result.measurements[key]
                    for row in bits:
                        k = "".join(str(b) for b in row)
                        counts[k] = counts.get(k, 0) + 1
                return counts

            def _run_pennylane(s: int) -> dict[str, int]:
                import pennylane as qml
                from qiskit import qasm2

                qc = circuit_fn()
                n = qc.num_qubits

                try:
                    from pennylane.io.qasm import (
                        from_qasm,  # type: ignore[import-not-found]
                    )

                    qasm_str = qasm2.dumps(qc)
                    pl_circuit = from_qasm(qasm_str)
                    dev = qml.device("default.qubit", wires=n, shots=s)
                    qnode = qml.QNode(pl_circuit, dev)
                    _ = qnode()
                except Exception:
                    return {}
                return {}

            runner_map: dict[str, Any] = {
                "qiskit_aer": _run_qiskit_aer,
                "cirq": _run_cirq,
                "pennylane": _run_pennylane,
            }

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(available)
            ) as executor:
                futures = {}
                for name in available:
                    if (
                        importlib.util.find_spec(name.replace("_", "-").split("-")[0])
                        is None
                        and importlib.util.find_spec(name.split("_")[0]) is None
                        and importlib.util.find_spec(name) is None
                    ):
                        continue
                    fn = runner_map.get(name)
                    if fn is not None:
                        futures[name] = executor.submit(fn, shots)

                for name, future in futures.items():
                    try:
                        r = future.result(timeout=60)
                        if r:
                            results[name] = r
                    except Exception:
                        pass

            return results

    return MultiBackendRunner()


@pytest.fixture(scope="session")
def benchmark_suite() -> Any:
    """Benchmark suite fixture: collects timing/shot data across multiple assertions.

    Returns a ``BenchmarkSuite`` that records per-assertion metrics during the
    test session and can print a summary table at the end.

    Usage::

        def test_bell_assertions(benchmark_suite):
            import time

            with benchmark_suite.record("assert_measurement_distribution"):
                # ... run assertion ...
                time.sleep(0.001)  # placeholder
            benchmark_suite.print_summary()
    """
    import contextlib
    import time

    class BenchmarkSuite:
        """Collects timing measurements for quantum assertion benchmarking."""

        def __init__(self) -> None:
            self._records: dict[str, list[float]] = {}

        @contextlib.contextmanager
        def record(self, name: str) -> Any:
            """Context manager: time the enclosed block and record under *name*."""
            start = time.perf_counter()
            try:
                yield self
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._records.setdefault(name, []).append(elapsed_ms)

        def mean_ms(self, name: str) -> float:
            """Return mean elapsed time in ms for *name*."""
            vals = self._records.get(name, [])
            return float(np.mean(vals)) if vals else 0.0

        def print_summary(self) -> None:
            """Print a formatted summary table of all recorded benchmarks."""
            if not self._records:
                print("No benchmark records.")
                return
            header = f"{'Assertion':<45} {'Calls':>5} {'Mean ms':>10} {'Total ms':>10}"
            print(f"\n{'=' * len(header)}")
            print(header)
            print(f"{'-' * len(header)}")
            for name in sorted(self._records):
                vals = self._records[name]
                print(
                    f"  {name:<43} {len(vals):>5} "
                    f"{float(np.mean(vals)):>10.2f} "
                    f"{float(sum(vals)):>10.2f}"
                )
            print(f"{'=' * len(header)}\n")

        def __repr__(self) -> str:
            return f"BenchmarkSuite(assertions={list(self._records.keys())})"

    return BenchmarkSuite()


@pytest.fixture(scope="session")
def tequila_backend() -> Any:
    """Tequila quantum chemistry circuit backend.

    Returns the tequila module itself, since Tequila uses a functional API
    (tq.simulate, tq.minimize, tq.ExpectationValue) rather than a backend object.

    Requires: pip install tequila-basic

    Usage::

        def test_h2_vqe(tequila_backend):
            import tequila as tq
            import numpy as np

            # Simple 1-qubit circuit
            U = tq.gates.H(target=0)
            result = tq.simulate(U)
            assert abs(result[0]) ** 2 == pytest.approx(0.5, abs=1e-6)
    """
    _require("tequila", "pip install tequila-basic")
    import tequila as tq

    return tq
