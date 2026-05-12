"""Other framework fixtures for pytest-quantum."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import pytest

from pytest_quantum.plugin import _require

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


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
    import numpy as np

    def _solve(
        H: Any,
        psi0: Any,
        tlist: Any,
        c_ops: list[Any] | None = None,
    ) -> NDArray:
        import qutip

        result = qutip.mesolve(H, psi0, tlist, c_ops or [], [])
        final_state = result.states[-1]
        # Convert to density matrix if pure state
        rho = qutip.ket2dm(final_state) if final_state.type == "ket" else final_state
        return np.asarray(rho.full(), dtype=np.complex128)

    return _solve


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


@pytest.fixture(scope="session")
def cuda_quantum_simulator() -> object:
    """Session-scoped CUDA Quantum module.

    CUDA Quantum uses kernel-based execution, so this returns the ``cudaq``
    module itself rather than a simulator object.

    Example::

        def test_cudaq(cuda_quantum_simulator):
            cudaq = cuda_quantum_simulator
            kernel = cudaq.make_kernel()
            q = kernel.qalloc()
            kernel.h(q)
    """
    _require("cudaq", "cuda_quantum")
    import cudaq

    return cudaq


@pytest.fixture(scope="session")
def qibo_backend() -> object:
    """Session-scoped Qibo backend (numpy).

    Sets the Qibo backend to ``numpy`` and returns the ``qibo`` module.

    Example::

        def test_qibo(qibo_backend):
            from qibo import Circuit, gates

            c = Circuit(2)
            c.add(gates.H(0))
            c.add(gates.CNOT(0, 1))
    """
    _require("qibo", "qibo")
    import qibo

    qibo.set_backend("numpy")
    return qibo


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
        "cuda_quantum": "cudaq",
        "qibo": "qibo",
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
        "cuda_quantum": "cuda_quantum_simulator",
        "qibo": "qibo_backend",
    }
    fixture_name = fixture_map.get(quantum_backend_name)
    if fixture_name is None:
        pytest.skip(f"No default fixture for backend {quantum_backend_name!r}")
    return request.getfixturevalue(fixture_name)
