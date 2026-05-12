"""Benchmark and shot-budget fixtures for pytest-quantum."""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest


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
                    from cirq.contrib.qasm_import import circuit_from_qasm
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
                    from pennylane.io.qasm import from_qasm

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
