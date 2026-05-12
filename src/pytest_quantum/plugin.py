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
    from pytest import Config, Item, Metafunc, Parser

# ---------------------------------------------------------------------------
# Sub-plugin registration — fixtures live in dedicated modules
# ---------------------------------------------------------------------------

pytest_plugins = [
    "pytest_quantum.fixtures.qiskit",
    "pytest_quantum.fixtures.cirq",
    "pytest_quantum.fixtures.hardware",
    "pytest_quantum.fixtures.benchmarks",
    "pytest_quantum.fixtures.other",
    "pytest_quantum.retry",
]


# ---------------------------------------------------------------------------
# Helper — shared with fixture sub-modules
# ---------------------------------------------------------------------------


def _require(package: str, extra: str) -> None:
    """Raise a helpful ImportError if *package* is not installed."""
    import importlib.util

    if importlib.util.find_spec(package) is None:
        pytest.skip(
            f"{package!r} is not installed. "
            f"Install it with: pip install pytest-quantum[{extra}]"
        )


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
    group.addoption(
        "--quantum-only",
        action="store_true",
        default=False,
        help="Run only tests marked with @pytest.mark.quantum or using quantum fixtures.",
    )

    # ini options
    parser.addini("quantum_shots", "Default shot count for quantum tests", type="string", default="")
    parser.addini("quantum_significance", "Default p-value threshold", type="string", default="")
    parser.addini("quantum_slow", "Run quantum_slow tests by default", type="bool", default=False)
    parser.addini("quantum_real", "Run quantum_real tests by default", type="bool", default=False)
    parser.addini("quantum_update_snapshots", "Update snapshots by default", type="bool", default=False)


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
    config.addinivalue_line(
        "markers",
        "quantum_retry(n): retry a quantum test up to n times on failure.",
    )
    # Set env var so snapshot helpers can detect the update flag without
    # needing access to the pytest config object.
    try:
        if config.getoption("--quantum-update-snapshots", default=False):
            os.environ["PYTEST_QUANTUM_UPDATE_SNAPSHOTS"] = "1"
    except (ValueError, AttributeError):
        # getoption raises ValueError if the option is not registered yet
        pass

    # Load and store configuration
    from pytest_quantum.config import load_config

    config._quantum_config = load_config(config)  # type: ignore[attr-defined]


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

    # --quantum-only filtering
    if config.getoption("--quantum-only", default=False):
        quantum_fixtures = {
            "aer_simulator", "aer_statevector_simulator", "aer_noise_simulator",
            "cirq_simulator", "cirq_sampler", "braket_simulator", "graphix_backend",
            "pennylane_device", "pytket_circuit_factory", "stim_sampler",
            "qiskit_sampler", "qiskit_estimator", "quantum_shots",
            "quantum_significance", "quantum_benchmark", "shot_budget",
            "quantum_backend_name", "quantum_backend", "multi_backend_runner",
            "benchmark_suite", "ibm_backend", "ionq_backend",
            "quantinuum_backend", "braket_cloud_device", "qutip_solver",
            "tequila_backend",
        }
        selected = []
        deselected = []
        for item in items:
            if (
                item.get_closest_marker("quantum")
                or item.get_closest_marker("quantum_slow")
                or item.get_closest_marker("quantum_real")
            ):
                selected.append(item)
            elif quantum_fixtures & set(item.fixturenames):
                selected.append(item)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Parametrize tests with @pytest.mark.quantum_backends across backend names."""
    marker = metafunc.definition.get_closest_marker("quantum_backends")
    if marker and "quantum_backend_name" in metafunc.fixturenames:
        backends = list(marker.args)
        if not backends:
            backends = ["qiskit", "cirq", "pennylane", "braket"]
        metafunc.parametrize("quantum_backend_name", backends, scope="function")


# ---------------------------------------------------------------------------
# Hooks — test setup (marker consumption)
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item: Item) -> None:
    """Wire up @pytest.mark.shots(n) and @pytest.mark.significance(p) markers."""
    shots_marker = item.get_closest_marker("shots")
    if shots_marker:
        item._quantum_shots = shots_marker.args[0] if shots_marker.args else None  # type: ignore[attr-defined]
    else:
        item._quantum_shots = None  # type: ignore[attr-defined]

    sig_marker = item.get_closest_marker("significance")
    if sig_marker:
        item._quantum_significance = sig_marker.args[0] if sig_marker.args else None  # type: ignore[attr-defined]
    else:
        item._quantum_significance = None  # type: ignore[attr-defined]


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
# Fixtures — shots & significance (remain in plugin.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def quantum_shots(request: pytest.FixtureRequest) -> int | None:
    """Returns shot count: per-test marker > CLI > ini > None."""
    # Per-test marker (highest priority)
    marker_val = getattr(request.node, "_quantum_shots", None)
    if marker_val is not None:
        return marker_val
    # CLI / ini config
    qc = getattr(request.config, "_quantum_config", None)
    if qc is not None and qc.shots is not None:
        return qc.shots
    # Legacy CLI fallback
    return request.config.getoption("quantum_shots", default=None)  # type: ignore[no-any-return]


@pytest.fixture(scope="function")
def quantum_significance(request: pytest.FixtureRequest) -> float | None:
    """Returns significance: per-test marker > CLI > ini > None."""
    marker_val = getattr(request.node, "_quantum_significance", None)
    if marker_val is not None:
        return marker_val
    qc = getattr(request.config, "_quantum_config", None)
    if qc is not None and qc.significance is not None:
        return qc.significance
    return request.config.getoption("quantum_significance", default=None)  # type: ignore[no-any-return]
