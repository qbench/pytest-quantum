"""Snapshot / golden-file assertions for quantum circuits and distributions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np

_SNAPSHOT_DIR = Path(".pytest-quantum-snapshots")


def assert_unitary_snapshot(
    circuit: object,
    name: str,
    *,
    update: bool = False,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert that a circuit's unitary matches its saved snapshot.

    First run: saves the unitary to .pytest-quantum-snapshots/<name>.npy.
    Subsequent runs: loads and compares. Pass --quantum-update-snapshots
    on the CLI to regenerate all snapshots.

    Args:
        circuit:            Any supported quantum circuit (Qiskit, Cirq, Braket,
                            PennyLane).
        name:               Unique snapshot name (used as filename, no path
                            separators).
        update:             If True, overwrite existing snapshot.
        atol:               Absolute tolerance (default 1e-6).
        allow_global_phase: If True (default), ignore global phase differences.

    Raises:
        AssertionError: If the unitary differs from the snapshot.
        TypeError:      If the circuit type is not supported.

    Example::

        def test_compiler_stable(compiled_circuit):
            from pytest_quantum import assert_unitary_snapshot

            assert_unitary_snapshot(compiled_circuit, "my_compiler_bell")
    """
    from pytest_quantum.converters.to_unitary import to_unitary

    _update = update or _should_update()
    path = _path(name, ".npy")
    actual = to_unitary(circuit)
    if _update or not path.exists():
        _write_snapshot_safe(path, actual)
        return
    expected = np.load(str(path))
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Snapshot '{name}': shape changed {expected.shape} → {actual.shape}.\n"
            f"  Run with --quantum-update-snapshots to regenerate."
        )
    if np.allclose(actual, expected, atol=atol):
        return
    if allow_global_phase:
        flat_idx = int(np.argmax(np.abs(expected)))
        e_val = expected.flat[flat_idx]
        a_val = actual.flat[flat_idx]
        if abs(e_val) > 1e-10 and abs(a_val) > 1e-10:
            phase = a_val / e_val
            if np.allclose(actual, phase * expected, atol=atol):
                return
    max_diff = float(np.max(np.abs(actual - expected)))
    raise AssertionError(
        f"Snapshot '{name}': unitary has changed.\n"
        f"  Max |diff|: {max_diff:.2e}   (tolerance: {atol:.2e})\n"
        f"  Run with --quantum-update-snapshots to accept the new behaviour."
    )


def assert_distribution_snapshot(
    counts: dict[str, int],
    name: str,
    *,
    update: bool = False,
    max_tvd: float = 0.05,
) -> None:
    """Assert that a measurement distribution matches its saved snapshot.

    Saves the normalised probability distribution derived from counts.
    Comparison uses Total Variation Distance (TVD).

    Args:
        counts:  Measurement count dict e.g. {"00": 512, "11": 512}.
        name:    Unique snapshot name.
        update:  If True, overwrite existing snapshot.
        max_tvd: Maximum allowed TVD from snapshot (default 0.05).

    Raises:
        AssertionError: If TVD from snapshot exceeds max_tvd.
        ValueError:     If counts is empty.

    Example::

        def test_distribution_stable(aer_simulator):
            counts = run_bell(aer_simulator, shots=4000)
            from pytest_quantum import assert_distribution_snapshot

            assert_distribution_snapshot(counts, "bell_distribution")
    """
    from pytest_quantum.stats.tests import tvd_from_counts

    _update = update or _should_update()
    path = _path(name, "_dist.npy")
    total = sum(counts.values())
    if total == 0:
        raise ValueError("counts dict is empty — nothing to snapshot")
    probs = {k: v / total for k, v in counts.items()}
    if _update or not path.exists():
        keys = np.array(list(probs.keys()))
        vals = np.array(list(probs.values()), dtype=np.float64)
        _write_snapshot_safe(path, np.array([keys, vals], dtype=object))
        return
    raw = np.load(str(path), allow_pickle=True)
    snap_keys = list(raw[0])
    snap_vals = [float(v) for v in raw[1]]
    snap_counts = {
        k: round(v * 10000) for k, v in zip(snap_keys, snap_vals, strict=False)
    }
    current_counts = {k: round(v * 10000) for k, v in probs.items()}
    distance = tvd_from_counts(current_counts, snap_counts)
    if distance <= max_tvd:
        return
    raise AssertionError(
        f"Snapshot '{name}': distribution has changed.\n"
        f"  TVD from snapshot: {distance:.4f}   (max allowed: {max_tvd:.4f})\n"
        f"  Run with --quantum-update-snapshots to accept the new distribution."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_snapshot_safe(path: Path, data: object) -> None:
    """Write snapshot file with advisory file lock for xdist safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    try:
        import filelock

        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            np.save(str(path), cast("Any", data))
    except ImportError:
        # filelock not installed — just write directly (acceptable for single-worker)
        np.save(str(path), cast("Any", data))


def _path(name: str, suffix: str) -> Path:
    safe = name.replace("/", "_").replace("\\", "_")
    return _SNAPSHOT_DIR / f"{safe}{suffix}"


def _should_update() -> bool:
    return os.environ.get("PYTEST_QUANTUM_UPDATE_SNAPSHOTS", "") == "1"
