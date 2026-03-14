"""Tests for snapshot / golden-file assertions.

Uses tmp_path and monkeypatch to isolate snapshot files.
Uses monkeypatched to_unitary to avoid needing a real quantum SDK.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pytest_quantum.assertions.snapshot as snap_mod
from pytest_quantum.assertions.snapshot import (
    assert_distribution_snapshot,
    assert_unitary_snapshot,
)

# ---------------------------------------------------------------------------
# Autouse fixture: isolate snapshot directory per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(snap_mod, "_SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.delenv("PYTEST_QUANTUM_UPDATE_SNAPSHOTS", raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

BELL_COUNTS: dict[str, int] = {"00": 512, "11": 512}
BIASED_COUNTS: dict[str, int] = {"00": 900, "11": 100}


def _make_mock_circuit(matrix: np.ndarray) -> object:
    """Return a fake circuit object whose to_unitary returns matrix."""
    return matrix  # will be used via patched to_unitary


# ---------------------------------------------------------------------------
# assert_unitary_snapshot
# ---------------------------------------------------------------------------


def _patch_to_unitary(monkeypatch: pytest.MonkeyPatch, fn: object) -> None:
    """Patch to_unitary in sys.modules so the lazy import inside snapshot sees it."""
    import importlib
    import sys

    # Ensure the module is loaded first
    importlib.import_module("pytest_quantum.converters.to_unitary")
    conv_mod = sys.modules["pytest_quantum.converters.to_unitary"]
    monkeypatch.setattr(conv_mod, "to_unitary", fn)


class TestAssertUnitarySnapshot:
    def test_first_run_creates_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First call saves the snapshot file."""

        def mock_to_unitary(circuit: object) -> np.ndarray:
            return HADAMARD

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        snap_path = snap_mod._SNAPSHOT_DIR / "test_snap.npy"
        assert not snap_path.exists()

        assert_unitary_snapshot("fake_circuit", "test_snap")
        assert snap_path.exists()

    def test_second_run_passes_same_unitary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second call with same unitary passes without error."""

        def mock_to_unitary(circuit: object) -> np.ndarray:
            return HADAMARD

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        # First run — creates snapshot
        assert_unitary_snapshot("fake_circuit", "snap2")
        # Second run — should pass
        assert_unitary_snapshot("fake_circuit", "snap2")

    def test_changed_unitary_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After snapshot is saved, a different unitary fails."""
        call_count = 0

        def mock_to_unitary(circuit: object) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return HADAMARD
            return PAULI_X

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        # First run: save snapshot
        assert_unitary_snapshot("fake_circuit", "snap3")
        # Second run: different unitary → should fail
        with pytest.raises(AssertionError, match="unitary has changed"):
            assert_unitary_snapshot("fake_circuit", "snap3")

    def test_changed_unitary_with_update_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """update=True regenerates the snapshot without raising."""
        call_count = 0

        def mock_to_unitary(circuit: object) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return HADAMARD if call_count == 1 else PAULI_X

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        assert_unitary_snapshot("fake_circuit", "snap4")
        # update=True: should overwrite and not raise
        assert_unitary_snapshot("fake_circuit", "snap4", update=True)
        # Now the snapshot contains PAULI_X — a third call with PAULI_X passes
        _patch_to_unitary(monkeypatch, lambda c: PAULI_X)
        assert_unitary_snapshot("fake_circuit", "snap4")

    def test_env_var_triggers_update(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PYTEST_QUANTUM_UPDATE_SNAPSHOTS=1 regenerates without raising."""
        call_count = 0

        def mock_to_unitary(circuit: object) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return HADAMARD if call_count == 1 else PAULI_X

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        # Create initial snapshot
        assert_unitary_snapshot("fake_circuit", "snap5")
        # Set env var — should regenerate
        monkeypatch.setenv("PYTEST_QUANTUM_UPDATE_SNAPSHOTS", "1")
        assert_unitary_snapshot("fake_circuit", "snap5")  # should not raise

    def test_global_phase_ignored_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unitary differing by global phase still passes."""
        call_count = 0

        def mock_to_unitary(circuit: object) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return HADAMARD
            # Second call: same unitary but with global phase
            return np.exp(1j * 0.7) * HADAMARD

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        assert_unitary_snapshot("fake_circuit", "snap6")
        # Should pass (allow_global_phase=True by default)
        assert_unitary_snapshot("fake_circuit", "snap6")

    def test_shape_change_fails_with_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shape change gives informative error."""
        call_count = 0
        cnot = np.eye(4, dtype=np.complex128)

        def mock_to_unitary(circuit: object) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return HADAMARD if call_count == 1 else cnot

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        assert_unitary_snapshot("fake_circuit", "snap7")
        with pytest.raises(AssertionError, match="shape changed"):
            assert_unitary_snapshot("fake_circuit", "snap7")

    def test_error_message_mentions_update_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0

        def mock_to_unitary(circuit: object) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return HADAMARD if call_count == 1 else PAULI_X

        _patch_to_unitary(monkeypatch, mock_to_unitary)

        assert_unitary_snapshot("fake_circuit", "snap8")
        with pytest.raises(AssertionError) as exc_info:
            assert_unitary_snapshot("fake_circuit", "snap8")
        assert "--quantum-update-snapshots" in str(exc_info.value)


# ---------------------------------------------------------------------------
# assert_distribution_snapshot
# ---------------------------------------------------------------------------


class TestAssertDistributionSnapshot:
    def test_first_run_creates_file(self, tmp_path: Path) -> None:
        """First call saves the snapshot file."""
        snap_path = snap_mod._SNAPSHOT_DIR / "dist_snap1_dist.npy"
        assert not snap_path.exists()
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap1")
        assert snap_path.exists()

    def test_second_run_same_distribution_passes(self) -> None:
        """Second call with same distribution passes."""
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap2")
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap2")

    def test_changed_distribution_fails(self) -> None:
        """Significantly different distribution raises AssertionError."""
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap3")
        with pytest.raises(AssertionError, match="distribution has changed"):
            assert_distribution_snapshot(BIASED_COUNTS, "dist_snap3", max_tvd=0.01)

    def test_update_flag_regenerates(self) -> None:
        """update=True overwrites snapshot without raising."""
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap4")
        # Different distribution but update=True — should not raise
        assert_distribution_snapshot(BIASED_COUNTS, "dist_snap4", update=True)
        # Now snapshot is BIASED_COUNTS — third call should be consistent
        assert_distribution_snapshot(BIASED_COUNTS, "dist_snap4")

    def test_env_var_triggers_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap5")
        monkeypatch.setenv("PYTEST_QUANTUM_UPDATE_SNAPSHOTS", "1")
        # Should not raise even with different data
        assert_distribution_snapshot(BIASED_COUNTS, "dist_snap5")

    def test_empty_counts_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            assert_distribution_snapshot({}, "empty_snap")

    def test_max_tvd_exceeded_message_content(self) -> None:
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap6")
        with pytest.raises(AssertionError) as exc_info:
            assert_distribution_snapshot(BIASED_COUNTS, "dist_snap6", max_tvd=0.01)
        msg = str(exc_info.value)
        assert "TVD" in msg
        assert "--quantum-update-snapshots" in msg

    def test_large_max_tvd_accepts_different_distribution(self) -> None:
        """Very large max_tvd accepts almost any distribution."""
        assert_distribution_snapshot(BELL_COUNTS, "dist_snap7")
        # BIASED vs BELL — large TVD but max_tvd=1.0 allows it
        assert_distribution_snapshot(BIASED_COUNTS, "dist_snap7", max_tvd=1.0)

    def test_safe_name_with_slashes(self) -> None:
        """Snapshot names with slashes get sanitised."""
        assert_distribution_snapshot(BELL_COUNTS, "group/test_snap")
        # File should exist with safe name
        safe_path = snap_mod._SNAPSHOT_DIR / "group_test_snap_dist.npy"
        assert safe_path.exists()
