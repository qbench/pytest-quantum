"""Tests for pytest-quantum configuration support."""

from __future__ import annotations

from pytest_quantum.config import QuantumConfig


class TestQuantumConfig:
    def test_defaults(self) -> None:
        cfg = QuantumConfig()
        assert cfg.shots is None
        assert cfg.significance is None
        assert cfg.slow is False
        assert cfg.real is False
        assert cfg.update_snapshots is False
        assert cfg.report == "none"
        assert cfg.report_path == "quantum-report"

    def test_custom_values(self) -> None:
        cfg = QuantumConfig(shots=2048, significance=0.01, slow=True)
        assert cfg.shots == 2048
        assert cfg.significance == 0.01
        assert cfg.slow is True

    def test_report_fields(self) -> None:
        cfg = QuantumConfig(report="json", report_path="my-report")
        assert cfg.report == "json"
        assert cfg.report_path == "my-report"
