"""Tests for pytest-quantum configuration support."""
from __future__ import annotations

import pytest
from pytest_quantum.config import QuantumConfig, load_config


class TestQuantumConfig:
    def test_defaults(self) -> None:
        cfg = QuantumConfig()
        assert cfg.shots is None
        assert cfg.significance is None
        assert cfg.slow is False
        assert cfg.real is False
        assert cfg.update_snapshots is False

    def test_custom_values(self) -> None:
        cfg = QuantumConfig(shots=2048, significance=0.01, slow=True)
        assert cfg.shots == 2048
        assert cfg.significance == 0.01
        assert cfg.slow is True
