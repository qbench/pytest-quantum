"""Configuration support for pytest-quantum.

Reads defaults from pyproject.toml [tool.pytest.ini_options], pytest.ini,
or setup.cfg. CLI options override ini values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import Config


@dataclass
class QuantumConfig:
    """Resolved configuration for pytest-quantum."""

    shots: int | None = None
    significance: float | None = None
    slow: bool = False
    real: bool = False
    update_snapshots: bool = False
    report: str = "none"
    report_path: str = "quantum-report"


def load_config(config: Config) -> QuantumConfig:
    """Load quantum config from ini values + CLI overrides."""
    # Read ini values (from pyproject.toml [tool.pytest.ini_options], pytest.ini, etc.)
    ini_shots = config.getini("quantum_shots")
    ini_significance = config.getini("quantum_significance")
    ini_slow = config.getini("quantum_slow")
    ini_real = config.getini("quantum_real")
    ini_update = config.getini("quantum_update_snapshots")
    ini_report = config.getini("quantum_report")
    ini_report_path = config.getini("quantum_report_path")

    # CLI overrides ini — all CLI options default to None so INI can take effect
    try:
        cli_shots = config.getoption("quantum_shots", default=None)
    except (ValueError, AttributeError):
        cli_shots = None
    try:
        cli_significance = config.getoption("quantum_significance", default=None)
    except (ValueError, AttributeError):
        cli_significance = None
    try:
        cli_slow = config.getoption("--quantum-slow", default=False)
    except (ValueError, AttributeError):
        cli_slow = False
    try:
        cli_real = config.getoption("--quantum-real", default=False)
    except (ValueError, AttributeError):
        cli_real = False
    try:
        cli_update = config.getoption("--quantum-update-snapshots", default=False)
    except (ValueError, AttributeError):
        cli_update = False
    try:
        cli_report = config.getoption("quantum_report", default=None)
    except (ValueError, AttributeError):
        cli_report = None
    try:
        cli_report_path = config.getoption("quantum_report_path", default=None)
    except (ValueError, AttributeError):
        cli_report_path = None

    # Resolve: CLI > ini > default
    shots = cli_shots if cli_shots is not None else (int(ini_shots) if ini_shots else None)
    significance = cli_significance if cli_significance is not None else (float(ini_significance) if ini_significance else None)
    slow = cli_slow or bool(ini_slow)
    real = cli_real or bool(ini_real)
    update = cli_update or bool(ini_update)
    report = cli_report if cli_report is not None else (ini_report if ini_report else "none")
    report_path = cli_report_path if cli_report_path is not None else (ini_report_path if ini_report_path else "quantum-report")

    return QuantumConfig(
        shots=shots,
        significance=significance,
        slow=slow,
        real=real,
        update_snapshots=update,
        report=report,
        report_path=report_path,
    )
