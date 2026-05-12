"""Quantum-specific test reporting: JSON, HTML, and terminal summary."""
from __future__ import annotations

import dataclasses
import html
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class QuantumSessionData:
    """Aggregated data for quantum test reporting."""

    total_quantum_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    retried: int = 0
    total_shots: int = 0
    frameworks_used: set[str] = field(default_factory=set)
    assertion_counts: dict[str, int] = field(default_factory=dict)
    benchmark_results: list[dict[str, Any]] = field(default_factory=list)
    test_details: list[dict[str, Any]] = field(default_factory=list)


class QuantumReportPlugin:
    """Pytest plugin that collects quantum test data and generates reports."""

    def __init__(self, report_format: str, report_path: str) -> None:
        self.report_format = report_format
        self.report_path = report_path
        self.data = QuantumSessionData()
        self._quantum_fixtures = frozenset({
            "aer_simulator", "aer_statevector_simulator", "aer_noise_simulator",
            "qiskit_sampler", "qiskit_estimator",
            "cirq_simulator", "cirq_sampler",
            "braket_simulator", "braket_cloud_device",
            "pennylane_device",
            "pytket_circuit_factory",
            "stim_sampler",
            "qutip_solver",
            "tequila_backend",
            "graphix_backend",
            "quantum_backend",
            "quantum_benchmark",
            "shot_budget",
            "benchmark_suite",
            "multi_backend_runner",
            "ibm_backend", "ionq_backend", "quantinuum_backend",
            "cuda_quantum_simulator", "qibo_backend",
        })
        self._framework_fixture_map = {
            "aer_simulator": "qiskit",
            "aer_statevector_simulator": "qiskit",
            "aer_noise_simulator": "qiskit",
            "qiskit_sampler": "qiskit",
            "qiskit_estimator": "qiskit",
            "cirq_simulator": "cirq",
            "cirq_sampler": "cirq",
            "braket_simulator": "braket",
            "braket_cloud_device": "braket",
            "pennylane_device": "pennylane",
            "pytket_circuit_factory": "pytket",
            "stim_sampler": "stim",
            "qutip_solver": "qutip",
            "tequila_backend": "tequila",
            "graphix_backend": "graphix",
            "ibm_backend": "qiskit",
            "ionq_backend": "ionq",
            "quantinuum_backend": "quantinuum",
            "cuda_quantum_simulator": "cuda_quantum",
            "qibo_backend": "qibo",
        }

    def _is_quantum_test(self, item: pytest.Item) -> bool:
        """Check if a test item uses quantum fixtures or markers."""
        # Check fixtures
        if hasattr(item, "fixturenames"):
            if self._quantum_fixtures & set(item.fixturenames):
                return True
        # Check markers
        for marker_name in ("quantum", "quantum_slow", "quantum_retry"):
            if item.get_closest_marker(marker_name):
                return True
        return False

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo) -> Any:  # type: ignore[type-arg]
        """Track quantum test results."""
        outcome = yield
        report = outcome.get_result()

        if report.when != "call":
            return

        if not self._is_quantum_test(item):
            return

        self.data.total_quantum_tests += 1

        if report.passed:
            self.data.passed += 1
        elif report.failed:
            self.data.failed += 1
        elif report.skipped:
            self.data.skipped += 1

        # Track shots
        shots = getattr(item, "_quantum_shots", None)
        if shots is not None:
            self.data.total_shots += int(shots)

        # Track frameworks
        if hasattr(item, "fixturenames"):
            for fixture_name in item.fixturenames:
                framework = self._framework_fixture_map.get(fixture_name)
                if framework:
                    self.data.frameworks_used.add(framework)

        # Track retries
        if getattr(item, "_quantum_retried", False):
            self.data.retried += 1

        # Store test detail
        self.data.test_details.append({
            "nodeid": item.nodeid,
            "outcome": report.outcome,
            "duration": report.duration,
        })

    def pytest_terminal_summary(
        self, terminalreporter: Any, exitstatus: int, config: pytest.Config
    ) -> None:
        """Print quantum test summary at end of session."""
        if self.data.total_quantum_tests == 0:
            return

        terminalreporter.section("Quantum Test Summary")
        terminalreporter.line(
            f"Total quantum tests: {self.data.total_quantum_tests} "
            f"({self.data.passed} passed, {self.data.failed} failed, "
            f"{self.data.skipped} skipped)"
        )
        if self.data.frameworks_used:
            terminalreporter.line(
                f"Frameworks tested: {', '.join(sorted(self.data.frameworks_used))}"
            )
        if self.data.total_shots > 0:
            terminalreporter.line(
                f"Total shots consumed: {self.data.total_shots:,}"
            )
        if self.data.retried > 0:
            terminalreporter.line(f"Retried tests: {self.data.retried}")

    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: int
    ) -> None:
        """Write report file at end of session."""
        if self.report_format == "none":
            return
        if self.data.total_quantum_tests == 0:
            return

        if self.report_format == "json":
            path = self.report_path + ".json"
            _generate_json_report(self.data, path)
        elif self.report_format == "html":
            path = self.report_path + ".html"
            _generate_html_report(self.data, path)


def _generate_json_report(data: QuantumSessionData, path: str) -> None:
    """Write quantum session data as JSON."""
    d = dataclasses.asdict(data)
    # Convert sets to sorted lists for JSON serialization
    d["frameworks_used"] = sorted(d["frameworks_used"])
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)


def _generate_html_report(data: QuantumSessionData, path: str) -> None:
    """Generate a self-contained HTML report."""
    frameworks = ", ".join(sorted(data.frameworks_used)) or "none"

    rows = []
    for detail in data.test_details:
        outcome_class = {
            "passed": "pass",
            "failed": "fail",
            "skipped": "skip",
        }.get(detail["outcome"], "")
        rows.append(
            f'<tr class="{outcome_class}">'
            f'<td>{html.escape(detail["nodeid"])}</td>'
            f'<td>{html.escape(detail["outcome"])}</td>'
            f'<td>{detail["duration"]:.3f}s</td>'
            f"</tr>"
        )

    html_content = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quantum Test Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2em; }}
  h1 {{ color: #2c3e50; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1em; margin: 1em 0; }}
  .card {{ background: #f8f9fa; border-radius: 8px; padding: 1em; text-align: center; }}
  .card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
  .card .label {{ color: #6c757d; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
  th, td {{ border: 1px solid #dee2e6; padding: 0.5em 1em; text-align: left; }}
  th {{ background: #e9ecef; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr.pass td:nth-child(2) {{ color: #28a745; font-weight: bold; }}
  tr.fail td:nth-child(2) {{ color: #dc3545; font-weight: bold; }}
  tr.skip td:nth-child(2) {{ color: #ffc107; font-weight: bold; }}
</style>
</head>
<body>
<h1>Quantum Test Report</h1>

<div class="summary">
  <div class="card"><div class="value">{data.total_quantum_tests}</div><div class="label">Total Tests</div></div>
  <div class="card"><div class="value" style="color:#28a745">{data.passed}</div><div class="label">Passed</div></div>
  <div class="card"><div class="value" style="color:#dc3545">{data.failed}</div><div class="label">Failed</div></div>
  <div class="card"><div class="value" style="color:#ffc107">{data.skipped}</div><div class="label">Skipped</div></div>
</div>

<h2>Framework Coverage</h2>
<p>Frameworks tested: <strong>{html.escape(frameworks)}</strong></p>
<p>Total shots consumed: <strong>{data.total_shots:,}</strong></p>
{f"<p>Retried tests: <strong>{data.retried}</strong></p>" if data.retried else ""}

<h2>Test Results</h2>
<table>
<thead><tr><th>Test</th><th>Outcome</th><th>Duration</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(html_content)
