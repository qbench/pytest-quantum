"""Tests for quantum reporting plugin."""

from __future__ import annotations

import json


class TestJsonReport:
    def test_json_report_created(self, pytester):
        """Test that --quantum-report=json creates a JSON file."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.quantum
            def test_quantum_example(quantum_shots):
                assert True
            """
        )
        result = pytester.runpytest("--quantum-report=json", "-v")
        result.assert_outcomes(passed=1)
        report_file = pytester.path / "quantum-report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert data["total_quantum_tests"] == 1
        assert data["passed"] == 1
        assert data["failed"] == 0

    def test_json_custom_path(self, pytester):
        """Test custom report path."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.quantum
            def test_q():
                pass
            """
        )
        result = pytester.runpytest(
            "--quantum-report=json",
            "--quantum-report-path=my-report",
            "-v",
        )
        result.assert_outcomes(passed=1)
        assert (pytester.path / "my-report.json").exists()

    def test_no_report_when_no_quantum_tests(self, pytester):
        """Test that no report is generated when there are no quantum tests."""
        pytester.makepyfile(
            """
            def test_regular():
                assert True
            """
        )
        result = pytester.runpytest("--quantum-report=json", "-v")
        result.assert_outcomes(passed=1)
        assert not (pytester.path / "quantum-report.json").exists()


class TestHtmlReport:
    def test_html_report_created(self, pytester):
        """Test that --quantum-report=html creates an HTML file."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.quantum
            def test_quantum_html():
                assert True
            """
        )
        result = pytester.runpytest("--quantum-report=html", "-v")
        result.assert_outcomes(passed=1)
        report_file = pytester.path / "quantum-report.html"
        assert report_file.exists()
        content = report_file.read_text()
        assert "Quantum Test Report" in content
        assert "test_quantum_html" in content


class TestTerminalSummary:
    def test_summary_printed(self, pytester):
        """Test that terminal summary is printed for quantum tests."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.quantum
            def test_q1():
                pass

            @pytest.mark.quantum
            def test_q2():
                pass
            """
        )
        result = pytester.runpytest("--quantum-report=json", "-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*Quantum Test Summary*"])
        result.stdout.fnmatch_lines(["*Total quantum tests: 2*"])

    def test_no_summary_without_quantum_tests(self, pytester):
        """Test that no summary is printed when there are no quantum tests."""
        pytester.makepyfile(
            """
            def test_regular():
                pass
            """
        )
        result = pytester.runpytest("--quantum-report=json", "-v")
        result.assert_outcomes(passed=1)
        assert "Quantum Test Summary" not in result.stdout.str()


class TestShotCounting:
    def test_shots_tracked(self, pytester):
        """Test that shots are tracked in the report."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.shots(1024)
            @pytest.mark.quantum
            def test_with_shots(quantum_shots):
                assert quantum_shots == 1024
            """
        )
        result = pytester.runpytest("--quantum-report=json", "-v")
        result.assert_outcomes(passed=1)
        report_file = pytester.path / "quantum-report.json"
        data = json.loads(report_file.read_text())
        assert data["total_shots"] == 1024

    def test_cli_shots_tracked_in_report(self, pytester):
        """Test that --quantum-shots is reflected in the report."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.quantum
            def test_with_cli_shots(quantum_shots):
                assert quantum_shots == 2048
            """
        )
        result = pytester.runpytest(
            "--quantum-report=json", "--quantum-shots=2048", "-v"
        )
        result.assert_outcomes(passed=1)
        report_file = pytester.path / "quantum-report.json"
        data = json.loads(report_file.read_text())
        assert data["total_shots"] == 2048


class TestIniConfiguration:
    def test_ini_report_format(self, pytester):
        """Test that quantum_report INI option triggers report generation."""
        pytester.makeini(
            """
            [pytest]
            quantum_report = json
            """
        )
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.quantum
            def test_q():
                pass
            """
        )
        # No --quantum-report CLI flag: INI should take effect
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)
        report_file = pytester.path / "quantum-report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert data["total_quantum_tests"] == 1

    def test_ini_report_path(self, pytester):
        """Test that quantum_report_path INI option is respected."""
        pytester.makeini(
            """
            [pytest]
            quantum_report = json
            quantum_report_path = custom-report
            """
        )
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.quantum
            def test_q():
                pass
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)
        assert (pytester.path / "custom-report.json").exists()
