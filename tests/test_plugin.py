"""Tests for the pytest plugin itself — markers, CLI options, hooks.

Uses pytester (the modern replacement for testdir) to run pytest in an
isolated subprocess and assert on the outcomes.
"""

from __future__ import annotations

import pytest


class TestMarkerRegistration:
    """Verify that custom markers are registered and --strict-markers passes."""

    def test_quantum_marker_no_warning(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.quantum
            def test_something():
                assert 1 + 1 == 2
        """)
        result = pytester.runpytest("--strict-markers", "-v")
        result.assert_outcomes(passed=1)

    def test_quantum_slow_marker_skipped_by_default(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.quantum_slow
            def test_slow():
                assert True
        """)
        result = pytester.runpytest("-v", "-rs")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*quantum-slow*"])

    def test_quantum_slow_runs_with_flag(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.quantum_slow
            def test_slow():
                assert True
        """)
        result = pytester.runpytest("--quantum-slow", "-v")
        result.assert_outcomes(passed=1)

    def test_shots_marker_no_warning(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.shots(2048)
            def test_something():
                assert True
        """)
        result = pytester.runpytest("--strict-markers", "-v")
        result.assert_outcomes(passed=1)

    def test_significance_marker_no_warning(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.significance(0.01)
            def test_something():
                assert True
        """)
        result = pytester.runpytest("--strict-markers", "-v")
        result.assert_outcomes(passed=1)


class TestCliOptions:
    """Verify that CLI options are added and accessible."""

    def test_quantum_shots_option_exists(self, pytester: pytest.Pytester) -> None:
        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*--quantum-shots*"])

    def test_quantum_significance_option_exists(
        self, pytester: pytest.Pytester
    ) -> None:
        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*--quantum-significance*"])

    def test_quantum_slow_option_exists(self, pytester: pytest.Pytester) -> None:
        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*--quantum-slow*"])


class TestPluginLoading:
    """Verify the plugin loads cleanly and does not break vanilla tests."""

    def test_vanilla_test_unaffected(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            def test_add():
                assert 1 + 1 == 2

            def test_str():
                assert "hello".upper() == "HELLO"
        """)
        result = pytester.runpytest()
        result.assert_outcomes(passed=2)

    def test_plugin_version_accessible(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            def test_version():
                import pytest_quantum
                assert pytest_quantum.__version__ == "1.0.0"
        """)
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)
