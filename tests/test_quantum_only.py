"""Tests for --quantum-only flag."""
from __future__ import annotations

import pytest


class TestQuantumOnly:
    def test_quantum_only_selects_marked_tests(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.quantum
            def test_quantum():
                pass

            def test_normal():
                pass
        """)
        result = pytester.runpytest("--quantum-only", "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_quantum*PASSED*"])
        result.stdout.no_fnmatch_line("*test_normal*")

    def test_without_flag_runs_all(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.quantum
            def test_quantum():
                pass

            def test_normal():
                pass
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)
