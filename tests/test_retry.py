"""Tests for quantum_retry marker."""
from __future__ import annotations

import pytest


class TestQuantumRetry:
    def test_retry_passes_on_second_attempt(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            _counter = {"n": 0}

            @pytest.mark.quantum_retry(n=3)
            def test_flaky():
                _counter["n"] += 1
                if _counter["n"] < 2:
                    raise AssertionError("flaky failure")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_retry_fails_after_all_attempts(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.quantum_retry(n=2)
            def test_always_fails():
                raise AssertionError("always fails")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)

    def test_no_retry_on_pass(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            _counter = {"n": 0}

            @pytest.mark.quantum_retry(n=3)
            def test_passes():
                _counter["n"] += 1
                assert _counter["n"] == 1  # Only called once
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)
