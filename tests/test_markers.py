"""Tests for shots/significance marker wiring."""

from __future__ import annotations

import pytest


class TestShotsMarker:
    def test_shots_marker_overrides_default(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.shots(2048)
            def test_shots(quantum_shots):
                assert quantum_shots == 2048
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_significance_marker_overrides_default(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.significance(0.01)
            def test_sig(quantum_significance):
                assert quantum_significance == 0.01
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_cli_shots_without_marker(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            def test_shots(quantum_shots):
                assert quantum_shots == 4096
        """)
        result = pytester.runpytest("--quantum-shots=4096", "-v")
        result.assert_outcomes(passed=1)

    def test_marker_overrides_cli(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile("""
            import pytest

            @pytest.mark.shots(2048)
            def test_shots(quantum_shots):
                assert quantum_shots == 2048
        """)
        result = pytester.runpytest("--quantum-shots=4096", "-v")
        result.assert_outcomes(passed=1)
