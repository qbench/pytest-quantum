"""Quantum test retry support for flaky statistical tests."""
from __future__ import annotations

import pytest


def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> object | None:
    """Retry tests marked with @pytest.mark.quantum_retry(n=N)."""
    marker = item.get_closest_marker("quantum_retry")
    if marker is None:
        return None  # default protocol

    max_attempts = marker.kwargs.get("n", marker.args[0] if marker.args else 3)

    from _pytest.runner import runtestprotocol

    for attempt in range(1, max_attempts + 1):
        # Store retry metadata on the item so reporting hooks can inspect it.
        item._quantum_retry_attempt = attempt  # type: ignore[attr-defined]
        item._quantum_retry_max = max_attempts  # type: ignore[attr-defined]
        # Mark as retried before the protocol runs so that makereport
        # (which fires during runtestprotocol) can see the flag.
        if attempt > 1:
            item._quantum_retried = True  # type: ignore[attr-defined]

        reports = runtestprotocol(item, nextitem=nextitem, log=False)
        # Check if test passed
        call_report = next((r for r in reports if r.when == "call"), None)
        test_failed = call_report is not None and call_report.failed

        if not test_failed or attempt == max_attempts:
            for report in reports:
                if test_failed and attempt == max_attempts and report.when == "call":
                    # Add retry info to the final failure report
                    report.sections.append(
                        ("quantum_retry", f"Failed after {max_attempts} attempts")
                    )
                item.ihook.pytest_runtest_logreport(report=report)
            return True  # We handled the protocol

        # Failed but have retries left — don't log, just retry
    return True
