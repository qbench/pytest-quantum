"""Test suite configuration for pytest-quantum.

Enables the pytester fixture (used to test the plugin itself) and adds any
shared fixtures needed across the test suite.
"""

from __future__ import annotations

# Enable pytester — the modern fixture for testing pytest plugins.
# This is required to use pytest.Pytester in tests.
pytest_plugins = ["pytester"]
