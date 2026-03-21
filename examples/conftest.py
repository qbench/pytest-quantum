"""
Example conftest.py for a quantum test suite using pytest-quantum.

Copy this file to your project root and customize as needed.
All pytest-quantum fixtures are auto-injected — no imports needed here.
"""

import pytest

# ---------------------------------------------------------------------------
# Register markers to prevent PytestUnknownMarkWarning
# (also add these to your pyproject.toml under [tool.pytest.ini_options])
# ---------------------------------------------------------------------------
# See pyproject.toml for marker registration:
#   [tool.pytest.ini_options]
#   markers = [
#       "quantum: mark test as quantum",
#       "quantum_slow: ...",
#       "quantum_snapshot: ...",
#   ]


# ---------------------------------------------------------------------------
# Optional: project-specific fixtures that wrap pytest-quantum fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def my_qiskit_sim(aer_simulator):
    """Wrap aer_simulator to add project-specific behavior."""
    # aer_simulator is auto-provided by pytest-quantum
    return aer_simulator


@pytest.fixture
def bell_circuit():
    """Reusable Bell state circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
