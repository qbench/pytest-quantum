"""Pytest fixtures provided by pytest-quantum.

Fixtures are automatically available in any test session after installing
pytest-quantum — no import required.  Simply declare them as test parameters:

    def test_something(aer_simulator, graphix_backend):
        ...

Available fixtures
------------------
* ``aer_simulator``           — Qiskit AerSimulator (shot-based)
* ``aer_statevector_simulator`` — Qiskit AerSimulator (statevector mode)
* ``cirq_simulator``          — cirq.Simulator
* ``braket_simulator``        — Amazon Braket LocalSimulator
* ``graphix_backend``         — Graphix PatternSimulator wrapper
* ``pennylane_device``        — Factory: ``pennylane_device(wires=2)``

All fixtures use ``scope="session"`` — the backend is initialised once per
test run and reused, which avoids repeated startup costs.

Missing SDK
-----------
If the required SDK for a fixture is not installed, the test is automatically
skipped with a clear message telling the user which extra to install:

    pip install pytest-quantum[qiskit]   # for aer_simulator
    pip install pytest-quantum[cirq]     # for cirq_simulator
    pip install pytest-quantum[braket]   # for braket_simulator
    pip install pytest-quantum[graphix]  # for graphix_backend
    pip install pytest-quantum[pennylane]# for pennylane_device
"""

from __future__ import annotations
