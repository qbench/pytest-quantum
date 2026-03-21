"""OpenQASM round-trip assertions."""

from __future__ import annotations


def assert_qasm_roundtrip(
    circuit: object,
    *,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert that a circuit survives an export/import round-trip.

    For Qiskit: uses OpenQASM 3 (``qiskit.qasm3.dumps`` / ``qasm3.loads``).
    For Cirq:   uses Cirq's native JSON serialisation (``cirq.to_json`` /
                ``cirq.read_json``), which provides an exact identity
                round-trip for all standard Cirq circuits.

    Args:
        circuit:           Qiskit QuantumCircuit or cirq.Circuit.
        atol:              Tolerance for unitary comparison (default 1e-6).
        allow_global_phase: Ignore global phase in comparison (default True).

    Raises:
        AssertionError:      If re-imported circuit has a different unitary.
        NotImplementedError: For unsupported frameworks.
        ImportError:         If the required package is missing.
    """
    module = type(circuit).__module__

    if module.startswith("qiskit"):
        try:
            from qiskit import qasm3

            from pytest_quantum.converters.to_unitary import to_unitary

            original_U = to_unitary(circuit)
            qasm_str = qasm3.dumps(circuit)
            reimported = qasm3.loads(qasm_str)
        except ImportError as exc:
            raise ImportError(
                "qiskit-qasm3-import is required for QASM round-trip: "
                "pip install qiskit-qasm3-import"
            ) from exc

    elif module.startswith("cirq"):
        try:
            import cirq

            from pytest_quantum.converters.to_unitary import to_unitary

            original_U = to_unitary(circuit)
            # Use Cirq's native JSON serialisation for a true identity round-trip.
            # cirq.contrib.qasm_import is not available in modern Cirq.
            json_str = cirq.to_json(circuit)
            reimported = cirq.read_json(json_text=json_str)
        except ImportError as exc:
            raise ImportError(
                "cirq is required for Cirq round-trip: pip install cirq"
            ) from exc

    else:
        raise NotImplementedError(
            f"assert_qasm_roundtrip supports Qiskit and Cirq; got {module!r}. "
            "For Braket, use circuit.to_unitary() manually."
        )

    # Compare unitaries using the existing assert_unitary helper
    from pytest_quantum.assertions.unitary import assert_unitary

    assert_unitary(
        reimported,
        original_U,
        atol=atol,
        allow_global_phase=allow_global_phase,
    )
