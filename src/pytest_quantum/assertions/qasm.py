"""OpenQASM round-trip assertions."""

from __future__ import annotations


def assert_qasm_roundtrip(
    circuit: object,
    *,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert that a circuit survives an OpenQASM export/import round-trip.

    Exports the circuit to QASM, re-imports it, and verifies the unitary
    is unchanged.

    Supported frameworks:
    - Qiskit: uses ``qiskit.qasm3.dumps`` / ``qasm3.loads``
    - Cirq:   uses ``cirq.qasm`` / ``cirq.contrib.qasm_import.circuit_from_qasm``

    Args:
        circuit:           Qiskit QuantumCircuit or cirq.Circuit.
        atol:              Tolerance for unitary comparison (default 1e-6).
        allow_global_phase: Ignore global phase in comparison (default True).

    Raises:
        AssertionError:      If re-imported circuit has a different unitary.
        NotImplementedError: For unsupported frameworks.
        ImportError:         If the required QASM import package is missing.
    """
    module = type(circuit).__module__

    if module.startswith("qiskit"):
        try:
            from qiskit import qasm3

            from pytest_quantum.converters.to_unitary import to_unitary

            original_U = to_unitary(circuit)
            qasm_str = qasm3.dumps(circuit)  # type: ignore[attr-defined]
            reimported = qasm3.loads(qasm_str)  # type: ignore[attr-defined]
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
            qasm_str = cirq.qasm(circuit)  # type: ignore[arg-type]
            from cirq.contrib.qasm_import import (
                circuit_from_qasm,  # type: ignore[import-untyped]
            )

            reimported = circuit_from_qasm(qasm_str)
        except ImportError as exc:
            raise ImportError(
                "cirq is required for QASM round-trip: pip install cirq"
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
