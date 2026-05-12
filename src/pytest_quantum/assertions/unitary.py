"""Unitary-level assertions for quantum circuits.

These functions compare circuits at the level of their unitary matrix — the
most rigorous form of correctness check for deterministic quantum operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytest_quantum.converters.to_unitary import (
    _reverse_qubit_order,
    to_unitary,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def assert_unitary(
    circuit: object,
    expected: NDArray[np.complex128],
    *,
    atol: float = 1e-6,
    allow_global_phase: bool = True,
) -> None:
    """Assert that *circuit* implements the expected unitary matrix.

    Args:
        circuit:  Any supported quantum circuit (Qiskit, Cirq, Braket,
                  PennyLane).
        expected: Target unitary as a numpy array, shape ``(2**n, 2**n)``.
        atol:     Absolute tolerance for element-wise comparison
                  (default ``1e-6``).
        allow_global_phase: If ``True`` (default), circuits that differ only
            by a global phase ``e^{iθ}`` are considered equivalent.  This is
            physically correct because global phase is not observable.

    Raises:
        AssertionError: If the circuit's unitary does not match *expected*.
        TypeError: If *circuit* is not a recognised framework type.

    Example::

        import numpy as np
        from pytest_quantum import assert_unitary

        HADAMARD = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


        def test_h_gate():
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(1)
            qc.h(0)
            assert_unitary(qc, HADAMARD)
    """
    actual = to_unitary(circuit)
    expected_arr = np.asarray(expected, dtype=np.complex128)

    if actual.shape != expected_arr.shape:
        raise AssertionError(
            f"Unitary shape mismatch.\n"
            f"  Circuit produces : {actual.shape}\n"
            f"  Expected         : {expected_arr.shape}\n"
            f"  Hint: check qubit count and ordering."
        )

    from pytest_quantum._internal import _unitaries_equivalent

    if _unitaries_equivalent(actual, expected_arr, atol=atol, allow_global_phase=allow_global_phase):
        return

    max_diff = float(np.max(np.abs(actual - expected_arr)))
    raise AssertionError(
        f"Circuit does not implement the expected unitary.\n"
        f"  Max |difference| : {max_diff:.2e}   (tolerance: {atol:.2e})\n"
        f"  allow_global_phase = {allow_global_phase}\n"
        f"  Hint: use allow_global_phase=True if you only care about "
        f"physical equivalence."
    )


def assert_circuits_equivalent(
    circuit_a: object,
    circuit_b: object,
    *,
    atol: float = 1e-6,
) -> None:
    """Assert that two circuits implement the same unitary, up to global phase.

    Works across frameworks — you can compare a Qiskit circuit against a Cirq
    circuit, a Braket circuit, or a PennyLane QNode.

    For two Qiskit circuits, ``mqt.qcec`` is used automatically when
    installed (faster, exact verification via decision diagrams / ZX-calculus).
    For cross-framework comparison the circuits are both converted to numpy
    matrices and compared numerically.

    Args:
        circuit_a: First circuit (any supported framework).
        circuit_b: Second circuit (any supported framework).
        atol:      Absolute tolerance for the numpy fallback comparison
                   (default ``1e-6``).

    Raises:
        AssertionError: If the circuits implement different unitaries.
        TypeError: If either argument is not a recognised circuit type.

    Example::

        from pytest_quantum import assert_circuits_equivalent


        def test_cnot_cross_framework():
            import cirq
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2)
            qc.cx(0, 1)

            q0, q1 = cirq.LineQubit.range(2)
            cc = cirq.Circuit(cirq.CNOT(q0, q1))

            assert_circuits_equivalent(qc, cc)
    """
    type_a = type(circuit_a).__module__
    type_b = type(circuit_b).__module__

    # Fast path: both Qiskit → use mqt.qcec (exact, scalable)
    if type_a.startswith("qiskit") and type_b.startswith("qiskit"):
        result = _qcec_verify(circuit_a, circuit_b)
        if result == "not_equivalent":
            raise AssertionError("Circuits are NOT equivalent (verified by mqt.qcec).")
        if result == "equivalent":
            return
        # result == "no_information" → fall through to numpy comparison

    # General path: convert both to unitary matrices
    u_a = to_unitary(circuit_a)
    u_b = to_unitary(circuit_b)

    # Normalize qubit ordering for cross-framework comparison.
    # Qiskit = little-endian, Cirq = big-endian, Pytket = big-endian.
    from pytest_quantum.adapters import get_adapter

    try:
        adapter_a = get_adapter(circuit_a)
        big_endian_a = adapter_a.big_endian
    except TypeError:
        big_endian_a = False
    try:
        adapter_b = get_adapter(circuit_b)
        big_endian_b = adapter_b.big_endian
    except TypeError:
        big_endian_b = False

    if not big_endian_a and big_endian_b:
        u_a = _reverse_qubit_order(u_a)
    elif big_endian_a and not big_endian_b:
        u_b = _reverse_qubit_order(u_b)

    if u_a.shape != u_b.shape:
        raise AssertionError(
            f"Circuits act on different-sized Hilbert spaces.\n"
            f"  circuit_a: {u_a.shape}  circuit_b: {u_b.shape}"
        )

    from pytest_quantum._internal import _unitaries_equivalent

    if _unitaries_equivalent(u_a, u_b, atol=atol, allow_global_phase=True):
        return

    max_diff = float(np.max(np.abs(u_a - u_b)))
    raise AssertionError(
        f"Circuits are NOT equivalent.\n"
        f"  Max |U_a - U_b|: {max_diff:.2e}   (tolerance: {atol:.2e})"
    )


def assert_transpilation_preserves_semantics(
    circuit: object,
    backend: object,
    *,
    optimization_level: int = 1,
    atol: float = 1e-6,
) -> None:
    """Assert that transpiling a Qiskit circuit preserves its unitary.

    Transpiles *circuit* for *backend* and verifies the resulting circuit
    implements the same unitary (up to global phase).

    Args:
        circuit:            Qiskit QuantumCircuit (must be unitary, no
                            measurements).
        backend:            Qiskit backend or ``FakeBackend`` target.
        optimization_level: Qiskit transpiler optimisation level 0-3
                            (default 1).
        atol:               Tolerance for unitary comparison (default 1e-6).

    Raises:
        AssertionError:      If transpiled circuit has different unitary.
        NotImplementedError: For non-Qiskit circuits.
        ImportError:         If qiskit is not installed.

    Example::

        from qiskit import QuantumCircuit
        from qiskit.providers.fake_provider import GenericBackendV2
        from pytest_quantum import assert_transpilation_preserves_semantics

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        backend = GenericBackendV2(num_qubits=2)
        assert_transpilation_preserves_semantics(qc, backend)
    """
    module = type(circuit).__module__
    if not module.startswith("qiskit"):
        raise NotImplementedError(
            f"assert_transpilation_preserves_semantics only supports Qiskit circuits; "
            f"got {module!r}"
        )
    try:
        from qiskit import transpile

        original_U = to_unitary(circuit)
        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
        )
        assert_unitary(transpiled, original_U, atol=atol, allow_global_phase=True)
    except ImportError as exc:
        raise ImportError(
            "qiskit is required: pip install pytest-quantum[qiskit]"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _qcec_verify(circuit_a: object, circuit_b: object) -> str:
    """Run mqt.qcec if available.  Returns 'equivalent', 'not_equivalent',
    or 'no_information'."""
    try:
        from mqt import qcec
    except ImportError:
        return "no_information"

    result = qcec.verify(circuit_a, circuit_b)
    result_str = str(result.equivalence).lower()
    if "not_equivalent" in result_str:
        return "not_equivalent"
    if "equivalent" in result_str:
        return "equivalent"
    return "no_information"
