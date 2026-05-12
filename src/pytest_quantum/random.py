"""Random quantum state and circuit generators for property-based testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def random_statevector(
    n_qubits: int,
    *,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Generate a random normalised statevector (Haar-random pure state).

    Args:
        n_qubits: Number of qubits (statevector length = 2**n_qubits).
        seed:     Optional random seed for reproducibility.

    Returns:
        Normalised complex128 array of shape ``(2**n_qubits,)``.

    Example::

        from pytest_quantum.random import random_statevector

        sv = random_statevector(2, seed=0)
        assert sv.shape == (4,)
        assert abs(sum(abs(sv) ** 2) - 1) < 1e-12
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    sv = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    result: NDArray[np.complex128] = (sv / np.linalg.norm(sv)).astype(np.complex128)
    return result


def random_density_matrix(
    n_qubits: int,
    rank: int | None = None,
    *,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Generate a random valid density matrix (PSD, trace 1).

    Args:
        n_qubits: Number of qubits (matrix size = 2**n_qubits × 2**n_qubits).
        rank:     Matrix rank (default: full rank = 2**n_qubits).
                  Use ``rank=1`` for a random pure state density matrix.
        seed:     Optional random seed.

    Returns:
        Complex128 array of shape ``(2**n_qubits, 2**n_qubits)``.

    Example::

        from pytest_quantum.random import random_density_matrix

        rho = random_density_matrix(1, seed=42)
        assert rho.shape == (2, 2)
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    r = rank if rank is not None else dim
    A = rng.standard_normal((dim, r)) + 1j * rng.standard_normal((dim, r))
    rho = A @ A.conj().T
    rho_n: NDArray[np.complex128] = (rho / np.trace(rho)).astype(np.complex128)
    return rho_n


def random_unitary(
    n_qubits: int,
    *,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Generate a Haar-random unitary matrix (CUE).

    Uses QR decomposition of the Ginibre ensemble with phase correction to
    guarantee an exact Haar distribution.

    Args:
        n_qubits: Number of qubits (matrix size = 2**n_qubits × 2**n_qubits).
        seed:     Optional random seed.

    Returns:
        Unitary complex128 array of shape ``(2**n_qubits, 2**n_qubits)``.

    Example::

        from pytest_quantum.random import random_unitary
        import numpy as np

        U = random_unitary(2, seed=0)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    # Phase correction: multiply each column by the sign of the diagonal of R
    phases = np.diag(R) / np.abs(np.diag(R))
    result: NDArray[np.complex128] = (Q * phases).astype(np.complex128)
    return result


def random_kraus_channel(
    n_qubits: int,
    n_kraus: int = 4,
    *,
    seed: int | None = None,
) -> list[NDArray[np.complex128]]:
    """Generate random valid Kraus operators for a CPTP channel.

    Constructs a random Stinespring isometry and extracts Kraus operators
    that satisfy the completeness relation ``∑ K†K = I``.

    Args:
        n_qubits: Number of qubits.
        n_kraus:  Number of Kraus operators (default 4).
        seed:     Optional random seed.

    Returns:
        List of *n_kraus* complex128 arrays of shape
        ``(2**n_qubits, 2**n_qubits)``.

    Example::

        from pytest_quantum.random import random_kraus_channel
        from pytest_quantum import assert_channel_is_cptp

        kraus = random_kraus_channel(1, seed=7)
        assert_channel_is_cptp(kraus)
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    total_rows = n_kraus * dim
    # Build a tall random matrix and QR-decompose to get an isometry
    A = rng.standard_normal((total_rows, dim)) + 1j * rng.standard_normal(
        (total_rows, dim)
    )
    V, _ = np.linalg.qr(A)
    # Each block of `dim` rows is one Kraus operator
    return [V[i * dim : (i + 1) * dim, :].astype(np.complex128) for i in range(n_kraus)]


def depolarizing_kraus(
    n_qubits: int,
    error_rate: float,
) -> list[NDArray[np.complex128]]:
    """Return Kraus operators for the single-qubit depolarising channel.

    Channel definition::

        E(ρ) = (1−p)ρ + (p/3)(XρX + YρY + ZρZ)

    Args:
        n_qubits:   Must be 1; the depolarising channel is defined per qubit.
        error_rate: Depolarisation probability *p* in [0, 1].

    Returns:
        List of 4 Kraus operators:
        ``[√(1−p)·I, √(p/3)·X, √(p/3)·Y, √(p/3)·Z]``.

    Raises:
        ValueError: If *n_qubits* ≠ 1 or *error_rate* is not in [0, 1].

    Example::

        from pytest_quantum.random import depolarizing_kraus
        from pytest_quantum import assert_channel_is_cptp

        assert_channel_is_cptp(depolarizing_kraus(1, 0.1))
    """
    if n_qubits != 1:
        raise ValueError(
            "depolarizing_kraus only supports single-qubit channels (n_qubits=1)"
        )
    if not (0 <= error_rate <= 1):
        raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")
    p = error_rate
    eye2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return [
        np.sqrt(1 - p) * eye2,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]


def random_qiskit_circuit(
    n_qubits: int,
    depth: int,
    *,
    gate_set: frozenset[str] | None = None,
    seed: int | None = None,
) -> object:
    """Generate a random Qiskit ``QuantumCircuit``.

    Args:
        n_qubits: Number of qubits.
        depth: Number of gate layers.
        gate_set: Optional frozenset of gate names. Defaults to
            ``{"h", "cx", "rz", "x", "y", "z", "s", "t", "swap"}``.
        seed: Random seed for reproducibility.

    Returns:
        A ``qiskit.QuantumCircuit`` instance.

    Raises:
        ImportError: If Qiskit is not installed.

    Example::

        qc = random_qiskit_circuit(3, 5, seed=42)
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError(
            "Qiskit is required for random_qiskit_circuit. "
            "Install it with: pip install pytest-quantum[qiskit]"
        )

    if gate_set is None:
        gate_set = frozenset({"h", "cx", "rz", "x", "y", "z", "s", "t", "swap"})

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    gate_list = sorted(gate_set)
    single_gates = [g for g in gate_list if g not in ("cx", "swap", "cz")]
    two_gates = [g for g in gate_list if g in ("cx", "swap", "cz")]

    for _ in range(depth):
        gate = rng.choice(gate_list)
        if gate in ("cx", "swap", "cz") and n_qubits >= 2:
            qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
            if gate == "cx":
                qc.cx(qubits[0], qubits[1])
            elif gate == "swap":
                qc.swap(qubits[0], qubits[1])
            elif gate == "cz":
                qc.cz(qubits[0], qubits[1])
        elif gate in ("cx", "swap", "cz") and n_qubits < 2:
            # Two-qubit gate on single qubit: fall back to single-qubit gate
            qubit = int(rng.integers(n_qubits))
            if single_gates:
                fallback = rng.choice(single_gates)
                if fallback == "rz":
                    angle = float(rng.uniform(0, 2 * np.pi))
                    qc.rz(angle, qubit)
                else:
                    getattr(qc, fallback)(qubit)
            else:
                qc.h(qubit)
        elif gate == "rz":
            qubit = int(rng.integers(n_qubits))
            angle = float(rng.uniform(0, 2 * np.pi))
            qc.rz(angle, qubit)
        elif single_gates:
            qubit = int(rng.integers(n_qubits))
            getattr(qc, gate)(qubit)
        else:
            # Fallback: apply H
            qubit = int(rng.integers(n_qubits))
            qc.h(qubit)

    return qc


def random_cirq_circuit(
    n_qubits: int,
    depth: int,
    *,
    gate_set: frozenset[str] | None = None,
    seed: int | None = None,
) -> object:
    """Generate a random Cirq ``Circuit``.

    Args:
        n_qubits: Number of qubits.
        depth: Number of gate layers.
        gate_set: Optional frozenset of gate names. Defaults to
            ``{"H", "CNOT", "Rz", "X", "Y", "Z", "S", "T", "SWAP"}``.
        seed: Random seed for reproducibility.

    Returns:
        A ``cirq.Circuit`` instance.

    Raises:
        ImportError: If Cirq is not installed.

    Example::

        circuit = random_cirq_circuit(3, 5, seed=42)
    """
    try:
        import cirq
    except ImportError:
        raise ImportError(
            "Cirq is required for random_cirq_circuit. "
            "Install it with: pip install pytest-quantum[cirq]"
        )

    if gate_set is None:
        gate_set = frozenset({"H", "CNOT", "Rz", "X", "Y", "Z", "S", "T", "SWAP"})

    rng = np.random.default_rng(seed)
    qubits = cirq.LineQubit.range(n_qubits)
    gate_map = {
        "H": cirq.H, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z,
        "S": cirq.S, "T": cirq.T,
    }
    two_qubit_gates = {"CNOT", "SWAP", "CZ"}
    gate_list = sorted(gate_set)
    ops = []

    for _ in range(depth):
        gate_name = rng.choice(gate_list)
        if gate_name in two_qubit_gates and n_qubits >= 2:
            idxs = rng.choice(n_qubits, size=2, replace=False)
            q0, q1 = qubits[idxs[0]], qubits[idxs[1]]
            if gate_name == "CNOT":
                ops.append(cirq.CNOT(q0, q1))
            elif gate_name == "SWAP":
                ops.append(cirq.SWAP(q0, q1))
            elif gate_name == "CZ":
                ops.append(cirq.CZ(q0, q1))
        elif gate_name == "Rz":
            qubit = qubits[int(rng.integers(n_qubits))]
            angle = float(rng.uniform(0, 2 * np.pi))
            ops.append(cirq.rz(angle)(qubit))
        elif gate_name in gate_map:
            qubit = qubits[int(rng.integers(n_qubits))]
            ops.append(gate_map[gate_name](qubit))
        else:
            qubit = qubits[int(rng.integers(n_qubits))]
            ops.append(cirq.H(qubit))

    return cirq.Circuit(ops)


def random_braket_circuit(
    n_qubits: int,
    depth: int,
    *,
    seed: int | None = None,
) -> object:
    """Generate a random Braket ``Circuit``.

    Args:
        n_qubits: Number of qubits.
        depth: Number of gate layers.
        seed: Random seed for reproducibility.

    Returns:
        A ``braket.circuits.Circuit`` instance.

    Raises:
        ImportError: If Braket SDK is not installed.

    Example::

        circuit = random_braket_circuit(3, 5, seed=42)
    """
    try:
        from braket.circuits import Circuit, gates
    except ImportError:
        raise ImportError(
            "Amazon Braket SDK is required for random_braket_circuit. "
            "Install it with: pip install pytest-quantum[braket]"
        )

    rng = np.random.default_rng(seed)
    circuit = Circuit()
    gate_names = ["h", "x", "y", "z", "s", "t", "cnot", "swap", "rz"]
    single_gates = {"h": gates.H, "x": gates.X, "y": gates.Y,
                    "z": gates.Z, "s": gates.S, "t": gates.T}
    two_gates = {"cnot", "swap"}

    for _ in range(depth):
        gate_name = rng.choice(gate_names)
        if gate_name in two_gates and n_qubits >= 2:
            qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
            if gate_name == "cnot":
                circuit.cnot(qubits[0], qubits[1])
            elif gate_name == "swap":
                circuit.swap(qubits[0], qubits[1])
        elif gate_name == "rz":
            qubit = int(rng.integers(n_qubits))
            angle = float(rng.uniform(0, 2 * np.pi))
            circuit.rz(qubit, angle)
        elif gate_name in single_gates:
            qubit = int(rng.integers(n_qubits))
            getattr(circuit, gate_name)(qubit)
        else:
            qubit = int(rng.integers(n_qubits))
            circuit.h(qubit)

    return circuit


def random_pennylane_circuit(
    n_qubits: int,
    depth: int,
    *,
    seed: int | None = None,
) -> object:
    """Generate a random PennyLane ``QNode``.

    Args:
        n_qubits: Number of qubits.
        depth: Number of gate layers.
        seed: Random seed for reproducibility.

    Returns:
        A ``pennylane.QNode`` instance.

    Raises:
        ImportError: If PennyLane is not installed.

    Example::

        qnode = random_pennylane_circuit(3, 5, seed=42)
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError(
            "PennyLane is required for random_pennylane_circuit. "
            "Install it with: pip install pytest-quantum[pennylane]"
        )

    rng = np.random.default_rng(seed)
    dev = qml.device("default.qubit", wires=n_qubits)

    # Pre-generate the random gate sequence
    gate_names = ["H", "X", "Y", "Z", "S", "T", "CNOT", "SWAP", "RZ"]
    single_gates = {"H": qml.Hadamard, "X": qml.PauliX, "Y": qml.PauliY,
                    "Z": qml.PauliZ, "S": qml.S, "T": qml.T}
    two_gates = {"CNOT", "SWAP"}

    gate_sequence = []
    for _ in range(depth):
        gate_name = rng.choice(gate_names)
        if gate_name in two_gates and n_qubits >= 2:
            qubits = rng.choice(n_qubits, size=2, replace=False).tolist()
            gate_sequence.append((gate_name, qubits, None))
        elif gate_name == "RZ":
            qubit = int(rng.integers(n_qubits))
            angle = float(rng.uniform(0, 2 * np.pi))
            gate_sequence.append((gate_name, [qubit], angle))
        else:
            qubit = int(rng.integers(n_qubits))
            gate_sequence.append((gate_name, [qubit], None))

    @qml.qnode(dev)
    def circuit():
        for gate_name, qubits, param in gate_sequence:
            if gate_name == "CNOT":
                qml.CNOT(wires=qubits)
            elif gate_name == "SWAP":
                qml.SWAP(wires=qubits)
            elif gate_name == "RZ":
                qml.RZ(param, wires=qubits[0])
            elif gate_name in single_gates:
                single_gates[gate_name](wires=qubits[0])
        return qml.state()

    return circuit
