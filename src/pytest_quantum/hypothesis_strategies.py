"""Hypothesis strategies for quantum states, distributions, and operators.

These strategies generate valid quantum objects for property-based testing.

Usage::

    from hypothesis import given
    from pytest_quantum.hypothesis_strategies import statevectors, density_matrices
    from pytest_quantum import assert_normalized, assert_hermitian


    @given(statevectors(n_qubits=1))
    def test_statevector_always_normalized(sv):
        assert_normalized(sv)


    @given(density_matrices(n_qubits=1))
    def test_density_matrix_always_hermitian(rho):
        assert_hermitian(rho)

Requires: pip install hypothesis
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

__all__ = [
    "braket_circuits",
    "cirq_circuits",
    "count_distributions",
    "density_matrices",
    "kraus_channels",
    "pennylane_circuits",
    "qiskit_circuits",
    "statevectors",
    "unitary_matrices",
]


def _require_hypothesis() -> None:
    if not _HYPOTHESIS_AVAILABLE:
        raise ImportError(
            "hypothesis is required for quantum strategies. "
            "Install it with: pip install hypothesis"
        )


def statevectors(n_qubits: int = 1) -> st.SearchStrategy[NDArray[np.complex128]]:
    """Strategy generating random normalized statevectors of shape (2^n_qubits,).

    All generated statevectors satisfy ||psi||_2 = 1.

    Args:
        n_qubits: Number of qubits (default 1). Dimension = 2^n_qubits.

    Returns:
        Hypothesis SearchStrategy yielding complex128 numpy arrays.

    Example::

        from hypothesis import given
        from pytest_quantum.hypothesis_strategies import statevectors
        from pytest_quantum import assert_normalized


        @given(statevectors(n_qubits=2))
        def test_always_normalized(sv):
            assert_normalized(sv)
    """
    _require_hypothesis()
    dim = 2**n_qubits

    @st.composite
    def _strategy(draw: st.DrawFn) -> NDArray[np.complex128]:
        floats = st.floats(
            min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        )
        real = draw(arrays(np.float64, (dim,), elements=floats))
        imag = draw(arrays(np.float64, (dim,), elements=floats))
        sv = real + 1j * imag
        norm = np.linalg.norm(sv)
        if norm < 1e-10:
            sv = np.zeros(dim, dtype=np.complex128)
            sv[0] = 1.0
            return sv
        return (sv / norm).astype(np.complex128)

    return _strategy()


def density_matrices(
    n_qubits: int = 1, rank: int | None = None
) -> st.SearchStrategy[NDArray[np.complex128]]:
    """Strategy generating random valid density matrices of shape (2^n, 2^n).

    All generated matrices are:
    - Hermitian (rho = rho†)
    - Positive semi-definite (all eigenvalues >= 0)
    - Trace 1 (Tr(rho) = 1)

    Args:
        n_qubits: Number of qubits (default 1).
        rank: Matrix rank. None = full rank. rank=1 = pure states only.

    Returns:
        Hypothesis SearchStrategy yielding complex128 numpy arrays.
    """
    _require_hypothesis()
    from pytest_quantum.random import random_density_matrix

    dim = 2**n_qubits
    r = rank if rank is not None else dim

    @st.composite
    def _strategy(draw: st.DrawFn) -> NDArray[np.complex128]:
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        return random_density_matrix(n_qubits, rank=r, seed=seed)

    return _strategy()


def unitary_matrices(n_qubits: int = 1) -> st.SearchStrategy[NDArray[np.complex128]]:
    """Strategy generating random Haar-random unitary matrices of shape (2^n, 2^n).

    All generated matrices satisfy U†U = I.

    Args:
        n_qubits: Number of qubits (default 1).

    Returns:
        Hypothesis SearchStrategy yielding complex128 numpy arrays.
    """
    _require_hypothesis()
    from pytest_quantum.random import random_unitary

    @st.composite
    def _strategy(draw: st.DrawFn) -> NDArray[np.complex128]:
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        return random_unitary(n_qubits, seed=seed)

    return _strategy()


def count_distributions(
    outcomes: list[str] | None = None,
    total_shots: int = 1000,
    n_qubits: int = 1,
) -> st.SearchStrategy[dict[str, int]]:
    """Strategy generating random measurement count dictionaries.

    All generated count dicts have sum == total_shots.

    Args:
        outcomes: List of outcome strings (default: all n-qubit bitstrings).
        total_shots: Total measurement shots (default 1000).
        n_qubits: Number of qubits — used only if outcomes is None.

    Returns:
        Hypothesis SearchStrategy yielding dict[str, int].
    """
    _require_hypothesis()

    if outcomes is None:
        outcomes = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
    k = len(outcomes)
    _outcomes = outcomes  # capture for closure

    @st.composite
    def _strategy(draw: st.DrawFn) -> dict[str, int]:
        # Draw k-1 random cut points, sort, compute differences
        cuts = sorted(
            draw(st.lists(st.integers(0, total_shots), min_size=k - 1, max_size=k - 1))
        )
        boundaries = [0, *cuts, total_shots]
        counts = [boundaries[i + 1] - boundaries[i] for i in range(k)]
        return {_outcomes[i]: counts[i] for i in range(k)}

    return _strategy()


def kraus_channels(
    n_qubits: int = 1, n_kraus: int = 4
) -> st.SearchStrategy[list[NDArray[np.complex128]]]:
    """Strategy generating valid CPTP Kraus operator sets.

    All generated Kraus operator sets satisfy the completeness relation:
    SumK†K = I (CPTP).

    Args:
        n_qubits: Number of qubits (default 1).
        n_kraus: Number of Kraus operators (default 4).

    Returns:
        Hypothesis SearchStrategy yielding list of complex128 numpy arrays.
    """
    _require_hypothesis()
    from pytest_quantum.random import random_kraus_channel

    @st.composite
    def _strategy(draw: st.DrawFn) -> list[NDArray[np.complex128]]:
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        return random_kraus_channel(n_qubits, n_kraus=n_kraus, seed=seed)

    return _strategy()


@st.composite
def qiskit_circuits(
    draw: st.DrawFn,
    n_qubits: st.SearchStrategy[int] = st.integers(1, 4),
    depth: st.SearchStrategy[int] = st.integers(1, 5),
) -> object:
    """Hypothesis strategy that generates random Qiskit circuits.

    Args:
        n_qubits: Strategy for number of qubits. Default: 1–4.
        depth: Strategy for circuit depth. Default: 1–5.

    Returns:
        A ``qiskit.QuantumCircuit`` instance.

    Example::

        @given(qiskit_circuits())
        def test_circuit_has_unitary(qc):
            from pytest_quantum import assert_unitary
            u = to_unitary(qc)
            assert_unitary(u)
    """
    from pytest_quantum.random import random_qiskit_circuit

    nq = draw(n_qubits)
    d = draw(depth)
    seed = draw(st.integers(0, 2**32 - 1))
    return random_qiskit_circuit(nq, d, seed=seed)


@st.composite
def cirq_circuits(
    draw: st.DrawFn,
    n_qubits: st.SearchStrategy[int] = st.integers(1, 4),
    depth: st.SearchStrategy[int] = st.integers(1, 5),
) -> object:
    """Hypothesis strategy that generates random Cirq circuits.

    Args:
        n_qubits: Strategy for number of qubits. Default: 1–4.
        depth: Strategy for circuit depth. Default: 1–5.

    Returns:
        A ``cirq.Circuit`` instance.

    Example::

        @given(cirq_circuits())
        def test_circuit_has_unitary(circuit):
            u = cirq.unitary(circuit)
            assert u.shape[0] == u.shape[1]
    """
    from pytest_quantum.random import random_cirq_circuit

    nq = draw(n_qubits)
    d = draw(depth)
    seed = draw(st.integers(0, 2**32 - 1))
    return random_cirq_circuit(nq, d, seed=seed)


@st.composite
def braket_circuits(
    draw: st.DrawFn,
    n_qubits: st.SearchStrategy[int] = st.integers(1, 4),
    depth: st.SearchStrategy[int] = st.integers(1, 5),
) -> object:
    """Hypothesis strategy that generates random Braket circuits.

    Args:
        n_qubits: Strategy for number of qubits. Default: 1–4.
        depth: Strategy for circuit depth. Default: 1–5.

    Returns:
        A ``braket.circuits.Circuit`` instance.

    Example::

        @given(braket_circuits())
        def test_circuit_has_unitary(circuit):
            u = circuit.to_unitary()
            assert u.shape[0] == u.shape[1]
    """
    from pytest_quantum.random import random_braket_circuit

    nq = draw(n_qubits)
    d = draw(depth)
    seed = draw(st.integers(0, 2**32 - 1))
    return random_braket_circuit(nq, d, seed=seed)


@st.composite
def pennylane_circuits(
    draw: st.DrawFn,
    n_qubits: st.SearchStrategy[int] = st.integers(1, 4),
    depth: st.SearchStrategy[int] = st.integers(1, 5),
) -> object:
    """Hypothesis strategy that generates random PennyLane QNodes.

    Args:
        n_qubits: Strategy for number of qubits. Default: 1–4.
        depth: Strategy for circuit depth. Default: 1–5.

    Returns:
        A ``pennylane.QNode`` instance.

    Example::

        @given(pennylane_circuits())
        def test_qnode_returns_state(qnode):
            state = qnode()
            assert len(state) > 0
    """
    from pytest_quantum.random import random_pennylane_circuit

    nq = draw(n_qubits)
    d = draw(depth)
    seed = draw(st.integers(0, 2**32 - 1))
    return random_pennylane_circuit(nq, d, seed=seed)
