"""Quantum error correction assertions."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _pauli_str_to_symplectic(pauli: str) -> NDArray[np.int8]:
    """Convert a Pauli string like 'XYZII' to symplectic representation.

    Returns a 1D array of length 2*n where the first n bits are the X component
    and the last n bits are the Z component.
    """
    n = len(pauli)
    vec = np.zeros(2 * n, dtype=np.int8)
    for i, c in enumerate(pauli):
        if c in ("X", "x"):
            vec[i] = 1
        elif c in ("Z", "z"):
            vec[n + i] = 1
        elif c in ("Y", "y"):
            vec[i] = 1
            vec[n + i] = 1
        elif c in ("I", "i"):
            pass
        else:
            raise ValueError(f"Invalid Pauli character: {c!r}")
    return vec


def _symplectic_weight(vec: NDArray[np.int8]) -> int:
    """Compute the weight of a Pauli in symplectic representation."""
    n = len(vec) // 2
    weight = 0
    for i in range(n):
        if vec[i] or vec[n + i]:
            weight += 1
    return weight


def _symplectic_product(a: NDArray[np.int8], b: NDArray[np.int8]) -> int:
    """Compute the symplectic inner product (mod 2)."""
    n = len(a) // 2
    return int(np.sum(a[:n] * b[n:] + a[n:] * b[:n]) % 2)


def _symplectic_multiply(a: NDArray[np.int8], b: NDArray[np.int8]) -> NDArray[np.int8]:
    """Multiply two Paulis in symplectic representation (mod 2, ignoring phase)."""
    return (a + b) % 2


def assert_code_distance(
    stabilizers: list[str],
    expected_distance: int,
) -> None:
    """Assert that a stabilizer code has the expected code distance.

    Computes the minimum weight of non-trivial logical operators by searching
    the normalizer of the stabilizer group. Feasible for small codes (up to
    ~15 qubits).

    Args:
        stabilizers: List of Pauli stabilizer strings (e.g. ``["XZZXI", "IXZZX"]``).
        expected_distance: Expected code distance.

    Raises:
        AssertionError: If the computed distance != *expected_distance*.

    Example::

        # 3-qubit bit-flip code
        assert_code_distance(["ZZI", "IZZ"], expected_distance=1)
    """
    if not stabilizers:
        raise ValueError("Stabilizer list must not be empty.")

    n = len(stabilizers[0])
    for s in stabilizers:
        if len(s) != n:
            raise ValueError(
                f"All stabilizers must have the same length. Got {len(s)} and {n}."
            )

    stab_vecs = np.array(
        [_pauli_str_to_symplectic(s) for s in stabilizers], dtype=np.int8
    )

    # Find minimum weight operator that commutes with all stabilizers
    # but is not in the stabilizer group
    min_distance = n + 1  # Start with impossible value

    # Generate all non-identity Paulis and check if they're in the normalizer
    # but not in the stabilizer group
    # For efficiency, we use the stabilizer group structure

    # Build stabilizer group (all products of stabilizers)
    k = len(stabilizers)
    stab_group = set()
    for r in range(k + 1):
        for combo in itertools.combinations(range(k), r):
            prod: np.ndarray[Any, np.dtype[np.int8]] = np.zeros(2 * n, dtype=np.int8)
            for idx in combo:
                prod = _symplectic_multiply(prod, stab_vecs[idx])
            stab_group.add(tuple(prod))

    # Search through all possible Paulis for normalizer elements
    # that are not in the stabilizer group
    for bits in range(1, 2 ** (2 * n)):
        candidate: np.ndarray[Any, np.dtype[np.int8]] = np.array(
            [(bits >> i) & 1 for i in range(2 * n)], dtype=np.int8
        )
        weight = _symplectic_weight(candidate)
        if weight >= min_distance:
            continue

        # Check if candidate commutes with all stabilizers (in normalizer)
        in_normalizer = all(_symplectic_product(candidate, sv) == 0 for sv in stab_vecs)
        if not in_normalizer:
            continue

        # Check if candidate is NOT in the stabilizer group
        if tuple(candidate) not in stab_group:
            min_distance = weight

    if min_distance > n:
        min_distance = 0  # No logical operators found — degenerate case

    if min_distance != expected_distance:
        raise AssertionError(
            f"Code distance is {min_distance}, expected {expected_distance}.\n"
            f"Stabilizers: {stabilizers}"
        )


def assert_syndrome_decoding_correct(
    stabilizers: list[str],
    error: NDArray[np.int8],
    decoder_correction: NDArray[np.int8],
) -> None:
    """Assert that a decoder's correction is equivalent to the error.

    Verifies that ``error * decoder_correction`` is in the stabilizer group,
    meaning the decoder found an equivalent correction (possibly differing by
    a stabilizer element).

    Args:
        stabilizers: List of Pauli stabilizer strings.
        error: Error as a symplectic vector (1D array of length 2*n) or
            Pauli string.
        decoder_correction: Decoder's correction as a symplectic vector or
            Pauli string.

    Raises:
        AssertionError: If the residual error is a non-trivial logical operator.

    Example::

        stabilizers = ["ZZI", "IZZ"]
        error = np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)  # X on qubit 0
        correction = np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)  # X on qubit 0
        assert_syndrome_decoding_correct(stabilizers, error, correction)
    """
    n = len(stabilizers[0])
    stab_vecs = np.array(
        [_pauli_str_to_symplectic(s) for s in stabilizers], dtype=np.int8
    )

    # Convert string inputs to symplectic if needed
    if isinstance(error, str):
        error_vec = _pauli_str_to_symplectic(error)
    else:
        error_vec = np.asarray(error, dtype=np.int8)

    if isinstance(decoder_correction, str):
        corr_vec = _pauli_str_to_symplectic(decoder_correction)
    else:
        corr_vec = np.asarray(decoder_correction, dtype=np.int8)

    # Residual = error * correction (in symplectic: XOR / mod-2 addition)
    residual = _symplectic_multiply(error_vec, corr_vec)

    # Check if residual is in the stabilizer group
    k = len(stabilizers)
    stab_group = set()
    for r in range(k + 1):
        for combo in itertools.combinations(range(k), r):
            prod: np.ndarray[Any, np.dtype[np.int8]] = np.zeros(2 * n, dtype=np.int8)
            for idx in combo:
                prod = _symplectic_multiply(prod, stab_vecs[idx])
            stab_group.add(tuple(prod))

    if tuple(residual) not in stab_group:
        # Check if it's a non-trivial logical operator
        in_normalizer = all(_symplectic_product(residual, sv) == 0 for sv in stab_vecs)
        if in_normalizer:
            raise AssertionError(
                "Decoder correction is incorrect: residual error is a "
                "non-trivial logical operator.\n"
                f"Residual (symplectic): {residual}"
            )
        else:
            raise AssertionError(
                "Decoder correction is incorrect: residual error "
                "anti-commutes with stabilizers.\n"
                f"Residual (symplectic): {residual}"
            )
