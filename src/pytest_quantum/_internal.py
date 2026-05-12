"""Shared internal utilities for pytest-quantum.

This module consolidates helper functions that were previously duplicated
across multiple assertion modules. It is NOT part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def _unitaries_equivalent(
    u_a: NDArray[np.complex128],
    u_b: NDArray[np.complex128],
    *,
    atol: float,
    allow_global_phase: bool,
) -> bool:
    """Return ``True`` if *u_a* and *u_b* are close (optionally up to global phase).

    Args:
        u_a: First unitary matrix.
        u_b: Second unitary matrix.
        atol: Absolute tolerance for element-wise comparison.
        allow_global_phase: If ``True``, matrices that differ only by a
            scalar phase ``e^{iθ}`` are considered equivalent.

    Returns:
        ``True`` if the matrices are equivalent within *atol*.

    Example::

        import numpy as np

        I = np.eye(2, dtype=np.complex128)
        assert _unitaries_equivalent(I, I, atol=1e-8, allow_global_phase=True)
    """
    if np.allclose(u_a, u_b, atol=atol):
        return True
    if not allow_global_phase:
        return False
    flat_idx = int(np.argmax(np.abs(u_a)))
    a_val = u_a.flat[flat_idx]
    b_val = u_b.flat[flat_idx]
    if abs(a_val) > 1e-10 and abs(b_val) > 1e-10:
        phase = a_val / b_val
        if np.allclose(u_a, phase * u_b, atol=atol):
            return True
    return False


def _kraus_to_choi(kraus_ops: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Convert Kraus operators to the Choi matrix via column vectorisation.

    The Choi matrix is defined as:

    .. math::

        Λ = \\sum_k |K_k⟩⟩⟨⟨K_k|

    where ``|K_k⟩⟩`` is the column vectorisation of Kraus operator ``K_k``.

    Args:
        kraus_ops: List of Kraus operator matrices, each of shape ``(d, d)``.

    Returns:
        Choi matrix of shape ``(d², d²)``.

    Example::

        import numpy as np

        choi = _kraus_to_choi([np.eye(2, dtype=np.complex128)])
        assert choi.shape == (4, 4)
    """
    K0 = kraus_ops[0]
    d = K0.shape[0]
    choi = np.zeros((d * d, d * d), dtype=np.complex128)
    for K in kraus_ops:
        k_vec = K.flatten(order="F").reshape(-1, 1)
        choi += k_vec @ k_vec.conj().T
    return choi


def _extract_sampler_counts(pub_result: Any) -> dict[str, int]:
    """Extract counts dict from a SamplerV2 ``PubResult``.

    Tries well-known classical register names (``meas``, ``c``, ``c0``,
    ``cr``, ``measure``) and then falls back to iterating dataclass fields.

    Args:
        pub_result: A ``qiskit.primitives.PubResult`` from ``SamplerV2``.

    Returns:
        Dictionary mapping bitstring to count.

    Raises:
        AssertionError: If counts cannot be extracted.
    """
    data = pub_result.data
    for name in ("meas", "c", "c0", "cr", "measure"):
        bit_array = getattr(data, name, None)
        if bit_array is not None and hasattr(bit_array, "get_counts"):
            counts: dict[str, int] = bit_array.get_counts()
            return counts
    for name in getattr(data, "__dataclass_fields__", {}):
        bit_array = getattr(data, name, None)
        if bit_array is not None and hasattr(bit_array, "get_counts"):
            counts = bit_array.get_counts()
            return counts
    raise AssertionError(
        "Could not extract counts from SamplerV2 result. "
        "Ensure the circuit has measurements."
    )


def _is_ibm_backend(backend: Any) -> bool:
    """Return ``True`` if *backend* is a ``qiskit_ibm_runtime.IBMBackend``.

    Args:
        backend: Any backend object.

    Returns:
        ``True`` if *backend* is an ``IBMBackend`` instance, ``False``
        otherwise (including when ``qiskit_ibm_runtime`` is not installed).
    """
    try:
        from qiskit_ibm_runtime import IBMBackend

        return isinstance(backend, IBMBackend)
    except ImportError:
        return False


def _backend_name(backend: object) -> str:
    """Return a human-readable name for *backend*.

    Args:
        backend: Any backend object.

    Returns:
        The backend's ``.name`` (or ``.name()``) if available, otherwise
        ``repr(backend)``.

    Example::

        class FakeBackend:
            name = "fake_backend"


        assert _backend_name(FakeBackend()) == "fake_backend"
    """
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    if isinstance(name, str):
        return name
    return repr(backend)


def _run_circuit(
    qc: Any,
    backend: Any,
    *,
    shots: int,
    is_ibm: bool,
    qk_transpile: Callable[..., Any],
) -> dict[str, int]:
    """Run *qc* on *backend* and return measurement counts.

    Handles both ``SamplerV2`` (IBM Runtime) and legacy ``backend.run()``
    execution paths.

    Args:
        qc: A Qiskit ``QuantumCircuit``.
        backend: A Qiskit backend instance.
        shots: Number of measurement shots.
        is_ibm: Whether the backend is an IBM Runtime backend.
        qk_transpile: The ``qiskit.transpile`` callable.

    Returns:
        Dictionary mapping bitstring to count.
    """
    transpiled = qk_transpile(qc, backend, optimization_level=0)
    if is_ibm:
        try:
            from qiskit_ibm_runtime import SamplerV2

            sampler = SamplerV2(backend)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()
            pub_result = result[0]
            return _extract_sampler_counts(pub_result)
        except ImportError:
            pass
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts: dict[str, int] = result.get_counts()
    return counts
