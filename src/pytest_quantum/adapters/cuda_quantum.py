"""CUDA Quantum framework adapter."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CudaQuantumAdapter(FrameworkAdapter):
    """Adapter for CUDA Quantum (cudaq) kernels."""

    framework_name = "cuda_quantum"
    big_endian = False

    @classmethod
    def detect(cls, circuit: object) -> bool:
        return type(circuit).__module__.startswith("cudaq")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        """Reconstruct unitary by simulating each basis state.

        CUDA Quantum kernels don't expose a direct unitary method,
        so we simulate column-by-column.
        """
        import cudaq

        # Infer qubit count from the kernel
        n_qubits = getattr(circuit, "num_qubits", None)
        if n_qubits is None:
            # Try running with zero state to infer dimension
            state = cudaq.get_state(circuit)
            dim = len(state)
            n_qubits = int(np.log2(dim))

        dim = 2**n_qubits
        unitary = np.zeros((dim, dim), dtype=np.complex128)

        for col in range(dim):
            # Create basis state |col>
            init_state = np.zeros(dim, dtype=np.complex128)
            init_state[col] = 1.0
            state = cudaq.get_state(circuit, init_state)
            unitary[:, col] = np.array(state, dtype=np.complex128)

        return unitary

    def get_depth(self, circuit: object) -> int:
        raise NotImplementedError(
            "CudaQuantumAdapter does not support get_depth. "
            "CUDA Quantum kernels don't expose circuit depth directly."
        )

    def get_width(self, circuit: object) -> int:
        n_qubits = getattr(circuit, "num_qubits", None)
        if n_qubits is not None:
            return int(n_qubits)
        import cudaq
        state = cudaq.get_state(circuit)
        return int(np.log2(len(state)))

    def count_gates(self, circuit: object) -> dict[str, int]:
        raise NotImplementedError(
            "CudaQuantumAdapter does not support count_gates."
        )

    def gate_names(self, circuit: object) -> set[str]:
        raise NotImplementedError(
            "CudaQuantumAdapter does not support gate_names."
        )

    def is_clifford(self, circuit: object) -> bool:
        raise NotImplementedError(
            "CudaQuantumAdapter does not support is_clifford."
        )

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        raise NotImplementedError(
            "CudaQuantumAdapter does not support has_mid_circuit_measurement."
        )

    def get_diagram(self, circuit: object) -> str:
        raise NotImplementedError(
            "CudaQuantumAdapter does not support get_diagram."
        )
