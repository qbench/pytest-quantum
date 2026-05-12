"""Qibo framework adapter."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytest_quantum.adapters.protocol import FrameworkAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CLIFFORD_QIBO = frozenset({
    "H", "X", "Y", "Z", "S", "SDG", "CNOT", "CZ", "SWAP",
    "SX", "SXDG",
})


class QiboAdapter(FrameworkAdapter):
    """Adapter for Qibo circuits."""

    framework_name = "qibo"
    big_endian = False

    @classmethod
    def detect(cls, circuit: object) -> bool:
        return type(circuit).__module__.startswith("qibo")

    def to_unitary(self, circuit: object) -> NDArray[np.complex128]:
        return np.array(circuit.unitary(), dtype=np.complex128)  # type: ignore[attr-defined]

    def get_depth(self, circuit: object) -> int:
        return int(circuit.depth)  # type: ignore[attr-defined]

    def get_width(self, circuit: object) -> int:
        return int(circuit.nqubits)  # type: ignore[attr-defined]

    def count_gates(self, circuit: object) -> dict[str, int]:
        counts: dict[str, int] = {}
        for gate in circuit.queue:  # type: ignore[attr-defined]
            name = gate.__class__.__name__
            counts[name] = counts.get(name, 0) + 1
        return counts

    def gate_names(self, circuit: object) -> set[str]:
        return {gate.__class__.__name__ for gate in circuit.queue}  # type: ignore[attr-defined]

    def is_clifford(self, circuit: object) -> bool:
        return all(
            gate.__class__.__name__ in _CLIFFORD_QIBO
            for gate in circuit.queue  # type: ignore[attr-defined]
        )

    def has_mid_circuit_measurement(self, circuit: object) -> bool:
        found_measurement = False
        for gate in circuit.queue:  # type: ignore[attr-defined]
            name = gate.__class__.__name__
            if name == "M":
                found_measurement = True
            elif found_measurement:
                return True
        return False

    def get_diagram(self, circuit: object) -> str:
        return str(circuit.draw())  # type: ignore[attr-defined]
