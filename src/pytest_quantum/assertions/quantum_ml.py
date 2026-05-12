"""Quantum machine learning assertions for pytest-quantum.

Provides assertions for cross-entropy benchmarking (XEB), circuit expressibility,
entanglement capability (Meyer-Wallach measure), and barren plateau detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Public assertions
# ---------------------------------------------------------------------------


def assert_xeb_fidelity_above(
    backend: Any,
    num_qubits: int,
    target_fidelity: float,
    *,
    num_circuits: int = 20,
    depth: int = 10,
    shots: int = 1024,
) -> float:
    """Assert cross-entropy benchmarking (XEB) fidelity is at least *target_fidelity*.

    XEB measures how well a noisy backend reproduces the ideal output distribution
    of random circuits.  For each random circuit, the XEB fidelity is:

        F_XEB = D * <P_ideal(x)>_measured - 1

    where ``D = 2**num_qubits`` and ``<P_ideal(x)>`` is the average ideal probability
    of observed bitstrings.  Perfect fidelity = 1.0; random noise = 0.0.

    Args:
        backend:         Qiskit-compatible backend (AerSimulator, IBM, etc.).
        num_qubits:      Number of qubits to benchmark.
        target_fidelity: Minimum acceptable XEB fidelity (0.0–1.0).
        num_circuits:    Number of random circuits to average over (default 20).
        depth:           Circuit depth in random layers (default 10).
        shots:           Shots per circuit (default 1024).

    Returns:
        float: Average XEB fidelity across all circuits.

    Raises:
        AssertionError: If XEB fidelity < target_fidelity.
        ImportError:    If qiskit is not installed.

    Example::

        from pytest_quantum.assertions.quantum_ml import assert_xeb_fidelity_above
        from qiskit_aer import AerSimulator


        def test_xeb(aer_simulator):
            fidelity = assert_xeb_fidelity_above(
                aer_simulator, num_qubits=2, target_fidelity=0.9
            )
            assert fidelity >= 0.9
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile as qk_transpile
        from qiskit.quantum_info import Statevector, random_unitary
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_xeb_fidelity_above. "
            "Install with: pip install qiskit"
        ) from exc

    d = 2**num_qubits
    xeb_fidelities: list[float] = []

    is_ibm = _is_ibm_backend(backend)

    for _ in range(num_circuits):
        # Build a random circuit with Haar-random single-qubit gates + CZ entanglers
        qc = QuantumCircuit(num_qubits, num_qubits)
        for layer in range(depth):
            # Random single-qubit unitaries on all qubits
            for q in range(num_qubits):
                u = random_unitary(2)
                qc.unitary(u, [q])
            # Entangling layer (alternating CZ pattern)
            if num_qubits > 1:
                start = layer % 2
                for q in range(start, num_qubits - 1, 2):
                    qc.cz(q, q + 1)
        qc.measure(list(range(num_qubits)), list(range(num_qubits)))

        # Get ideal probabilities via statevector simulation
        qc_no_meas = QuantumCircuit(num_qubits)
        for instr in qc.data:
            if instr.operation.name != "measure":
                qc_no_meas.append(instr)
        sv = Statevector(qc_no_meas)
        ideal_probs: np.ndarray[Any, Any] = sv.probabilities()

        # Run on backend
        transpiled = qk_transpile(qc, backend, optimization_level=0)
        if is_ibm:
            try:
                from qiskit_ibm_runtime import SamplerV2

                sampler = SamplerV2(backend)
                job = sampler.run([transpiled], shots=shots)
                pub_result = job.result()[0]
                counts = _extract_sampler_counts(pub_result)
            except ImportError:
                job = backend.run(transpiled, shots=shots)
                result = job.result()
                counts = result.get_counts()
        else:
            job = backend.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts()

        total = sum(counts.values())
        if total == 0:
            continue

        # Compute XEB fidelity for this circuit
        avg_ideal_prob = 0.0
        for bitstr, cnt in counts.items():
            idx = int(bitstr.replace(" ", ""), 2)
            if idx < d:
                avg_ideal_prob += (cnt / total) * float(ideal_probs[idx])

        f_xeb = d * avg_ideal_prob - 1.0
        xeb_fidelities.append(f_xeb)

    if not xeb_fidelities:
        raise AssertionError("No XEB results obtained — all circuits failed to run.")

    mean_xeb = float(np.mean(xeb_fidelities))

    if mean_xeb < target_fidelity:
        raise AssertionError(
            f"XEB fidelity below target.\n"
            f"  Measured XEB fidelity : {mean_xeb:.4f}\n"
            f"  Required fidelity     : {target_fidelity:.4f}\n"
            f"  Num qubits            : {num_qubits}\n"
            f"  Circuits / depth      : {num_circuits} / {depth}\n"
            f"  Hint: run on a less noisy backend or lower target_fidelity."
        )

    return mean_xeb


def assert_expressibility_above(
    ansatz_fn: Callable[..., Any],
    num_qubits: int,
    num_params: int,
    target_expressibility: float,
    *,
    num_samples: int = 200,
    num_bins: int = 75,
) -> float:
    """Assert circuit expressibility (frame potential / KL divergence from Haar).

    Measures how uniformly an ansatz samples the unitary group compared to the
    Haar measure.  Expressibility is quantified as:

        expr = 1 - KL(P_ansatz || P_Haar)  (normalized to [0, 1])

    Higher = more expressive.  A Haar-random circuit has expressibility ≈ 1.0.

    Args:
        ansatz_fn:           Callable ``(params: np.ndarray) -> QuantumCircuit``.
                             Returns a parameterised Qiskit QuantumCircuit.
        num_qubits:          Number of qubits in the ansatz.
        num_params:          Number of parameters the ansatz accepts.
        target_expressibility: Minimum required expressibility score (0.0–1.0).
        num_samples:         Number of random parameter samples (default 200).
        num_bins:            Histogram bins for fidelity distribution (default 75).

    Returns:
        float: Expressibility score (higher = better, 1.0 = Haar-equivalent).

    Raises:
        AssertionError: If expressibility < target_expressibility.
        ImportError:    If qiskit is not installed.
    """
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_expressibility_above. "
            "Install with: pip install qiskit"
        ) from exc

    rng = np.random.default_rng(42)
    fidelities: list[float] = []

    for _ in range(num_samples):
        p1 = rng.uniform(0, 2 * np.pi, num_params)
        p2 = rng.uniform(0, 2 * np.pi, num_params)
        try:
            qc1 = ansatz_fn(p1)
            qc2 = ansatz_fn(p2)
            sv1 = Statevector(qc1).data
            sv2 = Statevector(qc2).data
            fid = float(abs(np.vdot(sv1, sv2)) ** 2)
            fidelities.append(fid)
        except Exception:
            continue

    if len(fidelities) < 10:
        raise AssertionError(
            "assert_expressibility_above: ansatz_fn failed for too many parameter sets. "
            "Check that it returns a valid QuantumCircuit."
        )

    # Compare fidelity distribution to Haar distribution for n qubits
    # Haar fidelity distribution: P(F) = (2^n - 1) * (1 - F)^(2^n - 2)
    fidelities_arr = np.asarray(fidelities)
    dim = 2**num_qubits
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    # Empirical distribution
    hist, _ = np.histogram(fidelities_arr, bins=bins, density=True)
    p_ansatz = hist * bin_width + 1e-10

    # Haar distribution
    p_haar = (dim - 1) * (1 - bin_centers) ** (dim - 2) * bin_width + 1e-10

    # Normalize
    p_ansatz = p_ansatz / p_ansatz.sum()
    p_haar = p_haar / p_haar.sum()

    # KL divergence: lower = more Haar-like = more expressive
    kl = float(stats.entropy(p_ansatz, p_haar))

    # Convert to [0,1] score: expressibility = exp(-kl)
    # (0 = maximally inexpressive, 1 = Haar-equivalent)
    expressibility = float(np.exp(-kl / max(1.0, np.log(num_bins))))
    expressibility = float(np.clip(expressibility, 0.0, 1.0))

    if expressibility < target_expressibility:
        raise AssertionError(
            f"Circuit expressibility below target.\n"
            f"  Expressibility : {expressibility:.4f}\n"
            f"  Required       : {target_expressibility:.4f}\n"
            f"  Num qubits     : {num_qubits}\n"
            f"  KL divergence  : {kl:.4f}  (0 = Haar-random)\n"
            f"  Hint: increase circuit depth or use more entangling layers."
        )

    return expressibility


def assert_entanglement_capability_above(
    ansatz_fn: Callable[..., Any],
    num_qubits: int,
    num_params: int,
    target_capability: float,
    *,
    num_samples: int = 200,
) -> float:
    """Assert Meyer-Wallach entanglement capability of a parameterised ansatz.

    Computes the average Meyer-Wallach entanglement measure Q over random
    parameter samples.  Q ∈ [0, 1]: 0 = no entanglement, 1 = maximally entangled.

    Args:
        ansatz_fn:          Callable ``(params: np.ndarray) -> QuantumCircuit``.
        num_qubits:         Number of qubits.
        num_params:         Number of parameters.
        target_capability:  Minimum average Q value (0.0–1.0).
        num_samples:        Random parameter samples (default 200).

    Returns:
        float: Average Meyer-Wallach Q value.

    Raises:
        AssertionError: If average Q < target_capability.
        ImportError:    If qiskit is not installed.
    """
    try:
        from qiskit.quantum_info import Statevector, partial_trace
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_entanglement_capability_above. "
            "Install with: pip install qiskit"
        ) from exc

    rng = np.random.default_rng(42)
    q_values: list[float] = []

    for _ in range(num_samples):
        params = rng.uniform(0, 2 * np.pi, num_params)
        try:
            qc = ansatz_fn(params)
            sv = Statevector(qc)
            # Meyer-Wallach: Q = (2/n) * sum_k linear_entropy(rho_k)
            q = 0.0
            for k in range(num_qubits):
                # Trace out all qubits except k
                rho_k = partial_trace(sv, [j for j in range(num_qubits) if j != k])
                rho_arr = np.asarray(rho_k.data, dtype=complex)
                # Linear entropy: S_L = 1 - Tr(rho^2)
                s_linear = 1.0 - float(np.real(np.trace(rho_arr @ rho_arr)))
                q += s_linear
            q = (2.0 / num_qubits) * q if num_qubits > 0 else 0.0
            q_values.append(float(np.clip(q, 0.0, 1.0)))
        except Exception:
            continue

    if len(q_values) < 10:
        raise AssertionError(
            "assert_entanglement_capability_above: ansatz_fn failed for too many samples."
        )

    mean_q = float(np.mean(q_values))

    if mean_q < target_capability:
        raise AssertionError(
            f"Entanglement capability (Meyer-Wallach Q) below target.\n"
            f"  Measured Q : {mean_q:.4f}\n"
            f"  Required Q : {target_capability:.4f}\n"
            f"  Num qubits : {num_qubits}\n"
            f"  Hint: add CNOT/CZ layers to increase entanglement."
        )

    return mean_q


def assert_no_barren_plateau(
    ansatz_fn: Callable[..., Any],
    num_qubits: int,
    num_params: int,
    observable: Any | None = None,
    *,
    num_samples: int = 100,
    min_gradient_variance: float = 1e-4,
) -> float:
    """Assert no barren plateau: gradient variance is above *min_gradient_variance*.

    Detects barren plateaus by computing the variance of parameter-shift gradients
    over random parameter initializations.  A barren plateau manifests as an
    exponentially vanishing gradient variance.

    Args:
        ansatz_fn:            Callable ``(params: np.ndarray) -> QuantumCircuit``.
        num_qubits:           Number of qubits.
        num_params:           Number of parameters.
        observable:           Optional Qiskit SparsePauliOp. Defaults to Z⊗Z⊗…⊗Z.
        num_samples:          Random parameter initializations (default 100).
        min_gradient_variance: Minimum acceptable variance (default 1e-4).

    Returns:
        float: Gradient variance (higher = no barren plateau).

    Raises:
        AssertionError: If gradient variance < min_gradient_variance.
        ImportError:    If qiskit is not installed.
    """
    try:
        from qiskit.quantum_info import SparsePauliOp, Statevector
    except ImportError as exc:
        raise ImportError(
            "qiskit is required for assert_no_barren_plateau. "
            "Install with: pip install qiskit"
        ) from exc

    if observable is None:
        # Default: Z⊗Z⊗...⊗Z on all qubits
        pauli_str = "Z" * num_qubits
        observable = SparsePauliOp(pauli_str)

    rng = np.random.default_rng(42)
    gradients: list[float] = []

    for _ in range(num_samples):
        params = rng.uniform(0, 2 * np.pi, num_params)

        # Pick a random parameter to differentiate
        param_idx = int(rng.integers(0, num_params))

        try:
            # Parameter shift rule: ∂E/∂θ = (E(θ+π/2) - E(θ-π/2)) / 2
            params_plus = params.copy()
            params_plus[param_idx] += np.pi / 2
            params_minus = params.copy()
            params_minus[param_idx] -= np.pi / 2

            qc_plus = ansatz_fn(params_plus)
            qc_minus = ansatz_fn(params_minus)

            sv_plus = Statevector(qc_plus)
            sv_minus = Statevector(qc_minus)

            # Expectation value of observable
            e_plus = float(np.real(sv_plus.expectation_value(observable)))
            e_minus = float(np.real(sv_minus.expectation_value(observable)))

            grad = (e_plus - e_minus) / 2.0
            gradients.append(grad)
        except Exception:
            continue

    if len(gradients) < 10:
        raise AssertionError(
            "assert_no_barren_plateau: ansatz_fn failed for too many parameter sets."
        )

    grad_variance = float(np.var(gradients))

    if grad_variance < min_gradient_variance:
        raise AssertionError(
            f"Barren plateau detected: gradient variance too small.\n"
            f"  Gradient variance   : {grad_variance:.2e}\n"
            f"  Required variance   : {min_gradient_variance:.2e}\n"
            f"  Num qubits          : {num_qubits}\n"
            f"  Samples             : {len(gradients)}\n"
            f"  Hint: use local cost functions, layer-wise training, or"
            f" reduce circuit depth to avoid barren plateaus."
        )

    return grad_variance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


from pytest_quantum._internal import _is_ibm_backend, _extract_sampler_counts
