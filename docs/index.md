# pytest-quantum

> **The only pytest plugin built specifically for quantum software.**
> 80+ assertions. 10 frameworks. Zero boilerplate. Ship quantum code you trust.

[![PyPI](https://img.shields.io/pypi/v/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![Python](https://img.shields.io/pypi/pyversions/pytest-quantum)](https://pypi.org/project/pytest-quantum/)
[![CI](https://github.com/qbench/pytest-quantum/actions/workflows/ci.yml/badge.svg)](https://github.com/qbench/pytest-quantum/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Who is this for?

- **Quantum software engineers** writing gate-model circuits in Qiskit, Cirq, PennyLane, Pytket, or Braket who want rigorous, reproducible unit tests.
- **Researchers running VQE or QAOA** who need statistically sound assertions on variational algorithm convergence, expressibility, and barren plateaus.
- **Teams CI-testing quantum algorithms** across multiple frameworks and backends, including noisy simulation and real hardware targets.
- **Hardware validation engineers** characterising real devices with quantum volume, randomised benchmarking, T1/T2/T2* coherence, and XEB fidelity.

---

## The problem with quantum testing

```python
# ❌ The old way: brittle, flaky, wrong
def test_bell_state():
    counts = run_bell_circuit(shots=1024)
    assert counts["00"] == 512   # fails 50% of the time due to shot noise
    assert counts["11"] == 512   # still wrong — global phase?
```

```python
# ✅ The pytest-quantum way
from pytest_quantum import assert_measurement_distribution, assert_unitary

def test_bell_distribution(aer_simulator):
    counts = run_bell_circuit(aer_simulator, shots=2000)
    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
    # Chi-square tested. Won't flake. ✓

def test_hadamard():
    import numpy as np
    assert_unitary(h_circuit, np.array([[1,1],[1,-1]])/np.sqrt(2))
    # Global-phase safe. Works on Qiskit, Cirq, Braket, Pytket. ✓
```

One line install. Fixtures auto-injected. Tests just work.

---

## Install in 10 seconds

```bash
pip install pytest-quantum                   # core
pip install "pytest-quantum[qiskit]"         # + Qiskit + Aer  ← most common
pip install "pytest-quantum[all]"            # everything
```

Then just write tests. No conftest.py. No imports. Fixtures appear by magic.

---

## What's in the box

| Category | Count | Examples |
|---|---|---|
| Unitary & equivalence | 6 | `assert_unitary`, `assert_circuits_equivalent`, `assert_cross_platform_equivalent` |
| States & fidelity | 3 | `assert_state_fidelity_above`, `assert_states_close`, `assert_normalized` |
| Measurement distributions | 2 | `assert_measurement_distribution`, `assert_counts_close` |
| Density matrices | 4 | `assert_density_matrix_close`, `assert_purity_above`, `assert_trace_distance_below` |
| Quantum channels | 6 | `assert_channel_is_cptp`, `assert_process_fidelity_above`, `assert_hermitian` |
| Noise models | 6 | `assert_depolarizing_channel`, `assert_amplitude_damping_channel`, `assert_channel_diamond_norm_below` |
| Entanglement | 3 | `assert_entanglement_entropy_below`, `assert_bloch_sphere_close`, `assert_schmidt_rank_at_most` |
| Observables & VQE | 4 | `assert_expectation_value_close`, `assert_vqe_converges`, `assert_ground_state_energy_close` |
| Circuit structure | 7 | `assert_circuit_depth`, `assert_gate_count`, `assert_circuit_is_clifford`, `assert_no_mid_circuit_measurement` |
| Benchmarking | 7 | `assert_quantum_volume`, `assert_randomized_benchmarking`, `assert_t1_above`, `assert_t2_above`, `assert_interleaved_rb` |
| Quantum ML | 4 | `assert_xeb_fidelity_above`, `assert_expressibility_above`, `assert_no_barren_plateau` |
| Error mitigation | 7 | `assert_zne_reduces_error`, `assert_pec_expectation_close`, `assert_cdr_reduces_error` |
| Hardware | 5 | `assert_backend_calibration`, `assert_circuit_fits_backend`, `assert_mirror_fidelity` |
| QEC / Stim | 3 | `assert_stim_logical_error_rate_below`, `assert_stabilizer_state` |
| Sweeps | 3 | `assert_circuit_sweep`, `assert_parametrized_unitary_continuous` |
| QASM round-trips | 2 | `assert_qasm_roundtrip`, `assert_qasm2_roundtrip` |
| Snapshots | 2 | `assert_unitary_snapshot`, `assert_distribution_snapshot` |
| Information theory | 3 | `assert_hellinger_close`, `assert_kl_divergence_below` |
| **Total** | **80+** | |

---

## Framework support

Works everywhere quantum code runs:

| Framework | Install | Fixtures |
|---|---|---|
| **Qiskit + Aer** | `pytest-quantum[qiskit]` | `aer_simulator`, `qiskit_sampler`, `qiskit_estimator` |
| **Cirq** | `pytest-quantum[cirq]` | `cirq_simulator`, `cirq_sampler` |
| **Amazon Braket** | `pytest-quantum[braket]` | `braket_simulator`, `braket_cloud_device` |
| **PennyLane** | `pytest-quantum[pennylane]` | `pennylane_device` |
| **Pytket** | `pip install pytket` | `pytket_circuit_factory` |
| **Stim** (QEC) | `pip install stim` | `stim_sampler` |
| **Mitiq** | `pytest-quantum[mitiq]` | works with any backend |
| **IBM Quantum** (real HW) | `pytest-quantum[ibm]` | `ibm_backend` |
| **IonQ** (real HW) | `pip install qiskit-ionq` | `ionq_backend` |
| **Quantinuum** (real HW) | `pip install pytket-quantinuum` | `quantinuum_backend` |

---

## Documentation

```{toctree}
:maxdepth: 2
:caption: Start here

getting-started
concepts
```

```{toctree}
:maxdepth: 2
:caption: Reference

assertions
fixtures
stats
api
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/01_vqe_end_to_end
tutorials/02_noise_aware_testing
tutorials/03_real_hardware
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
contributing
```

---

## Quick links

- 🚀 [Getting Started](getting-started.md): Zero to passing tests in 5 minutes
- 📖 [Assertions Reference](assertions.md): All 80+ assertions with examples
- 🔧 [Fixtures Reference](fixtures.md): Every fixture documented
- 🧪 [Cookbook](cookbook.md): Copy-paste recipes for common patterns
- 💡 [Concepts](concepts.md): Why shot noise matters, global phase, and more
- 🏆 [Tutorials](tutorials/01_vqe_end_to_end.md): End-to-end: VQE, noise, real hardware

---

*Built with ❤️ for the quantum software community.*
*MIT License · [GitHub](https://github.com/qbench/pytest-quantum) · [PyPI](https://pypi.org/project/pytest-quantum/)*
