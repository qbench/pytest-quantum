# pytest-quantum

A cross-framework pytest plugin for quantum program testing.

Works with **Qiskit**, **Cirq**, **Amazon Braket**, **PennyLane**, **Graphix**, **Pytket**, and **Stim**.

---

## Install

```bash
pip install pytest-quantum             # core only
pip install "pytest-quantum[qiskit]"   # + Qiskit + Aer
pip install "pytest-quantum[cirq]"     # + Cirq
pip install "pytest-quantum[braket]"   # + Amazon Braket
pip install "pytest-quantum[pennylane]" # + PennyLane
pip install "pytest-quantum[graphix]"  # + Graphix
pip install "pytest-quantum[all]"      # everything
```

---

## Navigation

| Guide | Description |
|---|---|
| [Getting Started](getting-started.md) | Installation, first tests, markers, and CLI options |
| [Concepts](concepts.md) | Shot noise, global phase, qubit ordering, and session scope explained |
| [Assertions Reference](assertions.md) | All 38 assertions with examples and failure messages |
| [Fixtures Reference](fixtures.md) | All simulator fixtures, new in v0.2/v0.3, skip behavior |
| [Cookbook](cookbook.md) | Complete working recipes for 10 common patterns |
| [Statistics Utilities](stats.md) | `min_shots`, `recommended_shots`, `fidelity`, `tvd`, `chi_square_test` |
| [API Reference](api.md) | Auto-generated API docs |
| [Changelog](changelog.md) | What's new in each version |
| [Contributing](contributing.md) | How to contribute |

---

## Contents

```{toctree}
:maxdepth: 2
:caption: Guides

getting-started
concepts
assertions
fixtures
cookbook
stats
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
changelog
contributing
```
