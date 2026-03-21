# pytest-quantum

A cross-framework pytest plugin for quantum program testing.

Works with **Qiskit**, **Cirq**, **Amazon Braket**, **PennyLane**, and **Graphix**.

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

## Contents

```{toctree}
:maxdepth: 2
:caption: Guides

getting-started
assertions
fixtures
stats
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
changelog
contributing
```
