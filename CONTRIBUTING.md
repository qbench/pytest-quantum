# Contributing to pytest-quantum

Thank you for your interest in contributing. This document covers everything you need to get set up, run tests, and submit a pull request.

---

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (fast Python package manager — replaces pip/venv)

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

```bash
git clone https://github.com/qbench/pytest-quantum
cd pytest-quantum

# Create virtual environment and install all dependencies (all quantum SDKs + dev tools)
uv sync --all-extras --group dev
```

That's it. `uv sync` creates `.venv/` and installs everything.

---

## Running tests

```bash
# Full test suite (111+ tests across all frameworks)
uv run pytest

# Only unit tests (no quantum SDK needed — fast, < 1s)
uv run pytest tests/test_stats.py tests/test_assertions_distributions.py \
               tests/test_assertions_states.py tests/test_assertions_structure.py

# Integration tests only (real quantum circuits)
uv run pytest tests/test_integration_qiskit.py tests/test_integration_cirq.py \
               tests/test_integration_cross_framework.py \
               tests/test_integration_pennylane.py tests/test_integration_noise.py

# Include slow shot-heavy tests
uv run pytest --quantum-slow

# With coverage report
uv run pytest --cov=pytest_quantum --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_stats.py -v

# Run a specific test by name
uv run pytest -k "test_bell_state_distribution" -v
```

---

## Linting and type checking

All three must pass before a PR is merged.

```bash
# Linter + formatter (ruff does both)
uv run ruff check src/ tests/          # check
uv run ruff check src/ tests/ --fix    # auto-fix what's fixable
uv run ruff format src/ tests/         # format

# Type checker (strict mypy — no Any leakage in public API)
uv run mypy src/

# Run all three at once
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/
```

---

## Building docs

```bash
# Install docs dependencies
uv sync --group docs

# Build HTML docs
uv run sphinx-build docs docs/_build/html

# Open in browser (macOS)
open docs/_build/html/index.html

# Auto-rebuild on change (requires sphinx-autobuild)
uv run sphinx-autobuild docs docs/_build/html
```

---

## Project structure

```
src/pytest_quantum/
├── __init__.py              # public API — everything users import
├── plugin.py                # pytest entry point: markers, CLI options, all fixtures
├── assertions/
│   ├── distributions.py     # assert_measurement_distribution, assert_counts_close
│   ├── states.py            # assert_state_fidelity_above, assert_states_close
│   ├── structure.py         # assert_circuit_depth, assert_circuit_width, assert_gate_count
│   └── unitary.py           # assert_unitary, assert_circuits_equivalent
├── converters/
│   └── to_unitary.py        # any circuit → numpy unitary matrix
└── stats/
    ├── shots.py             # min_shots, recommended_shots
    └── tests.py             # fidelity, tvd, tvd_from_counts, chi_square_test

tests/
├── test_stats.py                          # pure math tests, no SDKs needed
├── test_assertions_*.py                   # unit tests (mock-based)
├── test_plugin.py                         # pytester-based plugin tests
├── test_integration_qiskit.py             # real Qiskit + AerSimulator
├── test_integration_cirq.py               # real Cirq
├── test_integration_cross_framework.py    # Qiskit vs Cirq equivalence
├── test_integration_pennylane.py          # real PennyLane
└── test_integration_noise.py             # depolarizing noise fixture
```

---

## Adding support for a new framework

1. Add the import detection to `converters/to_unitary.py` — add a `_is_<framework>` function and `_from_<framework>` converter.
2. Add a fixture to `plugin.py` following the same `scope="session"` pattern.
3. Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`.
4. Add integration tests to `tests/test_integration_<framework>.py`.
5. Update `README.md` and `docs/fixtures.md`.

---

## Adding a new assertion

1. Add the function to the appropriate file in `src/pytest_quantum/assertions/`.
2. Register assert rewriting in `__init__.py` (already done for all modules).
3. Export from `__init__.py` (`from pytest_quantum.assertions.X import ...` and add to `__all__`).
4. Add unit tests in the corresponding `tests/test_assertions_*.py`.
5. Add integration tests that use real circuits.
6. Update `docs/assertions.md`.

---

## Code style

- Type annotations required everywhere (`mypy --strict`).
- Google-style docstrings on every public function.
- `from __future__ import annotations` at the top of every file.
- Optional SDK imports must be inside functions (lazy imports) — never at module level.
- All assertions must produce clear, human-readable error messages with observed vs expected values.

---

## Pull request checklist

- [ ] `uv run pytest` passes (136+ tests)
- [ ] `uv run ruff check src/ tests/` passes
- [ ] `uv run mypy src/` passes
- [ ] New assertion or fixture has integration test with a real circuit
- [ ] Docstring updated with example
- [ ] `docs/` updated if public API changed
- [ ] `CHANGELOG.md` entry added under an `Unreleased` section

---

## Reporting bugs

Open an issue at https://github.com/qbench/pytest-quantum/issues with:

- Python version (`python --version`)
- pytest-quantum version (`pip show pytest-quantum`)
- Quantum SDK versions (e.g. `pip show qiskit qiskit-aer`)
- Minimal reproducible example
- Full error output
