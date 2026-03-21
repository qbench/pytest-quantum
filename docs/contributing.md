# Contributing

Full setup and contribution guide is in [CONTRIBUTING.md](https://github.com/qbench/pytest-quantum/blob/master/CONTRIBUTING.md) on GitHub.

## Quick reference

```bash
# Setup
git clone https://github.com/qbench/pytest-quantum
cd pytest-quantum
uv sync --all-extras --group dev

# Tests
uv run pytest                          # full suite
uv run pytest --quantum-slow           # include slow tests
uv run pytest --cov=pytest_quantum     # with coverage

# Lint + types
uv run ruff check src/ tests/
uv run mypy src/

# Docs
uv sync --group docs
uv run sphinx-build docs docs/_build/html
```
