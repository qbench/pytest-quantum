# Statistics Utilities Reference

pytest-quantum ships statistical primitives that underpin its assertions and
help you choose the right shot count for reliable tests. All functions are pure
numpy/scipy — no quantum SDK is required.

Import from the top-level package:

```python
from pytest_quantum import (
    min_shots,
    recommended_shots,
    fidelity,
    tvd,
    tvd_from_counts,
    chi_square_test,
)
```

---

## Shot-count calculators

### `min_shots`

```python
min_shots(
    epsilon,
    alpha=0.05,
    power=0.80,
) -> int
```

Returns the minimum number of shots to reliably detect a Total Variation
Distance of `epsilon` between two distributions.

**The formula**

Based on two-sample statistical power analysis:

$$
N = \left\lceil \frac{(z_{1-\alpha/2} + z_{\text{power}})^2}{2\varepsilon^2} \right\rceil
$$

where $z_p$ is the $p$-th quantile of the standard normal distribution.

With default settings ($\alpha = 0.05$, $\text{power} = 0.80$):
$z_{0.975} \approx 1.96$, $z_{0.80} \approx 0.84$.

**Parameters**

: `epsilon` — Minimum detectable TVD. `0.01` means the test can reliably
  catch a 1% deviation from the expected distribution.
: `alpha` — Significance level (default `0.05` → 95% confidence).
: `power` — Statistical power — the probability of detecting a real error
  (default `0.80` → 80% power).

**Returns**

: Minimum recommended shot count as an integer.

**Raises**

: `ValueError` — Any argument is outside its valid range `(0, 1)`.

**Worked examples**

```python
from pytest_quantum import min_shots

min_shots(0.10)                        # 74   — catch 10% TVD, 95% CI, 80% power
min_shots(0.05)                        # 293  — catch 5% TVD
min_shots(0.01)                        # 7299 — catch 1% TVD
min_shots(0.01, alpha=0.01, power=0.90)  # 11282 — stricter: 99% CI, 90% power
```

**Using in a test**

```python
import pytest
from pytest_quantum import assert_measurement_distribution, min_shots

@pytest.mark.quantum
def test_bell_5pct_sensitivity(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    shots = min_shots(epsilon=0.05)   # 293 shots

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=shots
    ).result().get_counts()

    assert_measurement_distribution(counts, {"00": 0.5, "11": 0.5})
```

**Choosing epsilon**

| Use case | Recommended epsilon |
|---|---|
| Smoke test — just check the circuit runs | `0.10` (74 shots) |
| Normal regression test | `0.05` (293 shots) |
| Precise distribution validation | `0.02` (1 825 shots) |
| High-precision scientific result | `0.01` (7 299 shots) |

Mark high-shot tests with `@pytest.mark.quantum_slow` and run them with
`--quantum-slow` to keep the default suite fast.

---

### `recommended_shots`

```python
recommended_shots(
    expected_probs,
    min_expected_per_bucket=5,
) -> int
```

Returns the shot count needed so every non-zero bucket in `expected_probs`
gets at least `min_expected_per_bucket` expected counts.

The chi-square goodness-of-fit test (used by `assert_measurement_distribution`)
requires expected count ≥ 5 per cell to produce valid p-values. Violating
this gives unreliable results and triggers a `UserWarning`.

`recommended_shots` targets the **rarest outcome**: if the rarest outcome has
probability $p_{\min}$, you need at least $\lceil k / p_{\min} \rceil$ shots
(where $k$ is `min_expected_per_bucket`).

**Parameters**

: `expected_probs` — Dict mapping outcome labels to probabilities. Must sum
  to 1. Zero-probability outcomes are ignored.
: `min_expected_per_bucket` — Minimum expected count per non-zero bucket
  (default `5`).

**Returns**

: Recommended shot count as an integer.

**Raises**

: `ValueError` — `expected_probs` is empty or all probabilities are zero.

**Examples**

```python
from pytest_quantum import recommended_shots

# Uniform Bell state — rarest outcome has probability 0.5
recommended_shots({"00": 0.5, "11": 0.5})           # 10

# Mostly-uniform, but one rare outcome at 0.1%
recommended_shots({"00": 0.499, "01": 0.001, "11": 0.5})  # 5000

# 3-qubit uniform (min_prob = 1/8)
recommended_shots({f"{i:03b}": 1/8 for i in range(8)})    # 40
```

**Using in a test**

```python
from pytest_quantum import assert_measurement_distribution, recommended_shots

def test_ghz_distribution(aer_simulator):
    from qiskit import QuantumCircuit, transpile

    expected = {"000": 0.5, "111": 0.5}
    shots = recommended_shots(expected)   # 10 — very cheap for uniform

    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.cx(1, 2); qc.measure_all()
    counts = aer_simulator.run(
        transpile(qc, aer_simulator), shots=max(shots, 500)
    ).result().get_counts()

    assert_measurement_distribution(counts, expected)
```

:::{note}
`recommended_shots` guarantees chi-square validity but may return fewer
shots than `min_shots` for detection power. For production tests combine
both: use `max(recommended_shots(probs), min_shots(epsilon=0.05))` to
satisfy both constraints.
:::

---

## Statistical primitives

### `fidelity`

```python
fidelity(
    psi,
    phi,
) -> float
```

Computes the pure-state fidelity $F = |\langle\psi|\phi\rangle|^2$.

Both arrays are flattened and normalised before computation, so minor
normalisation errors from simulators do not affect the result.

**Returns:** Float in `[0.0, 1.0]`. `1.0` means identical states (up to
global phase). `0.0` means orthogonal states.

**Raises:** `ValueError` if arrays have different sizes or are zero-norm.

```python
import numpy as np
from pytest_quantum import fidelity

zero = np.array([1, 0], dtype=complex)
one  = np.array([0, 1], dtype=complex)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

fidelity(zero, zero)    # 1.0 — identical
fidelity(zero, one)     # 0.0 — orthogonal
fidelity(zero, plus)    # 0.5 — |<0|+>|² = 0.5
fidelity(plus, plus)    # 1.0 — identical
```

**Global phase invariance:**

```python
psi   = np.array([1, 0], dtype=complex)
psi_j = 1j * np.array([1, 0], dtype=complex)   # global phase i·|0>

fidelity(psi, psi_j)    # 1.0 — global phase is invisible
```

---

### `tvd`

```python
tvd(
    p,
    q,
) -> float
```

Computes the Total Variation Distance between two probability distributions:

$$
\text{TVD}(p, q) = \frac{1}{2} \sum_x |p(x) - q(x)|
$$

**Parameters:** `p`, `q` — 1-D numpy arrays of probabilities (each sums to 1).

**Returns:** Float in `[0.0, 1.0]`. `0.0` means identical; `1.0` means
disjoint support.

```python
import numpy as np
from pytest_quantum import tvd

# Identical distributions
tvd(np.array([0.5, 0.5]), np.array([0.5, 0.5]))    # 0.0

# Small deviation
tvd(np.array([0.5, 0.5]), np.array([0.6, 0.4]))    # 0.1

# Orthogonal distributions
tvd(np.array([1.0, 0.0]), np.array([0.0, 1.0]))    # 1.0
```

**Interpreting TVD values:**

| TVD | Interpretation |
|---|---|
| `0.0` | Identical distributions |
| `< 0.05` | Very close — acceptable for most tests |
| `0.05 – 0.15` | Noticeable deviation — may indicate noise or error |
| `> 0.15` | Significant — likely a bug or misconfiguration |
| `1.0` | Completely disjoint — certain error |

---

### `tvd_from_counts`

```python
tvd_from_counts(
    counts_a,
    counts_b,
) -> float
```

Computes TVD between two shot-count dictionaries. Each dict is normalised to
a probability distribution before TVD is calculated. Outcomes present in one
dict but absent in the other are treated as having count 0.

**Parameters:**

: `counts_a` — First counts dict, e.g. `{"00": 489, "11": 511}`.
: `counts_b` — Second counts dict, e.g. `{"00": 501, "11": 499}`.

**Returns:** Float in `[0.0, 1.0]`.

**Raises:** `ValueError` if either dict is empty.

```python
from pytest_quantum import tvd_from_counts

# Nearly identical Bell distributions
tvd_from_counts(
    {"00": 489, "11": 511},
    {"00": 501, "11": 499},
)
# → 0.012

# One backend sees "01" where the other sees nothing
tvd_from_counts(
    {"00": 500, "11": 500},
    {"00": 450, "01": 50, "11": 500},
)
# → 0.05
```

**Using `tvd_from_counts` directly (instead of `assert_counts_close`):**

```python
from pytest_quantum import tvd_from_counts

def test_backend_drift(aer_simulator):
    """Fail if backend results drift more than 3% TVD day-over-day."""
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.measure_all()
    qc_t = transpile(qc, aer_simulator)

    run1 = aer_simulator.run(qc_t, shots=2000).result().get_counts()
    run2 = aer_simulator.run(qc_t, shots=2000).result().get_counts()

    distance = tvd_from_counts(run1, run2)
    assert distance < 0.03, f"Backend drift too large: TVD = {distance:.4f}"
```

---

### `chi_square_test`

```python
chi_square_test(
    observed,
    expected_probs,
    total_shots=None,
) -> tuple[float, float]
```

Chi-square goodness-of-fit test for quantum measurement distributions.
Tests whether `observed` counts are consistent with `expected_probs`.

This is the statistical engine behind `assert_measurement_distribution`.
Use it directly when you need the raw p-value or chi-square statistic.

**Parameters**

: `observed` — Either a count dict `{"00": 489, "11": 511}` or a 1-D numpy
  array of observed counts.
: `expected_probs` — Either a probability dict `{"00": 0.5, "11": 0.5}`
  (must sum to 1) or a 1-D numpy array of expected probabilities.
: `total_shots` — Required when both inputs are numpy arrays. Ignored when
  dict inputs are used (total is inferred from `observed`).

**Returns:** `(statistic, pvalue)` — the chi-square statistic and the p-value.

Reject the null hypothesis (i.e., declare the distributions inconsistent)
when `pvalue < significance`.

**Raises:** `ValueError` — Inconsistent inputs (mismatched keys, missing
`total_shots` for array inputs, observed counts summing to zero).

**Example — dict inputs**

```python
from pytest_quantum import chi_square_test

# 1000 shots on a Bell circuit — should give 50/50
stat, p = chi_square_test(
    observed={"00": 495, "11": 505},
    expected_probs={"00": 0.5, "11": 0.5},
)
print(f"χ² = {stat:.4f},  p = {p:.4f}")
# χ² = 0.1000,  p = 0.7518   → consistent

# Biased circuit — clearly wrong distribution
stat, p = chi_square_test(
    observed={"00": 800, "11": 200},
    expected_probs={"00": 0.5, "11": 0.5},
)
print(f"χ² = {stat:.4f},  p = {p:.6f}")
# χ² = 360.0000,  p = 0.000000   → reject null hypothesis
```

**Example — numpy array inputs**

```python
import numpy as np
from pytest_quantum import chi_square_test

observed_counts  = np.array([245, 255, 248, 252])    # 4-outcome uniform
expected_uniform = np.array([0.25, 0.25, 0.25, 0.25])

stat, p = chi_square_test(
    observed=observed_counts,
    expected_probs=expected_uniform,
    total_shots=1000,
)
assert p > 0.05   # consistent with uniform distribution
```

**Interpreting p-values**

| p-value | Interpretation |
|---|---|
| `> 0.05` | Consistent with expected distribution — pass |
| `0.01 – 0.05` | Marginal — consider more shots |
| `< 0.01` | Significant deviation — likely a bug |
| `< 0.001` | Strong evidence of error |

**Degrees of freedom**

The chi-square test has `k - 1` degrees of freedom, where `k` is the number
of non-zero expected outcome buckets. Adding outcomes with zero expected
probability that appear in counts does not add degrees of freedom.

The test requires expected count ≥ 5 per cell. Use `recommended_shots` to
compute the shot count that satisfies this for your distribution.
