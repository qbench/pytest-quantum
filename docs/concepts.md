# Concepts

Understanding *why* pytest-quantum makes certain design choices will help you
write better quantum tests. This page explains the physics and statistics
behind the library.

---

## Why shot noise is a first-class problem

Every quantum measurement is **fundamentally random**. Even a perfect Bell circuit
can produce `{"00": 487, "11": 537}` when the true probabilities are 50/50.
This isn't a bug; it's quantum mechanics.

Classical test frameworks don't know this. `assert counts["00"] == 512` fails
~5% of the time on a perfectly correct circuit simply because of Poisson sampling
fluctuations. This creates ghost failures in CI that waste hours of debugging time.

pytest-quantum's measurement assertions use **chi-square goodness-of-fit testing**:

```
H₀: the measured distribution matches the expected probabilities
Reject H₀ only if p-value < significance threshold (default: 0.01)
```

With 1024 shots and `significance=0.01`, you'll get a false positive ~1% of the
time on a correct circuit, not 5-50% like with hard-coded counts.

> **Rule of thumb:** If your quantum test flakes in CI, it probably needs more
> shots or a statistical assertion. Use `min_shots(epsilon=0.05)` to compute
> the minimum shots needed to detect a 5% deviation with 99% confidence.

---

## Why global phase is not a bug

Two quantum gates are physically identical if they differ only by a global phase:

```
U₂ = e^(iθ) · U₁   for any real θ
```

The circuit `H` and `e^(iπ/4) · H` are completely indistinguishable by any
quantum measurement. Yet `np.allclose(U1, U2)` returns `False` for them.

`assert_unitary` computes the *physical distance*:

```
distance = 1 - |Tr(U₁† U₂)| / d
```

This is zero if and only if U₁ and U₂ are equal up to global phase. You can
opt into strict phase comparison with `allow_global_phase=False` when you
specifically need to verify a global phase (rare, but sometimes relevant for
concatenating circuits).

---

## How fidelity is measured

The **state fidelity** between two pure states is:

```
F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²
```

This ranges from 0 (orthogonal) to 1 (identical up to global phase). A fidelity
of 0.99 means the states are 99% similar. In practice, fidelities above 0.999 are
considered "high fidelity" for near-term quantum devices.

For mixed states (density matrices), the **Uhlmann fidelity** generalises this:

```
F(ρ, σ) = (Tr √(√ρ σ √ρ))²
```

The **trace distance** is a complementary metric:

```
T(ρ, σ) = ½ ‖ρ - σ‖₁
```

Both satisfy 0 ≤ F ≤ 1 and 0 ≤ T ≤ 1, with T = 0 and F = 1 meaning identical states.
They're related by: `1 - √F ≤ T ≤ √(1 - F)`.

---

## Statistical distance between distributions

When comparing measurement distributions, you have several options depending
on what you care about:

### Total Variation Distance (TVD)

```
TVD(P, Q) = ½ Σₓ |P(x) - Q(x)|
```

TVD is the *operational* distance: it's the maximum probability difference
any event can have between the two distributions. TVD = 0 means identical,
TVD = 1 means completely different.

Use `assert_counts_close(counts_a, counts_b, max_tvd=0.05)`.

### Hellinger Distance

```
H(P, Q) = (1/√2) √(Σₓ (√P(x) - √Q(x))²)
```

More sensitive to small probabilities than TVD. Good for detecting rare
outcome deviations. Use `assert_hellinger_close`.

### KL Divergence

```
KL(P ‖ Q) = Σₓ P(x) log(P(x)/Q(x))
```

Asymmetric. Measures how much information is lost when using Q instead of P.
Useful for distribution comparison in variational algorithms.

---

## Quantum Volume

IBM's Quantum Volume (QV) protocol is the gold standard for characterising
the overall capability of a quantum processor.

A backend has QV = 2^n if it can execute "model circuits" of width n × depth n
using random SU(4) two-qubit unitaries, and achieve a **heavy output probability
(HOP) > 2/3** with statistical confidence:

```
HOP = fraction of measured bitstrings that are "heavy"
"heavy" = ideal probability above the median
```

The intuition: a perfect backend reproduces the ideal distribution exactly, so
HOP = expected fraction of outcomes above median ≈ 2/3 + ε. Noise makes the
distribution uniform, pushing HOP toward 0.5.

---

## Randomised Benchmarking

RB measures the average Clifford gate error rate by fitting an exponential
decay to survival probability vs sequence length:

```
P(m) = A · p^m + B
```

where `p` is the depolarising parameter and the **average gate fidelity** is:

```
F = 1 - (1-p)/d
```

for dimension `d = 2` (single qubit). The key insight: RB is robust to
**state preparation and measurement (SPAM) errors** because A and B absorb them.

**Interleaved RB (IRB)** isolates the error of a specific gate G by interleaving
G between random Cliffords. The gate fidelity is:

```
F_G = 1 - (d-1)/d · (1 - p_G/p_ref)
```

---

## Cross-Entropy Benchmarking (XEB)

XEB is Google's protocol for benchmarking (and demonstrating quantum advantage).
For random circuits, the XEB fidelity is:

```
F_XEB = D · ⟨P_ideal(x)⟩_measured - 1
```

where `D = 2^n` and the average is over measured bitstrings. For an ideal noiseless
backend, this equals 1.0 in expectation. Noise drives it toward 0.0.

XEB is more sensitive than QV for large circuits and is used in Google's
"quantum supremacy" and "quantum advantage" demonstrations.

---

## T1, T2, and T2* Coherence Times

These are the fundamental noise parameters for any quantum device:

| Parameter | Physical meaning | Measurement | Typical value |
|---|---|---|---|
| T1 | Energy relaxation: excited state decays to ground state | `assert_t1_above` | 50–200 µs |
| T2 | Dephasing with echo: Hahn echo sequence | `assert_t2_above` | 30–100 µs |
| T2* | Dephasing without echo: free induction decay (Ramsey) | `assert_t2star_above` | 10–50 µs |

All three use exponential decay fits: `P(t) = exp(-t/T)`.
Note T2 ≤ 2·T1 always (the Hahn echo refocuses static noise, T2 > T2* typically).

---

## Expressibility and Barren Plateaus

Two critical properties for variational quantum algorithms (VQAs):

### Expressibility

How uniformly does a parameterised ansatz sample the unitary group?
Measured via KL divergence between the ansatz's fidelity distribution
and the Haar-random (uniform) distribution:

```
expressibility ≈ exp(-KL(P_ansatz ‖ P_Haar))
```

Score = 1.0 means Haar-random (maximally expressive). Use `assert_expressibility_above`.

### Barren Plateaus

In deep VQAs, gradients can vanish exponentially with system size, a phenomenon known as the
**barren plateau** problem. pytest-quantum detects this via the **parameter-shift rule**:

```
∂E/∂θ = [E(θ+π/2) - E(θ-π/2)] / 2
```

If the variance of gradients across random initialisations is < threshold,
the landscape is too flat to train. Use `assert_no_barren_plateau`.

---

## Diamond Norm

The diamond norm measures the worst-case operational distance between two quantum
channels:

```
‖Φ - Ψ‖◇ = max_{ρ} ‖(Φ⊗I)(ρ) - (Ψ⊗I)(ρ)‖₁
```

It answers: "what's the maximum probability that any experiment can distinguish
channel Φ from channel Ψ?" A diamond norm of 0.01 means channels are at most
1% distinguishable.

Computing the diamond norm exactly requires a semidefinite programme (SDP).
pytest-quantum uses `cvxpy` when available, falling back to the spectral norm
as an upper bound.

---

## Why fixtures instead of setup code

**Without pytest-quantum:**

```python
# In every test file... every project... forever:
from qiskit_aer import AerSimulator

@pytest.fixture(scope="session")
def simulator():
    return AerSimulator()
```

**With pytest-quantum:**

```python
# Nothing. Just:
def test_bell(aer_simulator):
    ...
```

Benefits:
- **Session-scoped by default:** one simulator per pytest session, not one per test
- **Auto-skip:** tests are automatically skipped if the SDK isn't installed
- **Framework-agnostic:** same patterns work for Qiskit, Cirq, Braket, PennyLane
- **Composable:** fixtures can be combined with `multi_backend_runner` for cross-framework testing
