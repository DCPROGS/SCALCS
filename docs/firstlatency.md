# firstlatency — First-latency pdf after a concentration jump

`scalcs.firstlatency`

Computes the probability density of the time from a rapid agonist
concentration jump to the first channel opening, at three levels of
approximation: ideal, asymptotic (HJC), and exact (HJC).

---

## Physical scenario

The channel is held at zero agonist concentration `c₀ = 0`.  At `t = 0` the
concentration steps to `c₁ > 0`.  Before the jump the entire population
occupies shut states with equilibrium occupancies:

```
φ_shut = π(Q at c₀)[kA:]
```

The first-latency pdf `f_L(t)` is the probability density of the time to the
first opening.

---

## Ideal pdf (no missed events)

```
f_L(t) = φ_shut · exp(Q_FF · t) · (−Q_FF) · u_F
```

A mixture of `kF` exponentials with rates equal to the eigenvalues of
`−Q_FF`.

### `ideal_components(QFF, phi_shut)`

Returns eigenvalues and component areas of the ideal first-latency pdf.

```python
eigs, areas = ideal_components(mec.QFF, phi_shut)
# f(t) = sum_i areas[i] * eigs[i] * exp(-eigs[i]*t)
```

| Return | Shape | Description |
|--------|-------|-------------|
| `eigs` | (kF,) | Eigenvalues of −Q_FF (positive reals) |
| `areas` | (kF,) | Component areas; `sum(areas) = 1` |

### `ideal_pdf(t, QFF, phi_shut)`

Evaluate the ideal first-latency pdf at time(s) `t`.  Accepts scalar or
array input; returns matching type.

---

## Asymptotic pdf (HJC approximation)

Valid for `t >> tres`.  The `kF` asymptotic roots solve `det[W_F(s)] = 0`
with the Q-submatrix roles exchanged (A ↔ F) relative to the open-time
convention.

### `asymptotic_roots(tres, mec)`

Find the `kF` roots of the shut-side W matrix.

```python
roots = asymptotic_roots(tres, mec)   # shape (kF,), all < 0
```

Internally calls `scalcslib.asymptotic_roots` with transposed submatrices.
See [scalcslib.md](scalcslib.md#asymptotic_roots) for notes on numerical
stability at `tres > 0.7 ms` and at `tres = 0`.

### `asymptotic_areas(tres, roots, phi_shut, mec)`

Component areas using `φ_shut` as the initial vector:

```
area[i] = (−1/roots[i]) · φ_shut · R[i] · Q_FA · exp(Q_AA · tres) · u_A
```

```python
areas = asymptotic_areas(tres, roots, phi_shut, mec)
tau   = -1.0 / roots
```

### `asymptotic_pdf(t, tres, tau, areas)`

Evaluate the asymptotic first-latency pdf:

```
f(t) = 0                              for t ≤ tres
f(t) = expPDF(t − tres, tau, areas)   for t > tres
```

---

## Gamma coefficients for exact pdf

### `gamma_coefficients(tres, phi_shut, mec)`

Compute the γ coefficient arrays needed for the exact pdf, using the
shut-side (F) spectral decomposition.

```python
eigvals, g00, g10, g11 = gamma_coefficients(tres, phi_shut, mec)
```

| Return | Shape | Description |
|--------|-------|-------------|
| `eigvals` | (k,) | Eigenvalues of −Q (full matrix) |
| `g00` | (k,) | `φ_shut · Z00[i] · u_A` |
| `g10` | (k,) | `φ_shut · Z10[i] · u_A` |
| `g11` | (k,) | `φ_shut · Z11[i] · u_A` |

---

## Exact pdf (HJC full correction)

Applies the HJC exact correction for `tres ≤ t < 3·tres` and reverts to
the asymptotic form beyond.

### `exact_pdf(t, tres, roots, areas, eigvals, g00, g10, g11)`

```
f(t) = 0                                      for t < tres
f(t) = f0(t − tres)                           for tres ≤ t < 2·tres
f(t) = f0(t − tres) − f1(t − 2·tres)         for 2·tres ≤ t < 3·tres
f(t) = expPDF(t − tres, −1/roots, areas)      for t ≥ 3·tres
```

Accepts scalar or array `t`; returns matching type.

The asymptotic branch (`t ≥ 3·tres`) is fully vectorised — a single
`expPDF` call for all qualifying time points.

---

## Complete usage example

```python
import numpy as np
from scalcs import firstlatency as fl
from scalcs import qmatlib as qml
from scalcs.samples.CHME97 import CHME97

# Build mechanism at c1 = 1 µM
mec = CHME97()
mec.set_eff('c', 1e-6)

# Initial shut-state occupancies at c0 = 0
mec0 = CHME97()
mec0.set_eff('c', 0.0)
phi_shut = qml.pinf(mec0.Q)[mec0.kA:]

tres = 1e-3   # 1 ms dead time

# --- Asymptotic ---
roots = fl.asymptotic_roots(tres, mec)
areas = fl.asymptotic_areas(tres, roots, phi_shut, mec)
tau   = -1.0 / roots

t = np.linspace(tres, 0.1, 500)
f_asym = fl.asymptotic_pdf(t, tres, tau, areas)

# --- Exact ---
eigvals, g00, g10, g11 = fl.gamma_coefficients(tres, phi_shut, mec)
f_exact = fl.exact_pdf(t, tres, roots, areas, eigvals, g00, g10, g11)
```

---

## Numerical stability notes

### Float64 overflow at large `tres`

The matrix exponential `exp((Q_FF − s·I)·tres)` is evaluated during the
root-count sweep at `s = sas` (the lower search boundary).  When
`|sas| × tres > ln(float64_max) ≈ 709` this overflows.  For
`tres = 1 ms` overflow occurs at `|sas| > 7 × 10⁵`.

The adaptive bound `sas = max(−10⁶, −700/tres)` prevents overflow while
keeping the bracket wide enough to locate all roots.

### Infinite loop at `tres = 0`

At `tres = 0`, `H(s) = Q_AA` (constant), making the root-counting function
a step function.  The bisection algorithm discards zero-root sub-intervals
(rather than re-queuing them) to avoid an infinite loop.

---

## References

- CH82: Colquhoun & Hawkes (1982)
- HJC92: Hawkes, Jalali & Colquhoun (1992)
- CHME97: Colquhoun, Hawkes, Merlushkin & Edmonds (1997)
