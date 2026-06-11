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

## Concentration-pulse first latency

The functions above treat a *step* to `c₁` that stays on forever.  A finite
**pulse** holds `c₁` for a duration `T` and then returns to zero (`c = 0 → c₁
for 0 < t < T → 0`).  Recording starts at the beginning of the pulse, so
openings during the pulse are included.  These functions implement CHME97 §3
(ideal) and §4 (apparent, with missed events) for that scenario.

Two mechanisms are passed: `mec1` at the pulse concentration `c₁` and `mec0`
at zero concentration.  The shut set is ordered `[B, C]` where **B** are the
within-burst shut states (can still open at `c = 0`) and **C** is absorbing at
`c = 0` (cannot open without binding).

### Ideal pulse (no missed events — CHME97 §3)

The pdf splits at the end of the pulse `T`:

```
t < T :  f(t) = φ · exp(Q1_FF·t) · Q1_FA · u_A / P(R≥1)              (Eq 3.6)
t ≥ T :  f(t) = [φ·exp(Q1_FF·T)]_B · exp(Q0_BB·(t−T)) · Q0_BA · u_A
                / P(R≥1)                                              (Eq 3.8)
```

During the pulse the rates are the `kF` eigenvalues of `−Q1_FF`; after the
pulse only the within-burst states B can still open, giving `kB` exponentials
with rates `−Q0_BB`.  The pdf is conditional on at least one opening and so is
divided by `P(R≥1)`.

| Function | Returns |
|----------|---------|
| `pulse_PR_ge_one(T, phi_shut, mec1, mec0)` | `P(R≥1)`, the probability of at least one opening (Eq 3.10) |
| `ideal_pulse_components(T, phi_shut, mec1, mec0, PR=None)` | `(eigs_during, areas_during, eigs_after, areas_after)` |
| `ideal_pulse_pdf(t, T, phi_shut, mec1, mec0, PR=None)` | conditional pdf at time(s) `t` (scalar or array) |

As `T → ∞` the pulse reduces to the simple step (`P(R≥1) → 1`, `ideal_pulse_pdf
→ ideal_pdf`).  The pdf is discontinuous at `t = T` (the opening rate drops
when agonist is removed).

### Shut-time survivor matrix `ᶠR(t)`

The missed-event calculations rest on the HJC shut-time survivor matrix
(CHME97 Appendix A):

```
ᶠR(t)_ij = P[X(t)=j and no detectable opening over (0,t) | X(0)=i],  i,j ∈ F
```

exact for `t < 2·tres` (built from `qmatlib.Cxx` via `f0`/`f1`) and asymptotic
beyond.  `ᶠR(0) = I`; for `t < tres` it equals `[exp(Q·t)]_FF`.

| Function | Returns |
|----------|---------|
| `shut_survivor_components(tres, mec)` | dict of cached pieces (`eigvals`, `C00/C10/C11`, `roots`, `R`) |
| `shut_survivor(t, tres, components)` | the `(kF, kF)` survivor matrix `ᶠR(t)` |

### Apparent pulse (missed events — CHME97 §4)

With a finite dead time `tres` an opening is detectable only if the channel
stays open for at least `tres`.  Following the convention of the step module,
the latency `t` is measured to the instant of **detection** (the opening
transition plus `tres`), so the density is zero for `t ≤ tres`.  Three regimes
(CHME97 Eqs 4.2–4.4):

```
tres < t < T        within the pulse  (≡ exact_pdf, the step apparent pdf)
T ≤ t < T + tres     confirming sojourn straddles the end of the pulse
t ≥ T + tres         opening transition after the pulse (double integral)
```

| Function | Returns |
|----------|---------|
| `apparent_pulse_pdf(t, T, tres, phi_shut, mec1, mec0, components=None, nquad=24)` | **unconditional** density (scalar or array) |
| `apparent_pulse_PR_ge_one(T, tres, phi_shut, mec1, mec0, nquad=24, upper=None)` | `P(R≥1)`, by quadrature of the density |

`apparent_pulse_pdf` returns the *unconditional* density (so the within-pulse
regime equals `exact_pdf` exactly); divide by `apparent_pulse_PR_ge_one` for the
conditional pdf.  The after-pulse regime evaluates the Eq-4.4 double integral by
the trapezium rule (`nquad` points per axis) and uses a reduced zero-concentration
`{A, B}` survivor internally — see the module source for the derivation.  The
pulse must be longer than the dead time (`T > tres`, as assumed in CHME97).

```python
T = 50e-3          # 50 ms pulse
PR  = fl.apparent_pulse_PR_ge_one(T, tres, phi_shut, mec1, mec0)
afl = fl.apparent_pulse_pdf(t, T, tres, phi_shut, mec1, mec0) / PR   # conditional
```

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
