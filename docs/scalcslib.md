# scalcslib — Dwell-time pdfs and single-channel statistics

`scalcs.scalcslib`

Core calculations for dwell-time probability density functions (ideal,
asymptotic, and exact), mean open/shut times, correlations between
successive dwell times, and single-channel simulation.

---

## Ideal (no missed events) pdfs

These functions treat every event as perfectly resolved — no dead time.

### `ideal_dwell_time_pdf(t, QAA, phiA)`

Scalar pdf value at time `t`:

```
f(t) = φ_A · exp(Q_AA · t) · (−Q_AA) · u_A
```

To get the shut-time pdf, pass `QFF` as `QAA` and the shut initial vector
as `phiA`.

### `ideal_dwell_time_pdf_components(QAA, phiA)`

Eigenvalues (rates) and component areas of the ideal pdf.

```python
eigs, areas = ideal_dwell_time_pdf_components(mec.QAA, phiA)
# tau = 1/eigs; f(t) = sum areas[i] * eigs[i] * exp(-eigs[i]*t)
```

### `ideal_subset_time_pdf(Q, k1, k2, t)`

Ideal pdf restricted to a subset of states (rows `k1` to `k2` of Q).

### `ideal_subset_mean_life_time(Q, state1, state2)`

Mean life time in states `state1` to `state2`.

### `ideal_mean_latency_given_start_state(mec, state)`

Mean first-latency (time to first opening) given a specific starting state,
under the ideal (no missed events) approximation.

---

## Asymptotic pdf (HJC approximation)

Valid for `t >> tres`.  The missed-events correction is incorporated via the
HJC matrix W(s) whose roots define the asymptotic time constants.

### `asymptotic_roots(tres, QAA, QFF, QAF, QFA, kA, kF)`

Find the `kA` roots of `det[W(s)] = 0`.

```python
roots = asymptotic_roots(tres, mec.QAA, mec.QFF, mec.QAF, mec.QFA, mec.kA, mec.kF)
```

Returns `ndarray` of shape `(kA,)` — all roots are negative reals.

**Numerical stability** — The lower search bound is clamped adaptively:

```
sas = max(−1_000_000, −700 / tres)
```

This prevents `exp((QFF − s·I)·tres)` from overflowing float64 when
`tres > 0.7 ms`.  See [`bisect_intervals`](#bisect_intervals) for the
root-counting algorithm.

### `bisect_gFB(s, tres, Q11, Q22, Q12, Q21, k1, k2)`

Count how many eigenvalues of H(s) are ≤ s.  This is Frank Ball's
root-counting function used to locate brackets.

### `bisect_intervals(sa, sb, tres, Q11, Q22, Q12, Q21, k1, k2)`

Partition `[sa, sb]` into sub-intervals each containing exactly one root,
using Frank Ball's bisection method.

**Infinite-loop guard** — Sub-intervals with zero roots are discarded
silently.  Without this fix, `tres = 0` causes an infinite loop because
H(s) is constant (= Q_AA) and the root-count function is a step function,
making every left sub-interval vacuous.

### `bisect_split(sa, sb, nga, ngb, tres, Q11, Q22, Q12, Q21, k1, k2)`

Split one interval at its midpoint and count roots in each half.  Called
by `bisect_intervals`.

### `asymptotic_areas(tres, roots, QAA, QFF, QAF, QFA, kA, kF, GAF, GFA)`

Component areas for the asymptotic open-time pdf.  Uses the HJC equilibrium
vector φ_A (not φ_shut) as the initial vector.

### `asymptotic_pdf(t, tres, tau, area)`

Evaluate the asymptotic pdf at times `t`.  Returns zero for `t < tres`,
`expPDF(t − tres, tau, area)` otherwise.

---

## Exact pdf (HJC full correction)

Applies the exact missed-events correction for `tres ≤ t < 3·tres` and
reverts to the asymptotic form beyond.

### `exact_pdf(t, tres, roots, areas, eigvals, gamma00, gamma10, gamma11)`

```
f(t) = 0                                      for t < tres
f(t) = f0(t − tres)                           for tres ≤ t < 2·tres
f(t) = f0(t − tres) − f1(t − 2·tres)         for 2·tres ≤ t < 3·tres
f(t) = expPDF(t − tres, −1/roots, areas)      for t ≥ 3·tres
```

The γ coefficients (`gamma00`, `gamma10`, `gamma11`) and `eigvals` are
obtained from `exact_GAMAxx`.

### `exact_GAMAxx(mec, tres, open)`

Compute eigenvalues and γ coefficient arrays for the exact pdf.
Set `open=True` for the open-time pdf, `open=False` for the shut-time pdf.

### `exact_mean_open_shut_time(mec, tres)`

HJC-corrected mean open and mean shut time.

```python
hmopen, hmshut = exact_mean_open_shut_time(mec, tres)
```

### `exact_mean_time(tres, QAA, QFF, QAF, kA, kF, GAF, GFA)`

Mean dwell time under the HJC approximation.

---

## Correlations between successive dwell times

### `corr_variance_A(phiA, QAA, kA)`

Variance of the open-time distribution.

### `corr_covariance_A(lag, phiA, QAA, XAA, kA)`

Autocovariance of open times at lag `lag`.

### `corr_covariance_AF(lag, phiA, QAA, QFF, XAA, GAF, kA, kF)`

Cross-covariance between open times and the following shut time at lag `lag`.

### `corr_decay_amplitude_A(phiA, QAA, XAA, kA)`

Amplitudes of the exponential components of the open-time autocorrelation
function.

### `corr_limit_A(phiA, QAA, AXAA, eigXAA, kA)`

Limiting correlation coefficient at large lag.

---

## Adjacent dwell-time distributions

### `adjacent_open_to_shut_range_mean(u1, u2, QAA, QAF, QFF, QFA, phiA)`

Mean open time adjacent to a shut time in the range `[u1, u2]`.

### `adjacent_open_to_shut_range_pdf_components(u1, u2, QAA, QAF, QFF, QFA, phiA)`

Components of the open-time pdf conditioned on the adjacent shut time lying
in `[u1, u2]`.

### `HJC_dependency(top, tsh, tres, Q, QAA, QAF, QFF, QFA)`

HJC dependency ratio: tests for independence between open time `top` and
shut time `tsh`.

### `HJC_adjacent_mean_open_to_shut_time_pdf(sht, tres, Q, QAA, QAF, QFF, QFA)`

Pdf of the mean open time adjacent to a given shut time `sht` under the
HJC approximation.

---

## Likelihood

### `likelihood(theta, opts)` / `HJClik(theta, opts)`

Negative log-likelihood functions for maximum-likelihood fitting of
mechanisms to single-channel data.  `HJClik` applies the full HJC
missed-events correction.

---

## Simulation

### `simulate_intervals(mec, tres, state, opamp=5, nintmax=5000)`

Simulate a sequence of open and shut intervals by drawing dwell times from
the rate matrix.

### `next_state(present, picum, tmean, kA, opamp)`

Sample the next state given the current state and the cumulative transition
probability vector.

---

## Printout helpers

| Function | Description |
|----------|-------------|
| `printout_occupancies(mec, tres)` | Equilibrium and HJC occupancies |
| `printout_distributions(mec, tres)` | Summary of all dwell-time distributions |
| `printout_tcrit(mec)` | Critical time (tcrit) for burst definition |
| `printout_correlations(mec)` | Correlation coefficients at multiple lags |
| `printout_adjacent(mec, t1, t2)` | Adjacent open/shut time statistics |

---

## References

- CH82: Colquhoun & Hawkes (1982)
- HJC92: Hawkes, Jalali & Colquhoun (1992)
- CHS96: Colquhoun, Hawkes & Srodzinski (1996)
