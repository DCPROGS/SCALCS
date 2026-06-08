# pdfs — Exponential mixture pdf utilities

`scalcs.pdfs`

Evaluation and statistics for exponential mixture probability density
functions, plus utilities for choosing the critical time (tcrit) that divides
fast and slow components.

---

## Exponential mixture pdf

All dwell-time distributions in SCALCS are ultimately sums of exponentials:

```
f(t) = Σᵢ (areaᵢ / τᵢ) · exp(−t / τᵢ)
```

where `τᵢ = 1 / rateᵢ` are the time constants and `Σ areaᵢ = 1`.

### `expPDF(t, tau, area)`

Evaluate the pdf at time(s) `t`.

```python
f = expPDF(t, tau, area)
```

| Argument | Type | Description |
|----------|------|-------------|
| `t` | float or ndarray | Time (s) |
| `tau` | ndarray, shape (k,) | Time constants (s) |
| `area` | ndarray, shape (k,) | Component areas (sum = 1) |

Returns a scalar or ndarray matching the shape of `t`.

### `expPDF_mean_sd(tau, area)`

Mean and standard deviation of the distribution.

```python
mean, sd = expPDF_mean_sd(tau, area)
```

```
mean = Σ areaᵢ · τᵢ
var  = 2 · Σ areaᵢ · τᵢ²  −  mean²
```

### `expPDF_printout(eigs, ampl)`

Return a formatted string table of rates, time constants, areas, mean, and
CV.  `eigs` are rates (1/s); `ampl` are weights (before normalisation by
rate).

---

## Critical time (tcrit) for burst definition

The critical time divides the shut-time distribution into short gaps
(within-burst) and long gaps (between-burst).  Three criteria are available:

### `expPDF_misclassified(tcrit, tau, area, comp)`

Number and fraction of misclassified events when using `tcrit` to split
component `comp` from the rest.

```python
enf, ens, pf, ps = expPDF_misclassified(tcrit, tau, area, comp)
```

| Return | Description |
|--------|-------------|
| `enf` | Expected number of fast events misclassified as slow (per 100) |
| `ens` | Expected number of slow events misclassified as fast (per 100) |
| `pf` | Fraction of fast component misclassified |
| `ps` | Fraction of slow component misclassified |

The three criterion functions return a scalar that crosses zero at the
optimal `tcrit` — pass to `scipy.optimize.brentq`:

| Function | Criterion |
|----------|-----------|
| `expPDF_tcrit_DC(tcrit, tau, area, comp)` | Equal fraction misclassified: `pf = ps` (Colquhoun–Hawkes) |
| `expPDF_tcrit_CN(tcrit, tau, area, comp)` | Equal number misclassified: `enf = ens` (Clapham–Neher) |
| `expPDF_tcrit_Jackson(tcrit, tau, area, comp)` | Equal pdf value: `f_fast(tcrit) = f_slow(tcrit)` |

### `expPDF_misclassified_printout(tcrit, enf, ens, pf, ps)`

Return a formatted string summary of misclassification statistics.

---

## Geometric mixture pdf

For distributions over integer counts (e.g. number of openings per burst):

### `geometricPDF_mean_sd(rho, w)`

Mean and standard deviation of a geometric mixture:

```
mean = Σ wᵢ / (1 − ρᵢ)²
```

### `geometricPDF_printout(rho, w)`

Formatted table of geometric pdf parameters and statistics.

---

## Usage example

```python
import numpy as np
from scalcs import pdfs
from scalcs import scalcslib as scl

# Get ideal open-time components
eigs, areas = scl.ideal_dwell_time_pdf_components(mec.QAA, phiA)
tau = 1.0 / eigs

# Evaluate pdf
t = np.linspace(1e-5, 0.1, 500)
f = pdfs.expPDF(t, tau, areas)

mean, sd = pdfs.expPDF_mean_sd(tau, areas)
print(f'Mean open time: {mean*1000:.3f} ms  CV: {sd/mean:.3f}')
```
