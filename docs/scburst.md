# scburst — Burst and cluster analysis

`scalcs.scburst`

Distributions of burst length, number of openings per burst, open and shut
times within and between bursts, following the Q-matrix formalism of
Colquhoun & Hawkes (1982).

---

## Background

A **burst** is a sequence of openings separated only by short shut periods.
Shut periods longer than a critical time `tcrit` separate bursts.  In the
CH82 scheme:

- States **A** (open) and **B** (blocked) are within the burst.
- States **C** (closed within cluster) can terminate or extend the burst.
- States **D** (between-cluster gaps) terminate the burst.

The burst structure is determined by the matrices `Q_EE` (A+B subspace) and
the jump matrices `G_AB`, `G_BA`.

---

## Burst start and end vectors

### `phiBurst(mec)`

Start probability vector φ_B (Eq. 3.2, CH82):

```
φ_B = (p_C · (Q_CB · G_BA + Q_CA)) / (p_C · (Q_CB · G_BA + Q_CA) · u_A)
```

Shape `(1, kA)`.

### `endBurst(mec)`

End vector e_B (Eq. 3.4, CH82):

```
e_B = (I − G_AB · G_BA) · u_A
```

Shape `(kA, 1)`.  Probability that a burst ends after the current opening
rather than returning to a within-burst shut state.

---

## Burst length distribution

### `length_pdf(mec, t)`

Pdf of the burst length at time `t` (Eq. 3.17, CH82):

```
f(t) = φ_B · [exp(Q_EE · t)]_AA · (−Q_AA) · e_B
```

### `length_pdf_components(mec)`

Time constants and areas of the burst-length pdf (spectral expansion).

### `length_mean(mec)`

Mean burst length.

### `length_cond_pdf(mec, t)`

Burst-length pdf conditional on the burst containing more than one opening.

### `length_no_single_openings_pdf_components(mec)`

Components of the burst-length pdf excluding single-opening bursts.

---

## Number of openings per burst

### `openings_distr(mec, r)`

Probability that a burst contains exactly `r` openings (geometric
distribution in the simplest case).

### `openings_distr_components(mec)`

Components of the geometric mixture distribution of openings per burst.

### `openings_mean(mec)`

Mean number of openings per burst.

### `openings_cond_distr_depend_on_start_state(mec, r)`

Distribution of openings per burst conditioned on the starting state.

---

## Open and shut times within bursts

### `open_time_total_pdf_components(mec)`

Components of the total open-time pdf (summed over all openings in all
bursts).

### `open_time_mean(mec)`

Mean open time within a burst.

### `shut_times_inside_burst_pdf_components(mec)`

Components of the pdf of short shut times (gaps within a burst).

### `first_opening_length_pdf_components(mec)`

Components of the pdf of the first opening length within a burst.

---

## Between-burst shut times

### `shut_times_between_burst_pdf_components(mec)`

Components of the pdf of long shut times (gaps between bursts).

### `shut_times_between_burst_mean(mec)`

Mean between-burst shut time.

### `shut_time_total_mean(mec)`

Mean total shut time (within + between bursts combined).

### `shut_time_total_pdf_components_2more_openings(mec)`

Components of the total shut-time pdf restricted to bursts with ≥ 2
openings.

---

## Printout helper

### `printout_pdfs(mec)`

Human-readable table of all burst-length and opening-number statistics.

---

## Usage example

```python
from scalcs import scburst
from scalcs.samples.CH82 import CH82

mec = CH82()
mec.set_eff('c', 1e-6)

print('Mean burst length (ms):', scburst.length_mean(mec) * 1000)
print('Mean openings per burst:', scburst.openings_mean(mec))
print('Mean between-burst gap (ms):', scburst.shut_times_between_burst_mean(mec) * 1000)
```

---

## Reference

- CH82: Colquhoun & Hawkes (1982) — all burst equations
