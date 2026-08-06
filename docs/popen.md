# popen — Open probability and dose–response curves

`scalcs.popen`

Equilibrium open probability (P_open), dose–response curve shape
parameters (EC50, Hill slope), and related utilities.

---

## Open probability

### `Popen(mec, tres, conc=0, eff='c')`

Calculate equilibrium open probability at a given effector concentration.

```python
po = Popen(mec, tres=1e-4, conc=1e-6)   # Popen at 1 µM, tres = 0.1 ms
```

| Argument | Type | Description |
|----------|------|-------------|
| `mec` | Mechanism | Channel mechanism |
| `tres` | float | Dead time / time resolution (s) |
| `conc` | float | Effector concentration |
| `eff` | str | Effector name (default `'c'` for agonist concentration) |

**Algorithm:**

- `tres = 0`: uses equilibrium occupancy `π(Q)` directly — `P_open = Σπᵢ (open states) / Σπᵢ (all states)`.
- `tres > 0`: uses the HJC-corrected mean open and shut times —
  `P_open = hmopen / (hmopen + hmshut)`.
- If `mec.fastblock = True`, applies fast-pore-blocker correction:
  `P_open_corrected = P_open / (1 + conc / mec.fastKB)`.

### `Popen0(mec, tres, eff='c')`

`P_open` at zero effector concentration.  Returns the HJC-corrected value
if `P_open(0) > 10⁻¹⁰` (spontaneous activity); otherwise returns the ideal
(tres=0) value.

---

## Maximum open probability and EC50

### `maxPopen(mec, tres, eff='c')`

Estimate the maximum equilibrium open probability numerically.  Scans
concentration from 1 nM upward; if the curve has a peak (e.g. due to open-
channel block), returns the peak value and the concentration at which it
occurs.

```python
Pmax, c_at_max = maxPopen(mec, tres)
```

### `EC50(mec, tres, eff='c')`

Concentration at which `P_open = 50 %` of `maxPopen`.  For non-monotone
curves, returns the EC50 on the ascending limb (to the left of the peak).

```python
ec50 = EC50(mec, tres)   # in M; multiply by 1e6 for µM
```

### `decline(mec, tres, eff='c')`

Returns `True` if the `P_open` curve decreases with increasing concentration
(e.g. the effector is an inhibitor).

---

## Hill slope

### `nH(mec, tres, eff='c')`

Calculate the Hill slope at EC50 by numerical differentiation of the
log-log dose–response curve around EC50.

```python
hill = nH(mec, tres)
```

---

## Printout helpers

### `printout(mec, tres)`

Full dose–response report: HJC and ideal maxPopen, EC50, nH, and
(if applicable) fast-block correction factor.

### `print_pars(mec, tres)`

One-line string: `maxPopen = ... ; EC50 = ... µM ; nH = ...`.

---

## Usage example

```python
from scalcs import popen
from scalcs.samples.CHME97 import CHME97

mec = CHME97()
tres = 1e-4   # 0.1 ms

Pmax, cmax = popen.maxPopen(mec, tres)
ec50 = popen.EC50(mec, tres)
nh   = popen.nH(mec, tres)

print(f'maxPopen = {Pmax:.4f}')
print(f'EC50 = {ec50*1e6:.2f} µM')
print(f'Hill slope = {nh:.3f}')
```

---

## Reference

- CH82: Colquhoun & Hawkes (1982) — P_open definition
- HJC92: Hawkes, Jalali & Colquhoun (1992) — HJC-corrected mean times
