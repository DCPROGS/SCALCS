# cjumps — Macroscopic responses to concentration jumps

`scalcs.cjumps`

Calculates the time course of open probability and state occupancies when the
agonist concentration changes as a function of time.  Covers both numerical
(ODE) and analytical (matrix-exponential) solutions, and provides relaxation
time constants for ideal square pulses.

---

## Concentration pulse profiles

Four dataclasses describe the shape of the concentration waveform.  All
accept keyword arguments and are hashable / comparable.

### `ErfPulse`

Realistic jump with error-function rise and fall — matches fast-perfusion
(`rcj`) experiments where the solution exchange is not instantaneous.

```python
ErfPulse(
    c0=0.0,        # baseline concentration (M)
    c1=1e-6,       # peak concentration (M)
    t_rise=0.0,    # time of pulse start (s)
    t_fall=0.1,    # time of pulse end (s)
    rise_time=1e-4,  # 10–90 % rise time (s)
    fall_time=1e-4,  # 10–90 % fall time (s)
)
```

### `SquarePulse`

Ideal instantaneous step — required for analytical relaxation calculations.

```python
SquarePulse(c0=0.0, c1=1e-6, t_on=0.0, t_off=0.1)
```

### `InstExpPulse`

Instantaneous rise, exponential decay.

```python
InstExpPulse(c0=0.0, c1=1e-6, t_on=0.0, tau_off=0.05)
```

### `PairedSquarePulse`

Two square pulses separated by a gap (paired-pulse protocol).

```python
PairedSquarePulse(c0=0.0, c1=1e-6, t_on1=0.0, t_off1=0.05,
                  t_on2=0.1, t_off2=0.15)
```

---

## Numerical solution

### `solve(mec, pulse, reclen, step, method='ode')`

Compute the macroscopic open-probability time course by integrating the
occupancy differential equation:

```
dP/dt = P · Q(c(t))
```

```python
result = solve(mec, pulse, reclen=0.3, step=1e-5)
t, c, Popen, P = result          # 4-tuple unpacking (backward compatible)
```

| Argument | Type | Description |
|----------|------|-------------|
| `mec` | Mechanism | Channel mechanism |
| `pulse` | Any pulse | Concentration profile |
| `reclen` | float | Total record length (s) |
| `step` | float | Time step (s) |
| `method` | str | `'ode'` (default) or `'matrix'` |

`method='ode'` uses `scipy.integrate.odeint`.  `method='matrix'` uses
piecewise matrix exponentials (faster for step pulses).

**Returns** `JumpResult` — a dataclass that supports 4-tuple unpacking for
backward compatibility with old code expecting `(t, c, Popen, P)`.

| Field | Shape | Description |
|-------|-------|-------------|
| `t` | (N,) | Time vector (s) |
| `c` | (N,) | Concentration at each time point (M) |
| `Popen` | (N,) | Open probability time course |
| `P` | (N, k) | Full state occupancy matrix |

---

## Analytical relaxation (square pulses only)

### `relaxation_taus(mec, pulse)`

Analytical on/off relaxation time constants and amplitudes for a
`SquarePulse`.

```python
result = relaxation_taus(mec, SquarePulse(c0=0, c1=1e-6, t_on=0, t_off=0.1))
# result.on_taus, result.on_amps, result.off_taus, result.off_amps
```

Raises `TypeError` if `pulse` is not a `SquarePulse`.

**Returns** `RelaxationResult`:

| Field | Description |
|-------|-------------|
| `on_taus` | On-relaxation time constants (s) |
| `on_amps` | On-relaxation amplitudes |
| `off_taus` | Off-relaxation time constants (s) |
| `off_amps` | Off-relaxation amplitudes |

---

## Jump summary and printout

### `jump_summary(mec, pulse, gamma=30e-12, Vm=-80e-3)`

All analytical jump properties as a plain `dict` — no formatting.  Includes
relaxation time constants, peak current (given single-channel conductance
`gamma` and holding potential `Vm`), and rise/decay times.

Requires a `SquarePulse`.

### `printout(mec, pulse, gamma=30e-12, Vm=-80e-3)`

Human-readable report built from `jump_summary()`.

---

## Usage example

```python
from scalcs.cjumps import SquarePulse, ErfPulse, solve, relaxation_taus
from scalcs.samples.CHME97 import CHME97

mec = CHME97()
mec.set_eff('c', 0.0)   # start at zero — solve() updates concentration internally

pulse = ErfPulse(c0=0.0, c1=1e-6, t_rise=0.0, t_fall=0.05,
                 rise_time=2e-4, fall_time=2e-4)

result = solve(mec, pulse, reclen=0.2, step=1e-5)
t, c, Popen, P = result

# Analytical relaxation for a square step
sq = SquarePulse(c0=0.0, c1=1e-6, t_on=0.0, t_off=0.05)
rel = relaxation_taus(mec, sq)
print('On time constants (ms):', rel.on_taus * 1000)
```

---

## References

- CH77: Colquhoun & Hawkes (1977)
- CH81: Colquhoun & Hawkes (1981)
- CHME97: Colquhoun, Hawkes, Merlushkin & Edmonds (1997)
