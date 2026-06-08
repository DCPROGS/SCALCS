# mechanism — Kinetic scheme representation

`scalcs.mechanism`

Provides the `Mechanism` class, which holds the full kinetic scheme for an
ion channel: states, transition rates, effector dependences, and the assembled
Q matrix.

---

## State classification

States are divided into classes based on conductance and accessibility:

| Class | Symbol | Meaning |
|-------|--------|---------|
| Open | A | Conducting states |
| Blocked | B | Non-conducting, blocked |
| Shut (within cluster) | C | Non-conducting, reachable within a cluster |
| Gap (between clusters) | D | Long-lived non-conducting states |

The full Q matrix is ordered `[A, B, C, D]`. Submatrices are named `QAA`,
`QAF` (A→F where F = all shut), etc.

Key integer attributes on `Mechanism`:

| Attribute | Meaning |
|-----------|---------|
| `kA` | Number of open states |
| `kB` | Number of blocked states |
| `kC` | Number of shut-within-cluster states |
| `kD` | Number of between-cluster states |
| `kE = kA + kB` | Open + blocked |
| `kF = kB + kC + kD` | All shut (= k − kA) |
| `kG = kA + kB + kC` | States within cluster |
| `k`  | Total number of states |

---

## Core classes

### `State`

Represents a single kinetic state.

```python
State(name, conductance, state_class)
```

| Argument | Type | Description |
|----------|------|-------------|
| `name` | `str` | Label (e.g. `"R"`, `"AR"`, `"A2R*"`) |
| `conductance` | `float` | Single-channel conductance (S); 0 for shut states |
| `state_class` | `str` | `'A'`, `'B'`, `'C'`, or `'D'` |

### `Rate`

A directional transition between two states with an optional effector
dependence.

```python
Rate(rateconstants, State_from, State_to, name='', effectors=[], func=None)
```

The `func` argument is a callable `(rate_array, effdict) → float`.
Two built-in functions cover most cases:

| Function | Use |
|----------|-----|
| `identity(rate, effdict)` | Rate independent of effector — returns `rate[0]` |
| `multiply(rate, effdict)` | Rate proportional to a single effector — returns `rate[0] * effector_value` |

### `Mechanism`

Container for all states and rates; assembles the Q matrix.

```python
mec = Mechanism(rates, fastblock=False, fastKB=0)
```

#### Key methods

| Method | Description |
|--------|-------------|
| `set_eff(eff, value)` | Set effector (e.g. `mec.set_eff('c', 1e-6)` for 1 µM) and rebuild Q |
| `update_rates(theta)` | Update rate constants from a flat parameter vector |
| `printout()` | Human-readable rate table |

#### Key attributes set after `set_eff`

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `Q` | (k, k) | Full Q matrix |
| `QAA` | (kA, kA) | Open–open submatrix |
| `QFF` | (kF, kF) | Shut–shut submatrix |
| `QAF` | (kA, kF) | Open→shut submatrix |
| `QFA` | (kF, kA) | Shut→open submatrix |

---

## Rate constraint helpers

```python
constrain_rate_multiple(rate, factor)
```

Returns a rate-function closure that constrains a rate to be `factor` times
another rate constant.  Used to impose microscopic reversibility or other
linear constraints during fitting.

---

## Sample mechanisms

Pre-built mechanisms are in `scalcs/samples/`:

| Sample | Description |
|--------|-------------|
| `CO` | Simple two-state closed–open model |
| `CH82` | Colquhoun & Hawkes (1982) example scheme |
| `CHME97` | Five-state scheme from Colquhoun et al. (1997) |

```python
from scalcs.samples.CHME97 import CHME97
mec = CHME97()
mec.set_eff('c', 1e-6)   # 1 µM agonist
print(mec.kA, mec.kF)    # → 1 4
```

---

## Reference

- CH82: Colquhoun & Hawkes (1982) — state classification conventions
