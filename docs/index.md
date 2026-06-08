# SCALCS Documentation

Pure Python implementations of Q-matrix formalisms for single ion channel
kinetic analysis.

## Modules

| Module | Description |
|--------|-------------|
| [mechanism](mechanism.md) | Kinetic scheme representation — states, rates, Q matrix |
| [qmatlib](qmatlib.md) | Q-matrix algebra: eigendecomposition, matrix exponentials, HJC matrices |
| [scalcslib](scalcslib.md) | Dwell-time pdfs (ideal / asymptotic / exact), correlations, simulation |
| [firstlatency](firstlatency.md) | First-latency pdf after a concentration jump |
| [cjumps](cjumps.md) | Macroscopic open-probability time course for concentration jumps |
| [pdfs](pdfs.md) | Exponential mixture pdf evaluation and tcrit utilities |
| [popen](popen.md) | Equilibrium open probability, EC50, Hill slope |
| [scburst](scburst.md) | Burst-length and opening-number distributions |
| [scalcsio](scalcsio.md) | File I/O — MEC, SCN, SSD and ABF formats |

## Quick start

```python
from scalcs.mechanism import Mechanism
from scalcs.samples.CHME97 import CHME97

mec = CHME97()
mec.set_eff('c', 0.001)   # 1 µM agonist
```

## References

See [references.md](references.md) for the full citation list.

Key papers:

- **CH82** — Colquhoun & Hawkes (1982) *Phil Trans R Soc Lond B* **300**, 1–59
- **HJC92** — Hawkes, Jalali & Colquhoun (1992) *Phil Trans R Soc Lond B* **337**, 383–404
- **CHME97** — Colquhoun, Hawkes, Merlushkin & Edmonds (1997) *Phil Trans R Soc Lond A* **355**, 1743–1786
- **CH95a/b** — Colquhoun & Hawkes (1995) in *Single-Channel Recording*, 2nd ed.
