SCALCS
=======

Pure Python implementations of Q-matrix formalisms for single ion channel
kinetic analysis.

Modules
-------

| Module            | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `scalcslib`       | Core calculations: asymptotic roots, dwell-time pdfs, P(open)|
| `firstlatency`    | First-latency (first-opening) pdf after a concentration jump  |
| `cjumps`          | Concentration-jump macroscopic current calculations           |
| `qmatlib`         | Q-matrix algebra: eigendecomposition, matrix exponentials     |
| `pdfs`            | Exponential mixture pdf evaluation                            |
| `scburst`         | Burst-length and cluster distributions                        |
| `popen`           | Equilibrium open probability                                  |
| `mechanism`       | Ion channel mechanism (state/rate) representation             |
| `scalcsio`        | HJCFIT / DC_PyPs file I/O                                     |

### First-latency pdf (`firstlatency`)

Computes the distribution of times from a rapid agonist concentration jump
to the first channel opening, at three levels of approximation:

- **Ideal** — no missed events; mixture of `kF` exponentials.
- **Asymptotic** — HJC approximation valid for `t >> tres`; `kF` components
  with roots from det[W_F(s)] = 0.
- **Exact** — HJC exact correction for `tres ≤ t < 3·tres`, asymptotic
  beyond.

Usage example:

```python
from scalcs import firstlatency as fl
from scalcs import qmatlib as qml
import numpy as np

# phi_shut: shut-state occupancies at c0 = 0
phi_shut = qml.pinf(mec.Q)[mec.kA:]
tres = 1e-3  # 1 ms dead time

roots = fl.asymptotic_roots(tres, mec)
areas = fl.asymptotic_areas(tres, roots, phi_shut, mec)
tau   = -1.0 / roots

t = np.linspace(tres, 0.1, 500)
f_asym  = fl.asymptotic_pdf(t, tres, tau, areas)

eigvals, g00, g10, g11 = fl.gamma_coefficients(tres, phi_shut, mec)
f_exact = fl.exact_pdf(t, tres, roots, areas, eigvals, g00, g10, g11)
```

References
----------

CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

HJC92: Hawkes AG, Jalali A, Colquhoun D (1992)
Asymptotic distributions of apparent open times and shut times in a
single channel record allowing for the omission of brief events.
Phil Trans R Soc Lond B 337, 383-404.

CHME97: Colquhoun D, Hawkes AG, Merlushkin A, Edmonds B (1997)
Properties of single ion channel currents elicited by a pulse of agonist
concentration or voltage.
Phil Trans R Soc Lond A 355, 1743-1786.

CH95a: Colquhoun D, Hawkes AG (1995a)
The principles of the stochastic interpretation of ion channel mechanisms.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 397-482.

CH95b: Colquhoun D, Hawkes AG (1995b)
A Q-Matrix Cookbook.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 589-633.

CHS96: Colquhoun D, Hawkes AG, Srodzinski K (1996)
Joint distributions of apparent open and shut times of single-ion channels
and maximum likelihood fitting of mechanisms.
Phil Trans R Soc Lond A 354, 2555-2590.
