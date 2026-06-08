First-latency pdf
*****************

.. automodule:: firstlatency
   :members:

Background
----------

The first-latency (or first-opening) pdf describes the distribution of times
from a rapid concentration jump to the first channel opening in a patch
containing a single ion channel.

Physical scenario
~~~~~~~~~~~~~~~~~

The channel is held at zero agonist concentration.  At *t* = 0 the
concentration steps to *c*₁ > 0.  Before the jump the entire population
occupies shut states with equilibrium probabilities

.. math::

   \phi_\text{shut} = \pi(Q_{c_0})[k_A:]

where :math:`\pi` denotes the equilibrium occupancy vector and the slice
discards the *k*\ :sub:`A` open-state entries.

Three approximation levels
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ideal** (no missed events)

   .. math::

      f_L(t) = \phi_\text{shut} \, e^{Q_{FF} t} \, (-Q_{FF}) \, \mathbf{u}_F

   This is a mixture of :math:`k_F` exponentials with rates equal to the
   eigenvalues of :math:`-Q_{FF}`.

**Asymptotic** (HJC approximation, valid for :math:`t \gg t_{res}`)

   The :math:`k_F` roots :math:`s_i` solve :math:`\det[W_F(s)] = 0`
   (the same eigenvalue equation as for the shut-time asymptotic pdf,
   with Q-submatrix roles exchanged: A ↔ F).  The component areas use
   :math:`\phi_\text{shut}` as the initial vector rather than the HJC
   equilibrium vector used in ordinary shut-time distributions.

**Exact** (HJC exact correction)

   Applies corrections for :math:`t_{res} \le t < 3\,t_{res}` and
   reverts to the asymptotic form beyond:

   .. math::

      f(t) =
      \begin{cases}
        0 & t < t_{res} \\
        f_0(t - t_{res}) & t_{res} \le t < 2\,t_{res} \\
        f_0(t - t_{res}) - f_1(t - 2\,t_{res}) & 2\,t_{res} \le t < 3\,t_{res} \\
        \text{expPDF}(t - t_{res},\, \tau,\, \text{areas}) & t \ge 3\,t_{res}
      \end{cases}

Numerical stability notes
~~~~~~~~~~~~~~~~~~~~~~~~~

The matrix exponential :math:`\exp((Q_{FF} - sI)\,t_{res})` evaluated at the
lower search boundary :math:`s_{as}` can overflow float64 when
:math:`|s_{as}| \times t_{res} > \ln(\text{float64\_max}) \approx 709`.
For :math:`t_{res} = 1\,\text{ms}` this occurs at
:math:`|s_{as}| > 7 \times 10^5`.  The adaptive bound

.. math::

   s_{as} = \max(-10^6,\; -700 / t_{res})

prevents overflow while keeping the bracket wide enough to locate all roots.

At :math:`t_{res} = 0` the matrix :math:`H(s)` becomes constant
(:math:`= Q_{AA}`) and the root-counting function :math:`g_{FB}(s)` is a
step function.  The bisection algorithm discards sub-intervals that contain
zero roots (rather than re-queuing them) to avoid an infinite loop.

References
----------

CH82
  Colquhoun D, Hawkes AG (1982). On the stochastic properties of bursts of
  single ion channel openings and of clusters of bursts.
  *Phil Trans R Soc Lond B* **300**, 1–59.

HJC92
  Hawkes AG, Jalali A, Colquhoun D (1992). Asymptotic distributions of
  apparent open times and shut times in a single channel record allowing for
  the omission of brief events.
  *Phil Trans R Soc Lond B* **337**, 383–404.

CHME97
  Colquhoun D, Hawkes AG, Merlushkin A, Edmonds B (1997). Properties of
  single ion channel currents elicited by a pulse of agonist concentration or
  voltage.
  *Phil Trans R Soc Lond A* **355**, 1743–1786.
