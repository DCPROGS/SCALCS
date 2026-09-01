"""Macroscopic responses to concentration jumps.

This module calculates the time course of open probability and state
occupancies when the agonist (or any effector) concentration is changed as
a function of time.  The calculation uses the Q-matrix formalism of
Colquhoun & Hawkes (1977, 1981) and is the basis for simulating
concentration-clamp or fast-perfusion experiments.

Public API
----------
Pulse dataclasses  (one per concentration profile shape)
  ErfPulse          Realistic jump: error-function rise and fall.  This is the
                    profile used in fast-perfusion ('rcj') experiments.
  SquarePulse       Ideal square step.
  InstExpPulse      Instantaneous rise, exponential decay.
  PairedSquarePulse Two square pulses separated by a gap.

Result dataclasses
  JumpResult        Time-course output of solve(). Supports 4-tuple unpacking
                    ``t, c, Popen, P = result`` for backward compatibility.
  RelaxationResult  Analytical on/off time constants and amplitudes returned by
                    relaxation_taus().

Functions
  solve(mec, pulse, reclen, step, method='ode')
      Compute the macroscopic open-probability time course.

  relaxation_taus(mec, pulse)
      Analytical on/off relaxation time constants for a SquarePulse.

  jump_summary(mec, pulse, gamma=30e-12, Vm=-80e-3)
      All analytical jump properties as a plain dict — no formatting.

  printout(mec, pulse, gamma=30e-12, Vm=-80e-3)
      Human-readable report built from jump_summary().

Private helpers (prefixed ``_``)
  _dPdt, _P_t, _coefficient_calc, _solve_ode, _solve_matrix

Notes
-----
``relaxation_taus``, ``jump_summary`` and ``printout`` require a
``SquarePulse`` because the analytical expressions for on/off relaxation are
derived for ideal concentration steps.  Pass an ``ErfPulse`` and a
``TypeError`` is raised with a clear message.

References
----------
Colquhoun D & Hawkes AG (1977) Relaxation and fluctuations of membrane
  currents that flow through drug-operated channels.
  Proc R Soc Lond B 199, 231-262.

Colquhoun D & Hawkes AG (1981) On the stochastic properties of single ion
  channels.  Proc R Soc Lond B 211, 205-235.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import scipy.integrate as scpi
from scipy.special import erf

from scalcs import qmatlib as qml


# ===========================================================================
# Pulse profile dataclasses
# ===========================================================================

@dataclass
class ErfPulse:
    """Realistic concentration jump with error-function rise and fall.

    This profile matches the waveform produced by fast perfusion ('rcj')
    experiments in which the solution exchange is not instantaneous.  The
    concentration is::

        c(t) = cb + cmax/2 * [erf((t - centre + width/2) / rise)
                               - erf((t - centre - width/2) / decay)]

    Parameters
    ----------
    cmax : float
        Peak concentration (M).  Must be > 0.
    width : float
        Pulse half-width (s).  Must be > 0.
    cb : float
        Background (baseline) concentration (M).  Default 0.
    centre : float
        Time of the pulse centre (s).  Default 10 ms.
    rise : float
        10–90 % rise time constant for the error function (s).  Default 200 µs.
    decay : float
        90–10 % decay time constant for the error function (s).  Default 200 µs.
    """

    cmax:   float
    width:  float
    cb:     float = 0.0
    centre: float = 10e-3
    rise:   float = 200e-6
    decay:  float = 200e-6

    def __post_init__(self):
        if self.cmax <= 0:
            raise ValueError(f"ErfPulse: cmax must be > 0, got {self.cmax}")
        if self.width <= 0:
            raise ValueError(f"ErfPulse: width must be > 0, got {self.width}")
        if self.rise <= 0:
            raise ValueError(f"ErfPulse: rise must be > 0, got {self.rise}")
        if self.decay <= 0:
            raise ValueError(f"ErfPulse: decay must be > 0, got {self.decay}")

    def profile(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Concentration at time(s) *t* (M).

        Parameters
        ----------
        t : float or ndarray
            Time in seconds.

        Returns
        -------
        c : float or ndarray
            Concentration in M.  Same shape as *t*.
        """
        conc = (self.cmax * 0.5 *
                (erf((t - self.centre + self.width / 2.) / self.rise) -
                 erf((t - self.centre - self.width / 2.) / self.decay)))
        return conc + self.cb


@dataclass
class SquarePulse:
    """Ideal square concentration step.

    The concentration profile is::

        c(t) = cmax   if  prepulse < t <= prepulse + width
               cb     otherwise

    Parameters
    ----------
    cmax : float
        Pulse concentration (M).  Must be > 0.
    width : float
        Pulse duration (s).  Must be > 0.
    cb : float
        Background concentration (M).  Default 0.
    prepulse : float
        Time before the pulse starts (s).  Default 5 ms.
    """

    cmax:     float
    width:    float
    cb:       float = 0.0
    prepulse: float = 5e-3

    def __post_init__(self):
        if self.cmax <= 0:
            raise ValueError(f"SquarePulse: cmax must be > 0, got {self.cmax}")
        if self.width <= 0:
            raise ValueError(f"SquarePulse: width must be > 0, got {self.width}")

    def profile(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Concentration at time(s) *t* (M)."""
        if np.isscalar(t):
            conc = (self.cmax
                    if self.prepulse < t <= self.prepulse + self.width
                    else 0.0)
        else:
            t = np.asarray(t)
            conc = np.where(
                (t > self.prepulse) & (t <= self.prepulse + self.width),
                self.cmax, 0.0)
        return conc + self.cb


@dataclass
class InstExpPulse:
    """Concentration pulse with instantaneous rise and exponential decay.

    The profile is::

        c(t) = cb                                    if t <= prepulse
               cb + cmax * exp(-(t-prepulse) / tdec) if t >  prepulse

    Parameters
    ----------
    cmax : float
        Peak concentration immediately after the step (M).  Must be > 0.
    tdec : float
        Exponential decay time constant (s).  Must be > 0.
    cb : float
        Background concentration (M).  Default 0.
    prepulse : float
        Time before the step (s).  Default 5 ms.
    """

    cmax:     float
    tdec:     float
    cb:       float = 0.0
    prepulse: float = 5e-3

    def __post_init__(self):
        if self.cmax <= 0:
            raise ValueError(f"InstExpPulse: cmax must be > 0, got {self.cmax}")
        if self.tdec <= 0:
            raise ValueError(f"InstExpPulse: tdec must be > 0, got {self.tdec}")

    def profile(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Concentration at time(s) *t* (M)."""
        if np.isscalar(t):
            conc = (self.cmax * math.exp(-(t - self.prepulse) / self.tdec)
                    if t > self.prepulse else 0.0)
        else:
            t = np.asarray(t, dtype=float)
            conc = np.where(
                t > self.prepulse,
                self.cmax * np.exp(-(t - self.prepulse) / self.tdec),
                0.0)
        return conc + self.cb


@dataclass
class PairedSquarePulse:
    """Two square concentration pulses separated by a gap.

    The profile is::

        c(t) = cmax   if  prepulse < t <= prepulse + width
               cmax   if  prepulse + width + inter < t
                          <= prepulse + 2*width + inter
               cb     otherwise

    Parameters
    ----------
    cmax : float
        Pulse concentration (M).  Must be > 0.
    width : float
        Duration of each pulse (s).  Must be > 0.
    inter : float
        Gap between the two pulses (s).  Must be >= 0.
    cb : float
        Background concentration (M).  Default 0.
    prepulse : float
        Time before the first pulse (s).  Default 5 ms.
    """

    cmax:     float
    width:    float
    inter:    float
    cb:       float = 0.0
    prepulse: float = 5e-3

    def __post_init__(self):
        if self.cmax <= 0:
            raise ValueError(f"PairedSquarePulse: cmax must be > 0, got {self.cmax}")
        if self.width <= 0:
            raise ValueError(f"PairedSquarePulse: width must be > 0, got {self.width}")
        if self.inter < 0:
            raise ValueError(f"PairedSquarePulse: inter must be >= 0, got {self.inter}")

    def profile(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Concentration at time(s) *t* (M)."""
        p, w, g = self.prepulse, self.width, self.inter
        if np.isscalar(t):
            in_p1 = p < t <= p + w
            in_p2 = p + w + g < t <= p + 2 * w + g
            conc = self.cmax if (in_p1 or in_p2) else 0.0
        else:
            t = np.asarray(t, dtype=float)
            in_p1 = (t > p) & (t <= p + w)
            in_p2 = (t > p + w + g) & (t <= p + 2 * w + g)
            conc = np.where(in_p1 | in_p2, self.cmax, 0.0)
        return conc + self.cb


# Convenience type alias used in type hints below
AnyPulse = Union[ErfPulse, SquarePulse, InstExpPulse, PairedSquarePulse]


# ===========================================================================
# Result dataclasses
# ===========================================================================

@dataclass
class JumpResult:
    """Output of :func:`solve`.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Time axis (s).
    c : ndarray, shape (n,)
        Concentration profile (M) as seen by the mechanism.
    Popen : ndarray, shape (n,)
        Open probability at each time point.
    P : ndarray, shape (k, n)
        Occupancy of every state (rows) at every time point (columns).
    pulse : AnyPulse
        The pulse object used to generate this result.

    Notes
    -----
    The result supports 4-tuple unpacking for backward compatibility::

        t, c, Popen, P = solve(mec, pulse, reclen, step)
    """

    t:      np.ndarray
    c:      np.ndarray
    Popen:  np.ndarray
    P:      np.ndarray
    pulse:  AnyPulse

    def __iter__(self):
        """Yield (t, c, Popen, P) so the result can be unpacked as a 4-tuple."""
        yield self.t
        yield self.c
        yield self.Popen
        yield self.P


@dataclass
class RelaxationResult:
    """Output of :func:`relaxation_taus`.

    Attributes
    ----------
    tau_on_weighted : float
        Amplitude-weighted mean on-relaxation time constant (s).
    tau_off_weighted : float
        Amplitude-weighted mean off-relaxation time constant (s).
    tau_on : ndarray, shape (k-1,)
        Individual on-relaxation time constants (s), sorted fastest first.
    tau_off : ndarray, shape (k-1,)
        Individual off-relaxation time constants (s), sorted fastest first.
    ampl_on : ndarray, shape (k-1,)
        Relative amplitudes for each on-relaxation component.
    ampl_off : ndarray, shape (k-1,)
        Relative amplitudes for each off-relaxation component.
    """

    tau_on_weighted:  float
    tau_off_weighted: float
    tau_on:           np.ndarray
    tau_off:          np.ndarray
    ampl_on:          np.ndarray
    ampl_off:         np.ndarray


# ===========================================================================
# Private numerical core
# ===========================================================================

def _dPdt(P, t, mec, pulse):
    """Rate of change of state occupancies: dP/dt = P · Q(c(t)).

    Used as the right-hand side for scipy.odeint.

    Parameters
    ----------
    P : ndarray, shape (k,)
        Current state occupancies.
    t : float
        Current time (s).
    mec : Mechanism
        The mechanism (Q matrix updated via set_eff).
    pulse : AnyPulse
        Concentration profile; supplies c(t) via pulse.profile(t).
    """
    mec.set_eff('c', float(pulse.profile(t)))
    return np.dot(P, mec.Q)


def _P_t(t, eigs, w):
    """State occupancies at time *t* from eigendecomposition.

    Parameters
    ----------
    t : float
        Time elapsed since the step (s).
    eigs : ndarray, shape (k,)
        Eigenvalues of Q.
    w : ndarray, shape (k, k)
        Weighted spectral components from :func:`_coefficient_calc`.

    Returns
    -------
    Pt : ndarray, shape (k,)
        State occupancies at time *t*.
    """
    Pt = np.zeros(eigs.shape)
    for i in range(eigs.size):
        Pt[i] = np.sum(w[:, i] * np.exp(eigs * t))
    return Pt


def _coefficient_calc(k, A, p_occup):
    """Weighted spectral components p · A_n for each spectral matrix A_n.

    Parameters
    ----------
    k : int
        Number of states.
    A : ndarray, shape (k, k, k)
        Spectral matrices of Q.
    p_occup : ndarray, shape (k,)
        State occupancies at the start of the interval.

    Returns
    -------
    w : ndarray, shape (k, k)
        Element w[n, i] = (p · A_n)[i].
    """
    w = np.zeros((k, k))
    for n in range(k):
        w[n, :] = np.dot(p_occup, A[n, :, :])
    return w


def _solve_ode(mec, pulse, reclen, step):
    """Solve by ODE integration (scipy.odeint).

    Accurate for smoothly varying concentration profiles (e.g. ErfPulse).
    Also works for piecewise profiles, though the matrix method may be faster
    for those.

    Parameters
    ----------
    mec : Mechanism
    pulse : AnyPulse
    reclen : float   Recording length (s).
    step : float     Sampling interval (s).

    Returns
    -------
    t, c, Popen, P : as described in JumpResult.
    """
    t = np.arange(0, reclen, step)
    mec.set_eff('c', pulse.cb)
    P0 = qml.pinf(mec.Q)
    Pt = scpi.odeint(_dPdt, P0, t, args=(mec, pulse),
                     atol=1e-8, rtol=1e-6)
    P = Pt.transpose()
    Popen = np.sum(P[:mec.kA], axis=0)
    c = pulse.profile(t)
    return t, c, Popen, P


def _solve_matrix(mec, pulse, reclen, step):
    """Solve step-wise using the Q-matrix eigendecomposition.

    At each time step the concentration is treated as constant and the
    occupancies are propagated analytically via P(t+dt) = P(t) · exp(Q·dt).
    Accurate for profiles that are piecewise constant (SquarePulse,
    PairedSquarePulse) and useful as an independent check against _solve_ode.

    Parameters
    ----------
    mec : Mechanism
    pulse : AnyPulse
    reclen : float
    step : float

    Returns
    -------
    t, c, Popen, P : as described in JumpResult.
    """
    t = np.arange(0, reclen, step)
    c = pulse.profile(t)
    mec.set_eff('c', pulse.cb)
    pi = qml.pinf(mec.Q)
    Pt = np.array([pi.copy()])

    for i in range(1, t.shape[0]):
        mec.set_eff('c', float(c[i]))
        eigenvals, A = qml.eigs_sorted(mec.Q)
        w = _coefficient_calc(mec.k, A, pi)
        pi = _P_t(step, eigenvals, w)
        Pt = np.append(Pt, [pi.copy()], axis=0)

    P = Pt.transpose()
    Popen = np.sum(P[:mec.kA], axis=0)
    return t, c, Popen, P


# ===========================================================================
# Public API
# ===========================================================================

def solve(mec, pulse: AnyPulse, reclen: float, step: float,
          method: str = 'ode') -> JumpResult:
    """Compute the macroscopic open-probability response to a concentration pulse.

    Starting from the equilibrium state at the background concentration
    ``pulse.cb``, the state occupancies are evolved forward in time according
    to dP/dt = P · Q(c(t)).

    Parameters
    ----------
    mec : Mechanism
        The ion-channel mechanism (must have set_eff, Q, kA, k attributes).
    pulse : ErfPulse | SquarePulse | InstExpPulse | PairedSquarePulse
        Concentration profile.  The pulse's ``.cb`` attribute sets the initial
        equilibrium condition.
    reclen : float
        Total recording length (s).
    step : float
        Sampling interval (s).  ``reclen / step`` time points are returned.
    method : {'ode', 'matrix'}
        Numerical method:

        ``'ode'``
            Scipy ODE integrator (``odeint``).  Robust for all pulse shapes,
            especially smooth profiles like :class:`ErfPulse`.
        ``'matrix'``
            Step-wise Q-matrix eigendecomposition.  Each time step is treated
            as a constant-concentration interval.  Useful as an independent
            numerical check; may be slower.

    Returns
    -------
    JumpResult
        Named result with attributes ``t``, ``c``, ``Popen``, ``P``, ``pulse``.
        Supports 4-tuple unpacking: ``t, c, Popen, P = result``.

    Raises
    ------
    ValueError
        If *method* is not ``'ode'`` or ``'matrix'``.

    Examples
    --------
    >>> from scalcs.samples.samples import CH82
    >>> from scalcs.cjumps import ErfPulse, solve
    >>> mec = CH82()
    >>> pulse = ErfPulse(cmax=1e-6, width=10e-3)
    >>> result = solve(mec, pulse, reclen=50e-3, step=5e-6)
    >>> t, c, Popen, P = result          # backward-compatible unpacking
    """
    if method == 'ode':
        t, c, Popen, P = _solve_ode(mec, pulse, reclen, step)
    elif method == 'matrix':
        t, c, Popen, P = _solve_matrix(mec, pulse, reclen, step)
    else:
        raise ValueError(
            f"solve: unknown method {method!r}. Choose 'ode' or 'matrix'.")
    return JumpResult(t=t, c=c, Popen=Popen, P=P, pulse=pulse)


def relaxation_taus(mec, pulse: SquarePulse) -> RelaxationResult:
    """Analytical on/off relaxation time constants for a square concentration step.

    For a SquarePulse the on-relaxation is governed by the eigenvalues of Q at
    ``pulse.cmax`` (starting from equilibrium at ``pulse.cb``), and the
    off-relaxation by the eigenvalues of Q at ``pulse.cb`` (starting from the
    occupancies at the end of the pulse).

    Parameters
    ----------
    mec : Mechanism
    pulse : SquarePulse
        Must be a :class:`SquarePulse`.  Passing any other pulse type raises
        ``TypeError`` because the analytical expressions assume an ideal step.

    Returns
    -------
    RelaxationResult
        Attributes: ``tau_on_weighted``, ``tau_off_weighted`` (floats),
        ``tau_on``, ``tau_off``, ``ampl_on``, ``ampl_off`` (ndarrays, k-1).

    Raises
    ------
    TypeError
        If *pulse* is not a :class:`SquarePulse`.

    Notes
    -----
    This replaces the old ``weighted_taus()`` which returned only 2 scalars,
    breaking the ``sccurves`` call that expected 4 return values.
    """
    if not isinstance(pulse, SquarePulse):
        raise TypeError(
            "relaxation_taus() requires a SquarePulse (ideal step). "
            f"Got {type(pulse).__name__}. "
            "For non-square profiles use solve() and inspect the Popen trace.")

    # Equilibrium at background concentration
    mec.set_eff('c', pulse.cb)
    eigs0, A0 = qml.eigs_sorted(mec.Q)
    P0 = qml.pinf(mec.Q)

    # Equilibrium at pulse concentration
    mec.set_eff('c', pulse.cmax)
    eigsInf, Ainf = qml.eigs_sorted(mec.Q)

    # On-relaxation: step from cb → cmax
    w_on = _coefficient_calc(mec.k, Ainf, P0)
    ampl_on_raw = np.sum(w_on[:, :mec.kA], axis=1)          # shape (k,)
    max_ampl_on = np.max(np.abs(ampl_on_raw))
    rel_ampl_on = ampl_on_raw / max_ampl_on                  # normalised
    tau_on = -1.0 / eigsInf[:-1]                             # k-1 time constants (s)
    # The dominant component typically has a negative relative amplitude
    # (it drives occupancy toward the new equilibrium).  Negating gives the
    # conventional positive weighted time constant (matches DCPROGS HJC_JUMP).
    tau_on_weighted = float(np.sum(-rel_ampl_on[:-1] * tau_on))

    # Occupancies at end of pulse
    Pt = _P_t(pulse.width, eigsInf, w_on)

    # Off-relaxation: step from cmax → cb
    w_off = _coefficient_calc(mec.k, A0, Pt)
    ampl_off_raw = np.sum(w_off[:, :mec.kA], axis=1)
    max_ampl_off = np.max(np.abs(ampl_off_raw))
    rel_ampl_off = ampl_off_raw / max_ampl_off
    tau_off = -1.0 / eigs0[:-1]
    tau_off_weighted = float(np.sum(rel_ampl_off[:-1] * tau_off))

    return RelaxationResult(
        tau_on_weighted=tau_on_weighted,
        tau_off_weighted=tau_off_weighted,
        tau_on=tau_on,
        tau_off=tau_off,
        ampl_on=rel_ampl_on[:-1],
        ampl_off=rel_ampl_off[:-1],
    )


def jump_summary(mec, pulse: SquarePulse,
                 gamma: float = 30e-12,
                 Vm: float = -80e-3) -> dict:
    """Analytical properties of a square concentration jump.

    Computes all quantities needed for ``printout()`` and returns them as a
    plain dict so that each quantity is individually testable.

    Parameters
    ----------
    mec : Mechanism
    pulse : SquarePulse
        Must be a :class:`SquarePulse`; raises ``TypeError`` otherwise.
    gamma : float
        Single-channel conductance (S).  Default 30 pS.
    Vm : float
        Transmembrane voltage (V).  Default −80 mV.

    Returns
    -------
    dict with keys:

    ``p0``               ndarray — equilibrium occupancies before pulse.
    ``pinf``             ndarray — equilibrium occupancies at cmax.
    ``pt``               ndarray — occupancies at end of pulse.
    ``eigenvalues_on``   ndarray, shape (k-1,) — eigenvalues during on-phase (s⁻¹).
    ``eigenvalues_off``  ndarray, shape (k-1,) — eigenvalues during off-phase (s⁻¹).
    ``amplitudes_on``    ndarray, shape (k-1,) — current amplitudes, on-phase (A).
    ``amplitudes_off``   ndarray, shape (k-1,) — current amplitudes, off-phase (A).
    ``rel_amplitudes_on``   ndarray — normalised (dimensionless).
    ``rel_amplitudes_off``  ndarray — normalised.
    ``areas_on``         ndarray, shape (k-1,) — charge under each on component (C).
    ``areas_off``        ndarray, shape (k-1,) — charge under each off component (C).
    ``tau_on_weighted``  float — weighted on time constant (s).
    ``tau_off_weighted`` float — weighted off time constant (s).
    ``gamma``            float — conductance used (S).
    ``Vm``               float — voltage used (V).

    Raises
    ------
    TypeError
        If *pulse* is not a :class:`SquarePulse`.
    """
    if not isinstance(pulse, SquarePulse):
        raise TypeError(
            "jump_summary() requires a SquarePulse. "
            f"Got {type(pulse).__name__}.")

    # --- equilibrium before pulse ---
    mec.set_eff('c', pulse.cb)
    P0 = qml.pinf(mec.Q)
    eigs0, A0 = qml.eigs_sorted(mec.Q)

    # --- equilibrium at peak concentration ---
    mec.set_eff('c', pulse.cmax)
    Pinf = qml.pinf(mec.Q)
    eigsInf, Ainf = qml.eigs_sorted(mec.Q)

    # --- on-relaxation ---
    w_on = _coefficient_calc(mec.k, Ainf, P0)
    ampl_on_raw = np.sum(w_on[:, :mec.kA], axis=1)          # shape (k,)
    cur_on = ampl_on_raw * gamma * Vm                        # current (A)
    max_ampl_on = np.max(np.abs(ampl_on_raw))
    rel_ampl_on = ampl_on_raw / max_ampl_on
    tau_on = -1.0 / eigsInf[:-1]                            # k-1 time constants (s)
    area_on = -cur_on[:-1] / eigsInf[:-1]                   # charge (C)
    tau_on_weighted = float(np.sum(-rel_ampl_on[:-1] * tau_on))

    # --- occupancies at end of pulse ---
    Pt = _P_t(pulse.width, eigsInf, w_on)

    # --- off-relaxation ---
    w_off = _coefficient_calc(mec.k, A0, Pt)
    ampl_off_raw = np.sum(w_off[:, :mec.kA], axis=1)
    cur_off = ampl_off_raw * gamma * Vm
    max_ampl_off = np.max(np.abs(ampl_off_raw))
    rel_ampl_off = ampl_off_raw / max_ampl_off
    tau_off = -1.0 / eigs0[:-1]
    area_off = np.zeros(mec.k - 1)
    for i in range(mec.k - 1):
        area_off[i] = -1000.0 * cur_off[i] / eigs0[i]      # in pC·ms (×1000)
    tau_off_weighted = float(np.sum(rel_ampl_off[:-1] * tau_off))

    return {
        'p0':                 P0,
        'pinf':               Pinf,
        'pt':                 Pt,
        'eigenvalues_on':     eigsInf[:-1],
        'eigenvalues_off':    eigs0[:-1],
        'amplitudes_on':      cur_on[:-1],
        'amplitudes_off':     cur_off[:-1],
        'rel_amplitudes_on':  rel_ampl_on[:-1],
        'rel_amplitudes_off': rel_ampl_off[:-1],
        'areas_on':           area_on,
        'areas_off':          area_off,
        'tau_on_weighted':    tau_on_weighted,
        'tau_off_weighted':   tau_off_weighted,
        'gamma':              gamma,
        'Vm':                 Vm,
    }


def printout(mec, pulse: SquarePulse,
             gamma: float = 30e-12,
             Vm: float = -80e-3) -> str:
    """Human-readable summary of concentration jump properties.

    Produces the same analytical printout as the original ``cjumps.printout``
    but with configurable single-channel conductance and voltage (no more
    hard-coded values).

    Parameters
    ----------
    mec : Mechanism
    pulse : SquarePulse
        Must be a :class:`SquarePulse`; raises ``TypeError`` otherwise.
    gamma : float
        Single-channel conductance (S).  Default 30 pS.
    Vm : float
        Transmembrane voltage (V).  Default −80 mV.

    Returns
    -------
    str
        Formatted text report.

    Raises
    ------
    TypeError
        If *pulse* is not a :class:`SquarePulse`.
    """
    if not isinstance(pulse, SquarePulse):
        raise TypeError(
            "printout() requires a SquarePulse. "
            f"Got {type(pulse).__name__}. "
            "For other profiles, use solve() to obtain the Popen time course.")

    s = jump_summary(mec, pulse, gamma=gamma, Vm=Vm)
    k = mec.k

    out = ('\n*******************************************\n'
           'CONCENTRATION JUMPS\n')

    out += ('\nEquilibrium occupancies before t=0, at concentration = '
            '{:.5g} mM:\n'.format(pulse.cb * 1000))
    for i in range(k):
        out += '  p0({:d}) = {:.5g}\n'.format(i + 1, s['p0'][i])

    out += ('\nEquilibrium occupancies at peak concentration = '
            '{:.5g} mM:\n'.format(pulse.cmax * 1000))
    for i in range(k):
        out += '  pinf({:d}) = {:.5g}\n'.format(i + 1, s['pinf'][i])

    out += '\nOccupancies at end of {:.5g} ms pulse:\n'.format(pulse.width * 1000)
    for i in range(k):
        out += '  pt({:d}) = {:.5g}\n'.format(i + 1, s['pt'][i])

    out += ('\nSingle-channel conductance: {:.5g} pS\n'
            'Transmembrane voltage:      {:.5g} mV\n'.format(
                gamma * 1e12, Vm * 1e3))

    out += ('\nON-RELAXATION (ideal step {:s} → {:s} mM):\n'
            '  {:>4s}  {:>14s}  {:>10s}  {:>12s}  {:>12s}\n'.format(
                '{:.5g}'.format(pulse.cb * 1000),
                '{:.5g}'.format(pulse.cmax * 1000),
                'Comp', 'Eigenvalue', 'Tau (ms)',
                'Ampl (t=0, pA)', 'Rel. ampl.'))
    for i in range(k - 1):
        out += ('  {:>4d}  {:>14.5g}  {:>10.5g}  {:>12.5g}  {:>12.5g}\n'.format(
            i + 1,
            s['eigenvalues_on'][i],
            -1000.0 / s['eigenvalues_on'][i],
            s['amplitudes_on'][i] * 1e12,        # pA
            s['rel_amplitudes_on'][i]))
    out += ('  Weighted on tau = {:.5g} ms\n'.format(s['tau_on_weighted'] * 1000))
    out += ('  Total current at t=0  = {:.5g} pA\n'.format(
        np.sum(s['amplitudes_on']) * 1e12))
    cur_on_eq = s['amplitudes_on']   # last component is equilibrium term
    out += ('  Total area (on)       = {:.5g} pC\n'.format(
        np.sum(s['areas_on']) * 1e3))   # stored in C, convert to pC

    out += ('\nOFF-RELAXATION (ideal step {:s} → {:s} mM):\n'
            '  {:>4s}  {:>14s}  {:>10s}  {:>12s}  {:>12s}\n'.format(
                '{:.5g}'.format(pulse.cmax * 1000),
                '{:.5g}'.format(pulse.cb * 1000),
                'Comp', 'Eigenvalue', 'Tau (ms)',
                'Ampl (t=0, pA)', 'Rel. ampl.'))
    for i in range(k - 1):
        out += ('  {:>4d}  {:>14.5g}  {:>10.5g}  {:>12.5g}  {:>12.5g}\n'.format(
            i + 1,
            s['eigenvalues_off'][i],
            -1000.0 / s['eigenvalues_off'][i],
            s['amplitudes_off'][i] * 1e12,
            s['rel_amplitudes_off'][i]))
    out += ('  Weighted off tau = {:.5g} ms\n'.format(s['tau_off_weighted'] * 1000))
    out += ('  Total current at t=0  = {:.5g} pA\n'.format(
        np.sum(s['amplitudes_off']) * 1e12))
    out += ('  Total area (off)      = {:.5g} pC\n'.format(
        np.sum(s['areas_off'])))

    return out
