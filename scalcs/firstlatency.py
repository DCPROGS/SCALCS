"""First-latency pdf: time from concentration step to first channel opening.

Physical scenario — step from zero
------------------------------------
The channel is held at zero agonist concentration.  At t = 0 the entire
population is in shut states with equilibrium probabilities

    phi_shut = pinf(Q_at_c0)[kA:]

At t = 0+ the concentration steps to c1 > 0.  The first-latency pdf f_L(t)
is the probability density of the time to the first opening.

Three levels of approximation
------------------------------
ideal       No missed events.
                f_L(t) = phi_shut · exp(QFF_c1 · t) · (−QFF_c1) · u_F
            Spectral expansion: kF exponentials with rates = eigenvalues of −QFF.

asymptotic  HJC approximation (Colquhoun & Hawkes 1982).  Valid for t >> tres.
            kF components whose roots solve det[W_F(s)] = 0.  Same roots as the
            ordinary shut-time asymptotic pdf; only the area vector changes because
            phi_shut replaces the HJC equilibrium vector.

exact       HJC exact solution for tres ≤ t ≤ 3·tres, asymptotic beyond.

References
----------
CHME97 : Colquhoun, Hawkes, Merlushkin & Edmonds (1997)
         Phil Trans R Soc Lond A 355, 1743-1786.
CH82   : Colquhoun & Hawkes (1982)
         Phil Trans R Soc Lond B 300, 1-59.
HJC92  : Hawkes, Jalali & Colquhoun (1992)
         Phil Trans R Soc Lond B 337, 383-404.
"""

__author__ = "DC_PyPs project"

from types import SimpleNamespace

import numpy as np

from scalcs import qmatlib as qml
from scalcs import scalcslib as scl
from scalcs import pdfs


# ---------------------------------------------------------------------------
# Ideal (no missed events)
# ---------------------------------------------------------------------------

def ideal_components(QFF, phi_shut):
    """Eigenvalues and normalised areas for the ideal first-latency pdf.

    The ideal pdf is

        f_L(t) = sum_i  areas[i] · eigs[i] · exp(−eigs[i] · t)

    so that each component area integrates to areas[i] (and sum = 1).

    Parameters
    ----------
    QFF : ndarray, shape (kF, kF)
        Shut–shut submatrix of Q at post-jump concentration c1.
    phi_shut : ndarray, shape (kF,)
        Equilibrium shut-state occupancies at pre-jump concentration c0.

    Returns
    -------
    eigs : ndarray, shape (kF,)
        Eigenvalues of −QFF (positive reals).
    areas : ndarray, shape (kF,)
        Component areas; sum(areas) = 1 for a proper distribution.
    """
    eigs, w = scl.ideal_dwell_time_pdf_components(QFF, phi_shut)
    areas = w / eigs
    return eigs, areas


def ideal_pdf(t, QFF, phi_shut):
    """Evaluate the ideal first-latency pdf at time(s) t.

    f_L(t) = phi_shut · exp(QFF · t) · (−QFF) · u_F

    Parameters
    ----------
    t : float or ndarray
        Time (s).
    QFF : ndarray, shape (kF, kF)
    phi_shut : ndarray, shape (kF,)

    Returns
    -------
    f : float (if t is scalar) or ndarray (if t is array)
        pdf value(s) in s⁻¹.
    """
    eigs, areas = ideal_components(QFF, phi_shut)
    tau = 1.0 / eigs
    scalar_input = np.ndim(t) == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    f = pdfs.expPDF(t_arr, tau, areas)
    if scalar_input:
        return float(f[0])
    return f


# ---------------------------------------------------------------------------
# Ideal first-latency pdf of a concentration PULSE  (CHME97 §3(i))
# ---------------------------------------------------------------------------
#
# The channel is held at zero agonist concentration; at t = 0 the concentration
# steps to c1 (post-jump mechanism ``mec1``), is held for a pulse of duration T,
# then returns to zero (``mec0``) at t = T.  The ideal (no missed events)
# first-latency pdf, conditional on at least one opening (R >= 1), splits at the
# end of the pulse:
#
#   t < T  : first opening occurs while agonist is present (Eq 3.6).  Same kF
#            exponentials as the step-from-zero ideal pdf, only rescaled by the
#            conditioning factor 1 / P(R>=1).
#   t > T  : first opening occurs after agonist is removed (Eq 3.8).  Only the
#            within-burst shut states B can still open (C is absorbing at c = 0),
#            so the pdf is a mixture of kB exponentials with rates = eigenvalues
#            of -Q0_BB, lagged by T.
#
# As T -> infinity the pulse becomes a simple step from zero concentration
# (§3(iv)): P(R>=1) -> 1 and the pdf reduces to ``ideal_pdf``.


def _sum_exp_components(left, M, right):
    """Spectral components of  f(t) = left · exp(M·t) · right.

    Writes the scalar function as a sum of exponentials

        f(t) = sum_i  areas[i] · eigs[i] · exp(−eigs[i]·t)

    where ``eigs`` are the eigenvalues of −M (positive reals for a proper
    sub-generator).  This generalises :func:`ideal_components`, which is the
    special case left = phi_shut, M = QFF, right = (−QFF)·u_F.

    Parameters
    ----------
    left : ndarray, shape (n,)
        Left (row) vector.
    M : ndarray, shape (n, n)
        Sub-generator with eigenvalues having negative real part.
    right : ndarray, shape (n,) or (n, 1)
        Right (column) vector.

    Returns
    -------
    eigs : ndarray, shape (n,)
        Eigenvalues of −M.
    areas : ndarray, shape (n,)
        Component areas; area[i] = (left · A_i · right) / eigs[i], where A_i are
        the spectral matrices of −M.
    """
    eigs, A = qml.eigs_sorted(-M)
    coeff = np.einsum('j,ijk,k->i', left, A, np.asarray(right).ravel())
    return eigs, coeff / eigs


def pulse_PR_ge_one(T, phi_shut, mec1, mec0):
    """Probability of at least one opening during/after a concentration pulse.

    Implements CHME97 Eq (3.10):

        P(R>=1) = phi_F · { (−Q1_FF)^-1 · [I − exp(Q1_FF·T)] · Q1_FA
                            + [exp(Q1_FF·T)]_·B · G0_BA } · u_A

    with G0_BA = (−Q0_BB)^-1 · Q0_BA (Eq 3.11).  The first term is the
    probability of opening during the pulse; the second is the probability of
    surviving in the shut set to time T in a within-burst state B and then
    opening after agonist removal.

    Parameters
    ----------
    T : float
        Pulse duration (s).
    phi_shut : ndarray, shape (kF,)
        Initial shut-state occupancies (ordered [B, C]); at a pulse from zero
        this is ``pinf(Q_at_c0)[kA:]``.
    mec1 : Mechanism
        Mechanism at the pulse concentration c1 (supplies Q1_FF, Q1_FA).
    mec0 : Mechanism
        Mechanism at zero concentration (supplies Q0_BB, Q0_BA).

    Returns
    -------
    PR : float
        P(R>=1), in (0, 1].
    """
    kB = mec1.kB
    uA = np.ones((mec1.kA, 1))
    expM = qml.expQt(mec1.QFF, T)
    I = np.eye(mec1.QFF.shape[0])
    GBA0 = np.linalg.inv(-mec0.QBB) @ mec0.QBA
    term_during = np.linalg.inv(-mec1.QFF) @ (I - expM) @ mec1.QFA
    term_after = expM[:, :kB] @ GBA0
    return float((phi_shut @ (term_during + term_after) @ uA).ravel()[0])


def ideal_pulse_components(T, phi_shut, mec1, mec0, PR=None):
    """Spectral components of the ideal first-latency pdf of a pulse.

    Returns the exponential components for the two time regimes separately.

    Parameters
    ----------
    T : float
        Pulse duration (s).
    phi_shut : ndarray, shape (kF,)
        Initial shut-state occupancies (ordered [B, C]).
    mec1, mec0 : Mechanism
        Mechanisms at the pulse concentration c1 and at zero concentration.
    PR : float, optional
        P(R>=1).  Computed via :func:`pulse_PR_ge_one` if not supplied.

    Returns
    -------
    eigs_during : ndarray, shape (kF,)
        Rates for t < T (eigenvalues of −Q1_FF).
    areas_during : ndarray, shape (kF,)
        Areas for t < T (already divided by PR).
    eigs_after : ndarray, shape (kB,)
        Rates for t > T (eigenvalues of −Q0_BB).
    areas_after : ndarray, shape (kB,)
        Areas for t > T (already divided by PR); the exponentials are lagged
        by T, i.e. evaluated at t − T.

    Notes
    -----
    The during-pulse components are those of :func:`ideal_components` rescaled
    by 1 / PR: by conservation Q1_FA·u_A = (−Q1_FF)·u_F, so the right vector
    Q1_FA·u_A is identical to the one used there.
    """
    if PR is None:
        PR = pulse_PR_ge_one(T, phi_shut, mec1, mec0)

    uA = np.ones((mec1.kA, 1))
    kB = mec1.kB

    eigs_during, areas_during = _sum_exp_components(
        phi_shut, mec1.QFF, mec1.QFA @ uA)
    areas_during = areas_during / PR

    # Occupancy of the B states at time T, having survived in F throughout [0, T)
    vB = (phi_shut @ qml.expQt(mec1.QFF, T))[:kB]
    eigs_after, areas_after = _sum_exp_components(vB, mec0.QBB, mec0.QBA @ uA)
    areas_after = areas_after / PR

    return eigs_during, areas_during, eigs_after, areas_after


def ideal_pulse_pdf(t, T, phi_shut, mec1, mec0, PR=None):
    """Evaluate the ideal first-latency pdf of a concentration pulse at time(s) t.

        f_FL(t) = phi_F · exp(Q1_FF·t) · Q1_FA · u_A / P(R>=1),         t < T
        f_FL(t) = [phi_F · exp(Q1_FF·T)]_B · exp(Q0_BB·(t−T)) · Q0_BA · u_A
                  / P(R>=1),                                            t >= T

    (CHME97 Eqs 3.6 and 3.8).  The pdf is in general discontinuous at t = T
    because the opening rate changes when agonist is removed.

    Parameters
    ----------
    t : float or ndarray
        Time (s).
    T : float
        Pulse duration (s).
    phi_shut : ndarray, shape (kF,)
        Initial shut-state occupancies (ordered [B, C]).
    mec1, mec0 : Mechanism
        Mechanisms at the pulse concentration c1 and at zero concentration.
    PR : float, optional
        P(R>=1).  Computed via :func:`pulse_PR_ge_one` if not supplied.

    Returns
    -------
    f : float (if t is scalar) or ndarray (if t is array)
        pdf value(s) in s⁻¹.
    """
    eigs_d, areas_d, eigs_a, areas_a = ideal_pulse_components(
        T, phi_shut, mec1, mec0, PR)
    tau_d = 1.0 / eigs_d
    tau_a = 1.0 / eigs_a

    scalar_input = np.ndim(t) == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    f = np.zeros(len(t_arr))

    during = t_arr < T
    if np.any(during):
        f[during] = pdfs.expPDF(t_arr[during], tau_d, areas_d)
    after = ~during
    if np.any(after):
        f[after] = pdfs.expPDF(t_arr[after] - T, tau_a, areas_a)

    if scalar_input:
        return float(f[0])
    return f


# ---------------------------------------------------------------------------
# Asymptotic (HJC)
# ---------------------------------------------------------------------------

def asymptotic_roots(tres, mec):
    """Find the kF roots of det[W_F(s)] = 0 for the shut-state asymptotic pdf.

    This is the shut-side analogue of the open-time asymptotic roots: the
    arguments to scl.asymptotic_roots are transposed so that QFF and QAA
    swap roles (F ↔ A convention, HJC92 §4).

    Parameters
    ----------
    tres : float
        Dead time / time resolution (s).
    mec : Mechanism
        Post-jump mechanism at concentration c1.

    Returns
    -------
    roots : ndarray, shape (kF,)
        Negative roots (all roots < 0).
    """
    return scl.asymptotic_roots(
        tres,
        mec.QFF, mec.QAA, mec.QFA, mec.QAF,
        mec.kF, mec.kA,
    )


def asymptotic_areas(tres, roots, phi_shut, mec):
    """Component areas for the first-latency asymptotic pdf.

    Uses phi_shut as the initial vector (not the HJC equilibrium phiA) and
    the shut-side AR matrices.  Vectorised form of the notebook helper
    `asymptotic_areas_first_latency`.

    area[i] = (−1/roots[i]) · phi_shut · R[i] · QFA · exp(QAA·tres) · u_A

    Parameters
    ----------
    tres : float
        Dead time (s).
    roots : ndarray, shape (kF,)
        Negative roots from asymptotic_roots().
    phi_shut : ndarray, shape (kF,)
        Initial shut-state occupancies.
    mec : Mechanism
        Post-jump mechanism.

    Returns
    -------
    areas : ndarray, shape (kF,)
    """
    expQAA = qml.expQt(mec.QAA, tres)
    # Shut-side AR: transpose QAA↔QFF and QFA↔QAF, also kA↔kF
    R = qml.AR(roots, tres, mec.QFF, mec.QAA, mec.QFA, mec.QAF, mec.kF, mec.kA)
    uA = np.ones((mec.kA, 1))
    # v = QFA @ expQAA @ uA, shape (kF,)
    v = (mec.QFA @ expQAA @ uA).ravel()
    areas = (-1.0 / roots) * np.einsum('j,ijk,k->i', phi_shut, R, v)
    return areas


def asymptotic_pdf(t, tres, tau, areas):
    """Evaluate the asymptotic first-latency pdf at time(s) t.

    The pdf is zero for t ≤ tres.  For t > tres it is an exponential mixture
    evaluated at the lag-corrected time t − tres:

        f(t) = 0                              for t ≤ tres
        f(t) = expPDF(t − tres, tau, areas)   for t > tres

    Handles both scalar and array t.

    Parameters
    ----------
    t : float or ndarray
        Time (s).
    tres : float
        Dead time (s).
    tau : ndarray, shape (kF,)
        Time constants = −1 / roots.
    areas : ndarray, shape (kF,)
        Component areas from asymptotic_areas().

    Returns
    -------
    f : float (if t is scalar) or ndarray (if t is array)
    """
    scalar_input = np.ndim(t) == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    f = np.zeros(len(t_arr))
    mask = t_arr > tres
    if np.any(mask):
        f[mask] = pdfs.expPDF(t_arr[mask] - tres, tau, areas)
    if scalar_input:
        return float(f[0])
    return f


# ---------------------------------------------------------------------------
# Gamma coefficients for exact pdf
# ---------------------------------------------------------------------------

def gamma_coefficients(tres, phi_shut, mec):
    """Gamma coefficients for the exact first-latency pdf.

    Computes g00, g10, g11 using the shut-side spectral decomposition of Q
    (Zxx with open=False, i.e. A↔F exchanged).

    Parameters
    ----------
    tres : float
        Dead time (s).
    phi_shut : ndarray, shape (kF,)
        Initial shut-state occupancies.
    mec : Mechanism
        Post-jump mechanism.

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of −Q (full matrix; one eigenvalue ≈ 0).
    g00, g10, g11 : ndarray, shape (k,)
        Gamma coefficients for the exact pdf (Eq. 3.22, HJC90).
        g00[i] = phi_shut · Z00[i] · u_A
        etc.
    """
    expQAA = qml.expQt(mec.QAA, tres)
    eigs, A = qml.eigs_sorted(-mec.Q)
    # Shut-side: exchange A↔F — pass QAA as "QFF" arg, QFA as "QAF" arg, etc.
    eigen, Z00, Z10, Z11 = qml.Zxx(
        mec.Q, eigs, A, mec.kA,
        mec.QAA, mec.QFA, mec.QAF, expQAA, False,
    )
    uA = np.ones((mec.kA, 1))
    g00 = (phi_shut @ Z00 @ uA).T[0]
    g10 = (phi_shut @ Z10 @ uA).T[0]
    g11 = (phi_shut @ Z11 @ uA).T[0]
    return eigen, g00, g10, g11


# ---------------------------------------------------------------------------
# Exact pdf (HJC)
# ---------------------------------------------------------------------------

def exact_pdf(t, tres, roots, areas, eigvals, g00, g10, g11):
    r"""Evaluate the exact first-latency pdf at time(s) t.

    Applies the HJC exact correction for tres ≤ t < 3·tres and reverts to
    the asymptotic form for t ≥ 3·tres (Eq. 21, HJC92):

        f(t) = 0                                      for t < tres
        f(t) = f0(t − tres)                           for tres ≤ t < 2·tres
        f(t) = f0(t − tres) − f1(t − 2·tres)         for 2·tres ≤ t < 3·tres
        f(t) = expPDF(t − tres, −1/roots, areas)      for t ≥ 3·tres

    Handles both scalar and array t.  (Fixes the ``AttributeError: 'float'
    object has no attribute 'shape'`` bug in scl.exact_pdf when t is scalar.)

    Parameters
    ----------
    t : float or ndarray
        Time (s).
    tres : float
        Dead time (s).
    roots : ndarray, shape (kF,)
        Negative roots from asymptotic_roots().
    areas : ndarray, shape (kF,)
        Component areas from asymptotic_areas().
    eigvals : ndarray, shape (k,)
        Eigenvalues of −Q from gamma_coefficients().
    g00, g10, g11 : ndarray, shape (k,)
        Gamma coefficients from gamma_coefficients().

    Returns
    -------
    f : float (if t is scalar) or ndarray (if t is array)
    """
    scalar_input = np.ndim(t) == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    f = np.zeros(len(t_arr))
    tau = -1.0 / roots

    # Asymptotic branch (t >= 3*tres): fully vectorised — one expPDF call
    # for all qualifying points rather than one per element.
    asym_mask = t_arr >= 3.0 * tres
    if np.any(asym_mask):
        f[asym_mask] = pdfs.expPDF(t_arr[asym_mask] - tres, tau, areas)

    # Exact correction branches (tres <= t < 3*tres): scalar loop, but
    # there are at most a handful of points here in practice.
    for idx in np.where((t_arr >= tres) & ~asym_mask)[0]:
        ti = t_arr[idx]
        if ti < 2.0 * tres:
            f[idx] = float(qml.f0(ti - tres, eigvals, g00))
        else:
            f[idx] = float(qml.f0(ti - tres, eigvals, g00) -
                           qml.f1(ti - 2.0 * tres, eigvals, g10, g11))

    if scalar_input:
        return float(f[0])
    return f


# ---------------------------------------------------------------------------
# Apparent (missed-events) first-latency pdf of a concentration PULSE
# CHME97 §4(a), Eqs 4.1-4.3  (regimes 1 and 2; the after-pulse tail Eq 4.4
# requires the zero-concentration reducible survivor and is not yet provided)
# ---------------------------------------------------------------------------
#
# With a finite dead time tres an opening is only detectable if the channel
# stays in the open set A for at least tres.  The apparent first latency is the
# time to the first such detectable opening.  Following the convention of the
# step module (asymptotic_pdf / exact_pdf), the latency t is measured to the
# instant of detection (the opening transition at tau = t − tres, plus tres),
# so the density is zero for t <= tres.
#
# The calculation rests on the shut-time survivor matrix (Appendix A, CHME97)
#
#     TR(u)_ij = P[X(u)=j and no detectable opening over (0,u) | X(0)=i],
#
# for shut states i, j ∈ F.  TR is exact for u < 2·tres (HJC90 C-matrices via
# f0/f1) and asymptotic beyond.  In terms of TR the apparent first-latency
# density of a pulse (c = 0 → c1 for duration T → 0) is, for t < T + tres,
#
#   regime 1  (tres < t < T):       Eq 4.2 — opening detected within the pulse
#       phi · TR1(t−tres) · Q1_FA · exp(Q1_AA·tres) · u_A
#       (identical to the step apparent first latency, i.e. exact_pdf).
#
#   regime 2  (T <= t < T + tres):  Eq 4.3 — confirming sojourn straddles T
#       phi · TR1(tau) · Q1_FA · exp(Q1_AA·(T−tau)) · exp(Q0_AA·(tau+tres−T)) · u_A
#
# with tau = t − tres.  The two regimes meet continuously at t = T.  The
# returned density is UNCONDITIONAL (not divided by P(R>=1)); for t >= T + tres
# a NotImplementedError is raised (Eq 4.4 needs TR at zero concentration, where
# the absorbing set C makes Q reducible and the asymptotic roots degenerate).


def _asymptotic_R(roots, tres, QAA, QFF, QAF, QFA, kA, kF):
    """Asymptotic residue matrices of the survivor — robust (SVD) form of qml.AR.

    Identical to :func:`qmatlib.AR` except the left/right null vectors of
    ``W(s)`` are obtained by SVD instead of ``pinf``.  ``pinf`` finds a null
    vector by assuming its argument is a generator (rows summing to zero); that
    holds for the full reversible chain but NOT for the reduced {A,B} sub-system
    used after the pulse (CHME97 Eq 4.4), where B leaks to the absorbing set C.
    There ``pinf`` returns a wrong right null vector, the residue denominator
    collapses to ~1e-16 and ``R`` explodes (~1e15).  The residue
    R_i = c_i r_i / (r_i W'(s_i) c_i) is invariant to the scaling of c_i, r_i,
    so the SVD form reproduces :func:`qmatlib.AR` to ~1e-13 for conservative
    systems while staying well-conditioned for the leaky one.

    Same positional arguments and return shape as :func:`qmatlib.AR`.
    """
    R = np.zeros((kA, kA, kA))
    for i in range(kA):
        WA = qml.W(roots[i], tres, QAA, QFF, QAF, QFA, kA, kF)
        U, _, Vh = np.linalg.svd(WA)
        r = U[:, -1]            # left null vector:  r · WA ≈ 0
        c = Vh[-1, :]           # right null vector: WA · c ≈ 0
        W1 = qml.dW(roots[i], tres, QAF, QFF, QFA, kA, kF)
        R[i] = np.outer(c, r) / (r @ W1 @ c)
    return R


def shut_survivor_components(tres, mec):
    """Pre-compute the pieces of the shut-time survivor matrix TR(t).

    Bundles the exact C-matrices (valid for t < 2·tres) and the asymptotic
    roots/AR matrices (t >= 2·tres) so that :func:`shut_survivor` can be
    evaluated cheaply at many time points.

    Parameters
    ----------
    tres : float
        Dead time (s).
    mec : Mechanism
        Mechanism at the relevant concentration.

    Returns
    -------
    components : dict
        Keys: ``tres``, ``kF``, ``eigvals``, ``C00``, ``C10``, ``C11``
        (exact, shape (k, kF, kF)) and ``roots``, ``R`` (asymptotic).
    """
    eigs, A = qml.eigs_sorted(-mec.Q)
    expQAA = qml.expQt(mec.QAA, tres)
    # Shut side: exchange A↔F (open=False) — pass QAA as "QFF" arg etc.
    eigvals, C00, C10, C11 = qml.Cxx(
        mec.Q, eigs, A, mec.kA,
        mec.QAA, mec.QFA, mec.QAF, expQAA, False,
    )
    roots = asymptotic_roots(tres, mec)
    R = _asymptotic_R(roots, tres, mec.QFF, mec.QAA, mec.QFA, mec.QAF,
                      mec.kF, mec.kA)
    return {
        'tres': tres, 'kF': mec.kF,
        'eigvals': eigvals, 'C00': C00, 'C10': C10, 'C11': C11,
        'roots': roots, 'R': R,
    }


def shut_survivor(t, tres, components):
    r"""Evaluate the HJC shut-time survivor matrix TR(t) (Appendix A, CHME97).

        TR(t)_ij = P[X(t)=j and no detectable opening over (0,t) | X(0)=i]

    for shut states i, j ∈ F.  Exact for t < 2·tres, asymptotic beyond:

        t <= 0          : I  (identity)
        0 < t < tres    : f0(t, eigvals, C00)            = [exp(Q·t)]_FF
        tres <= t < 2tres: f0(t, .) − f1(t−tres, ., C10, C11)
        t >= 2·tres     : Σ_i R[i] · exp(roots[i]·t)

    Parameters
    ----------
    t : float
        Time (s).
    tres : float
        Dead time (s).
    components : dict
        Output of :func:`shut_survivor_components`.

    Returns
    -------
    TR : ndarray, shape (kF, kF)
    """
    if t <= 0.0:
        return np.eye(components['kF'])
    eigvals = components['eigvals']
    if t < tres:
        return qml.f0(t, eigvals, components['C00'])
    if t < 2.0 * tres:
        return (qml.f0(t, eigvals, components['C00']) -
                qml.f1(t - tres, eigvals, components['C10'], components['C11']))
    R = components['R']
    roots = components['roots']
    return np.sum(R * np.exp(roots * t).reshape(R.shape[0], 1, 1), axis=0)


def _reduced_burst_mechanism(mec0):
    """Lightweight {A, B} sub-mechanism at zero concentration (for Eq 4.4).

    The after-pulse survivor TR0(t) needs only its B-block ``R_BB(t)``: at zero
    concentration C is absorbing and cannot open (Q0_CA = 0), so the C-block and
    the B→C cross-block are annihilated by Q0_FA in every product of Eq 4.4.
    ``R_BB(t)`` is the shut-time survivor of the reduced system whose open set is
    A and whose (reduced) shut set is B.  Crucially Q0_BB carries the B→C leak in
    its diagonal, so it is *non-singular* — the asymptotic root-finder works,
    unlike the full reducible Q0_FF whose absorbing C gives a zero eigenvalue.

    Returns a duck-typed namespace exposing the attributes that
    :func:`shut_survivor_components` and :func:`asymptotic_roots` require.
    """
    red = SimpleNamespace()
    red.Q = mec0.QEE          # generator over [A, B]
    red.kA = mec0.kA
    red.kF = mec0.kB          # B is the reduced shut set
    red.k = mec0.kE
    red.QAA = mec0.QAA
    red.QFF = mec0.QBB        # carries the B→C leak (non-singular)
    red.QFA = mec0.QBA
    red.QAF = mec0.QAB
    return red


def _exp_spectral(eig, A, t):
    """exp(M·t) from a pre-computed spectral decomposition (eig, A) of M.

    Avoids re-doing an eigendecomposition per call (unlike ``qml.expQt``),
    which matters because the Eq-4.4 double integral evaluates exp(Q0_AA·r)
    O(nquad²) times per time point.
    """
    return np.sum(A * np.exp(eig * t).reshape(-1, 1, 1), axis=0)


def _regime3_context(T, tres, mec1, mec0, comp1, nquad, uA):
    """Pre-compute the tau-independent pieces of the Eq-4.4 after-pulse term."""
    comp_red = shut_survivor_components(tres, _reduced_burst_mechanism(mec0))
    eQ0 = qml.expQt(mec0.QAA, tres)                # exp(Q0_AA · tres)
    G = mec0.QBA @ eQ0                             # R_BB(·) · Q0_BA · exp(Q0_AA·tres)
    # Pre-computed spectral decomposition of Q0_AA for cheap exp(Q0_AA·r1).
    e0A, A0A = qml.eigs(mec0.QAA)
    r0 = np.linspace(0.0, tres, nquad + 1)
    w0 = np.full(nquad + 1, tres / nquad)
    w0[0] *= 0.5
    w0[-1] *= 0.5
    # Outer(r0) = TR1(T - r0) · Q1_FA · exp(Q1_AA · r0)  (independent of tau)
    outer = [shut_survivor(T - r0v, tres, comp1) @ mec1.QFA @ qml.expQt(mec1.QAA, r0v)
             for r0v in r0]
    return {
        'comp_red': comp_red, 'G': G, 'e0A': e0A, 'A0A': A0A,
        'r0': r0, 'w0': w0, 'outer': outer,
        'TR1_T_B': shut_survivor(T, tres, comp1)[:, :mec1.kB],
        'kB': mec1.kB, 'nquad': nquad, 'uA': uA,
    }


def _regime3_value(tau, T, tres, phi_shut, mec1, mec0, ctx):
    """Density for the after-pulse regime (CHME97 Eq 4.4), tau = t - tres >= T.

    phi · {  TR1(T)_·B · R_BB(tau-T) · Q0_BA · exp(Q0_AA·tres)            (term A)
           + ∫₀^tres dr0 Outer(r0) · ∫₀^u1 dr1 exp(Q0_AA·r1)·Q0_AB
               · R_BB(tau-T-r1) · Q0_BA · exp(Q0_AA·tres)  } · u_A         (term B)
    with u1 = min(tres - r0, tau - T).  Only R_BB (reduced B-survivor) appears.
    """
    comp_red = ctx['comp_red']
    G = ctx['G']
    e0A, A0A = ctx['e0A'], ctx['A0A']
    Q0AB = mec0.QAB

    # term A
    M = ctx['TR1_T_B'] @ shut_survivor(tau - T, tres, comp_red) @ G

    # term B — double trapezium over [0, tres] × [0, u1]
    nq = ctx['nquad']
    for w0v, r0v, outer in zip(ctx['w0'], ctx['r0'], ctx['outer']):
        u1 = min(tres - r0v, tau - T)
        if u1 <= 0.0:
            continue
        r1 = np.linspace(0.0, u1, nq + 1)
        w1 = np.full(nq + 1, u1 / nq)
        w1[0] *= 0.5
        w1[-1] *= 0.5
        inner = np.zeros((mec1.kA, mec1.kA))
        for w1v, r1v in zip(w1, r1):
            inner += w1v * (_exp_spectral(e0A, A0A, r1v) @ Q0AB @
                            shut_survivor(tau - T - r1v, tres, comp_red) @ G)
        M = M + w0v * (outer @ inner)

    return float((phi_shut @ M @ ctx['uA']).ravel()[0])


def apparent_pulse_pdf(t, T, tres, phi_shut, mec1, mec0, components=None,
                       nquad=24):
    """Apparent (missed-events) first-latency density of a concentration pulse.

    Evaluates the unconditional apparent first-latency density (CHME97 §4(a),
    Eqs 4.2-4.4) at time(s) *t*, dispatching on the three regimes:

        tres < t < T          regime 1 (Eq 4.2): opening detected within pulse.
                              Identical to the step apparent pdf (exact_pdf).
        T <= t < T + tres     regime 2 (Eq 4.3): confirming sojourn straddles T.
        t >= T + tres         regime 3 (Eq 4.4): opening transition after the
                              pulse; double-integral term evaluated by the
                              trapezium rule with *nquad* points per axis.

    The latency *t* is measured to the instant of detection (transition + tres),
    so the density is zero for t <= tres.  The returned density is UNCONDITIONAL
    (not divided by P(R>=1)); use :func:`apparent_pulse_PR_ge_one` to normalise.

    The pulse must be longer than the dead time (T > tres), as assumed in CHME97.

    Parameters
    ----------
    t : float or ndarray
        Time (s).
    T : float
        Pulse duration (s).  Must exceed *tres*.
    tres : float
        Dead time (s).
    phi_shut : ndarray, shape (kF,)
        Initial shut-state occupancies (ordered [B, C]).
    mec1 : Mechanism
        Mechanism at the pulse concentration c1.
    mec0 : Mechanism
        Mechanism at zero concentration.
    components : dict, optional
        Output of ``shut_survivor_components(tres, mec1)``; computed if omitted.
    nquad : int
        Trapezium points per axis for the Eq-4.4 double integral (regime 3).

    Returns
    -------
    f : float (if t is scalar) or ndarray (if t is array)
        Unconditional density (s⁻¹).
    """
    if T <= tres:
        raise ValueError(
            f"apparent_pulse_pdf requires T > tres (got T={T}, tres={tres}); "
            "CHME97 §4 assumes the pulse is longer than the dead time.")
    if components is None:
        components = shut_survivor_components(tres, mec1)
    uA = np.ones((mec1.kA, 1))
    expA1 = qml.expQt(mec1.QAA, tres)
    ctx = {}   # regime-3 context, built lazily and reused across array points

    def _one(ti):
        if ti <= tres:
            return 0.0
        tau = ti - tres
        if ti < T:                                   # regime 1 (Eq 4.2)
            M = shut_survivor(tau, tres, components) @ mec1.QFA @ expA1
            return float((phi_shut @ M @ uA).ravel()[0])
        if ti < T + tres:                            # regime 2 (Eq 4.3)
            M = (shut_survivor(tau, tres, components) @ mec1.QFA @
                 qml.expQt(mec1.QAA, T - tau) @
                 qml.expQt(mec0.QAA, tau + tres - T))
            return float((phi_shut @ M @ uA).ravel()[0])
        if not ctx:                                  # regime 3 (Eq 4.4)
            ctx.update(_regime3_context(T, tres, mec1, mec0, components, nquad, uA))
        return _regime3_value(tau, T, tres, phi_shut, mec1, mec0, ctx)

    scalar_input = np.ndim(t) == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    f = np.array([_one(ti) for ti in t_arr])
    if scalar_input:
        return float(f[0])
    return f


def apparent_pulse_PR_ge_one(T, tres, phi_shut, mec1, mec0, nquad=24,
                             upper=None):
    """Probability of at least one apparent opening after a concentration pulse.

    Computes P(R>=1) by integrating the unconditional apparent first-latency
    density (:func:`apparent_pulse_pdf`) over [tres, infinity).  This sidesteps
    CHME97 Eq 4.7 (which needs the same zero-concentration survivor) and is the
    robust route used for the ideal pulse in Phase 1.  Dividing
    ``apparent_pulse_pdf`` by this value yields the conditional pdf, which
    integrates to 1.

    Parameters
    ----------
    T, tres, phi_shut, mec1, mec0, nquad
        As for :func:`apparent_pulse_pdf`.
    upper : float, optional
        Upper integration limit (s).  Defaults to T + 20·(slowest after-pulse
        time constant), i.e. well into the tail.

    Returns
    -------
    PR : float
        P(R>=1), in (0, 1].
    """
    from scipy.integrate import quad

    components = shut_survivor_components(tres, mec1)
    if upper is None:
        red = _reduced_burst_mechanism(mec0)
        slowest = 1.0 / np.min(np.abs(asymptotic_roots(tres, red)))
        upper = T + tres + 20.0 * slowest

    def f(ti):
        return apparent_pulse_pdf(ti, T, tres, phi_shut, mec1, mec0,
                                  components=components, nquad=nquad)

    I1, _ = quad(f, tres, T, limit=100)
    I2, _ = quad(f, T, T + tres, limit=50)
    I3, _ = quad(f, T + tres, upper, limit=300)
    return I1 + I2 + I3
