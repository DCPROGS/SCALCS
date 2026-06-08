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
