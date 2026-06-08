# qmatlib — Q-matrix algebra

`scalcs.qmatlib`

Low-level Q-matrix operations used throughout SCALCS: eigendecomposition,
matrix exponentials, equilibrium occupancies, HJC matrices, and the exact-pdf
γ coefficients.

---

## Eigendecomposition

### `eigs(Q)`

Compute eigenvalues and spectral matrices of Q.

```
eigvals, A = eigs(Q)
```

Returns eigenvalues `eigvals` (shape `(k,)`) and spectral matrices `A`
(shape `(k, k, k)`) such that `Q = sum_i eigvals[i] * A[i]` and
`exp(Q·t) = sum_i exp(eigvals[i]·t) * A[i]`.

### `eigs_sorted(Q)`

Same as `eigs` but eigenvalues (and spectral matrices) are sorted by
ascending real part.  Used when a consistent ordering is required across
calls (e.g. for the exact pdf γ coefficients).

---

## Matrix exponential

### `expQt(M, t)`

Matrix exponential `exp(M·t)` via spectral decomposition.

```python
expQAA = expQt(mec.QAA, tres)   # shape (kA, kA)
```

More numerically stable than `scipy.linalg.expm` for the rate matrices
encountered in ion channel kinetics.

### `Qpow(M, n)`

Integer matrix power `M^n`.  Used in some burst-length calculations.

---

## Equilibrium occupancies

### `pinf(Q)`

Equilibrium occupancy vector π satisfying `π Q = 0`, `Σ πᵢ = 1`.

```python
p = pinf(mec.Q)          # full occupancy vector, shape (k,)
phi_shut = p[mec.kA:]    # shut-state occupancies for first-latency
```

### `pinf1(Q)`

Alternative implementation using the null-space of Q.  Prefer `pinf` for
general use.

---

## Steady-state jump matrices (iGs)

### `iGs(Q, kA, kB)`

Compute the idealised generator matrices G_AB and G_BA:

```
G_AB = −Q_AA⁻¹ · Q_AB
G_BA = −Q_BB⁻¹ · Q_BA
```

Used to compute burst start/end probabilities and HJC equilibrium vectors.

### `iGt(t, QAA, QAB)`

Time-dependent G matrix: `G(t) = exp(Q_AA · t) · (−Q_AA⁻¹) · Q_AB`.

### `eGs(GAF, GFA, kA, kF, expQFF)`

HJC time-averaged jump matrix:

```
eG_AF(t) = G_AF · exp(Q_FF · t)
```

---

## Initial vectors

| Function | Description |
|----------|-------------|
| `phiA(mec)` | HJC equilibrium open-state vector φ_A |
| `phiF(mec)` | HJC equilibrium shut-state vector φ_F |
| `phiSub(Q, k1, k2)` | Occupancy within a subspace (rows k1 to k2) |
| `phiHJC(eGAF, eGFA, kA)` | Full HJC equilibrium vector from jump matrices |

---

## HJC Laplace-domain matrices

These are the building blocks of the asymptotic root-finding algorithm
(HJC92 §4).

### `H(s, tres, QAA, QFF, QAF, QFA, kF)`

HJC matrix H(s):

```
H(s) = Q_FF + Q_FA · (s·I − Q_AA)⁻¹ · Q_AF · exp((Q_FF − s·I)·tres)
```

Shape `(kF, kF)`.  Its eigenvalues equal s at the asymptotic roots.

### `W(s, tres, QAA, QFF, QAF, QFA, kA, kF)`

```
W(s) = s·I − H(s)
```

The asymptotic roots are zeros of `det[W(s)]`.

### `detW(s, tres, QAA, QFF, QAF, QFA, kA, kF)`

`det[W(s)]` as a scalar.  Passed to `scipy.optimize.brentq` during root
refinement.

### `dW(s, tres, QAF, QFF, QFA, kA, kF)`

Derivative `d/ds det[W(s)]`.  Used in some root-refinement schemes.

---

## Residue matrices for asymptotic pdf

### `AR(roots, tres, QAA, QFF, QAF, QFA, kA, kF)`

Residue matrices R_i at each root s_i:

```
R_i = lim_{s→s_i} (s − s_i) · W(s)⁻¹
```

Shape `(kA, kA, kA)`.  Used by `scalcslib.asymptotic_areas` and
`firstlatency.asymptotic_areas`.

### `dARSdS(tres, QAA, QFF, GAF, GFA, expQFF, kA, kF)`

Derivative of the AR matrices with respect to s.  Used in likelihood
gradient calculations.

---

## Exact pdf γ coefficients

### `Zxx(Q, eigen, A, kopen, QFF, QAF, QFA, expQFF, open)`

Compute Z₀₀, Z₁₀, Z₁₁ matrices for the exact pdf (HJC90 Eq. 3.22).

```python
eigen, Z00, Z10, Z11 = Zxx(
    mec.Q, eigs, A, mec.kA,
    mec.QFF, mec.QAF, mec.QFA, expQFF, True,
)
```

Set `open=False` to use the shut-side (F) convention required by
`firstlatency.gamma_coefficients`.

### `f0(u, eigvals, Z00)`

First term of the exact pdf correction:

```
f0(u) = Σ_i Z00[i] · exp(−eigvals[i] · u)
```

### `f1(u, eigvals, Z10, Z11)`

Second term:

```
f1(u) = Σ_i (Z10[i] + Z11[i] · u) · exp(−eigvals[i] · u)
```

### `eGAF(t, tres, eigvals, Z00, Z10, Z11, roots, R, QAF, expQFF)`

Exact time-dependent jump matrix `eG_AF(t)` (HJC92 Eq. 2.2).  Used in the
exact open-time pdf and in CHS joint distributions.

---

## CHS (Colquhoun–Hawkes–Srodzinski) vectors

### `HAF(roots, tres, tcrit, QAF, expQFF, R)`

HAF matrix for CHS joint distributions (CHS96).

### `CHSvec(roots, tres, tcrit, QFA, kA, expQAA, phiF, R)`

CHS start vector for the joint open/shut time distribution.

---

## Reference

- HJC92: Hawkes, Jalali & Colquhoun (1992)
- CHS96: Colquhoun, Hawkes & Srodzinski (1996)
