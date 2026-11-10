## 1. Read the same fluctuations in two coordinate systems

Imagine a row of height sensors on a flexible surface. A snapshot gives one value at each position. Across many equilibrium snapshots, nearby sensors may fluctuate together. A covariance matrix records those relationships in position coordinates; a spatial spectrum describes how strongly broad and rapidly alternating patterns fluctuate.

These are two views of the same ensemble. Positions are not time steps, and a spatial spectrum is not a temporal frequency spectrum. We will connect both views to a third question: how far does a disturbance remain correlated in an infinite system?

Prerequisites are the Gaussian-field chapter's covariance and constrained ensembles, and the field-mode chapter's boundary conditions and Fourier basis. Here we add physical spacing, finite-size effects and a controlled comparison with a continuum model.

## 2. Build the energy before drawing a curve

Let φ be a real field on n sites separated by a positive distance a. Set Boltzmann's constant to one and use equilibrium weight exp(−U/T), with T>0 and zero applied source:

$$
U(\phi)=\frac{a m^2}{2}\sum_j\phi_j^2
+\frac{c^2}{2a}\sum_{(i,j)\in E}(\phi_j-\phi_i)^2,
\qquad K=M I+\kappa L,\quad M=a m^2,\quad\kappa=c^2/a.
$$

The supplied coefficients m² and c² are already squared quantities; do not square the inputs again. The first term discourages large heights. The second discourages roughness. The factor a comes from a spatial integral, while a difference quotient contributes 1/a² to the gradient term. This explains why changing spacing changes both coefficients in K.

For a full positive-definite model, the mean is zero and C=⟨φφᵀ⟩=T K⁻¹. C is a covariance with field-squared units, not a Pearson coefficient. A large C₀₀ alone says nothing about the distance over which correlations persist.

## 3. Boundary conditions change the experiment

| Boundary | Edges and exterior | Site positions | Box length |
|---|---|---|---|
| Periodic | Chain plus last-to-first edge | j a | n a |
| Open | Only n−1 chain edges | (j+1/2)a | n a |
| Fixed zero | Open chain plus two exterior zero pins | (j+1)a | (n+1)a |

For a mental picture, sketch a loop, a row with free ends, and a row whose two exterior endpoints are pinned. The pinned endpoints are not fluctuating integration variables. An open Laplacian has endpoint diagonal 1; the other two cases have diagonal 2. Neighbor entries are −1, including the wraparound pair only on the ring.

On a ring every origin is equivalent. On an open or pinned chain, moving the origin toward an endpoint can change the covariance profile. A single distance curve then cannot represent every matrix row.

## 4. Remove a zero mode only by declaring the ensemble

At m²=0, a constant field costs no energy on an open chain or ring. Integrating that direction gives an infinite normalization. A fixed-zero exterior removes this freedom when c²>0. Another valid model restricts every configuration to sum zero.

For the latter, write φ=Bz, where the columns of B are eᵢ−eₙ₋₁. Then

$$
G=B(B^TKB)^{-1}B^T,\qquad C=T G,
\qquad P=I-\frac{\mathbf1\mathbf1^T}{n},\qquad PC=C,\quad PKC=TP.
$$

This is a different ensemble, not a tiny diagonal regularizer. With no constraint, use B=P=I. In either case the reduced precision must be positive definite. If both coefficients vanish, even the constrained model is rejected. Positive rational LDLᵀ pivots and exact projected residuals certify the finite algebra.

## 5. Keep the Green function's spacing factor

The discrete differential operator approximating m²−c²∂² is A=K/a. On the full space its covariance equation is

$$
A C=\frac{T}{a}I.
$$

The factor 1/a represents a unit-integral spatial delta on a grid. Omitting it changes the source normalization. An energy term −Jᵀφ gives susceptibility C/T, whereas a source density coupled through −a hᵀφ gives susceptibility a C/T. On a constrained space the corresponding statements use projection.

Our calculation keeps the source zero. These response identities explain its normalization; they do not introduce an unimplemented source control or a causal inference from observed correlations.

## 6. Transform the whole covariance

Choose the unitary discrete Fourier transform

$$
F_{\ell j}=\frac{1}{\sqrt n}e^{-2\pi i\ell j/n},\qquad
\widetilde C=FCF^*,\qquad \widetilde K=FKF^*.
$$

A low spatial wave changes slowly across the sites; a high wave alternates more rapidly. Fourier coordinates form a complete basis for every boundary condition. They form an eigenbasis of this precision on a periodic ring, but generally not on open or pinned chains. Off-diagonal entries of C̃ measure covariance between Fourier coordinates; they must remain visible in those cases.

Because C is real symmetric, C̃ is Hermitian. Its diagonal is real and nonnegative. Trace is basis-independent: the sum of Fourier powers equals the sum of site variances. These facts provide checks beyond making a visually plausible spectrum.

## 7. Derive the periodic spectrum

Applying the ring Laplacian to a Fourier wave gives 2−2cos(2πℓ/n)=4sin²(πℓ/n). Therefore

$$
\lambda_\ell=a m^2+\frac{4c^2}{a}\sin^2(\pi\ell/n),
\qquad S_\ell=\widetilde C_{\ell\ell}=\frac{T}{\lambda_\ell}.
$$

In the mean-zero ensemble the constant Fourier coordinate is absent, so S₀=0 rather than T/λ₀. Nonconstant periodic modes retain their full-space values. A soft mass makes the unconstrained constant-mode variance large. Increasing c² suppresses nonconstant modes more strongly, while leaving the constant mode unchanged at fixed a,m²,T.

For nonperiodic boundaries, calculate FCF* directly. Applying the periodic formula there discards boundary-induced mixing and gives the wrong ensemble.

## 8. A four-site calculation you can check by hand

Set n=4 and a=m²=c²=T=1, with a periodic ring and no constraint. The precision has diagonal 3 and neighbor entries −1. Its four Fourier eigenvalues are 1,3,5,3. Inverting those eigenvalues gives

$$
C_{0,:}=\left(\frac7{15},\frac15,\frac2{15},\frac15\right),
\qquad S=\left(1,\frac13,\frac15,\frac13\right).
$$

The first inverse-DFT entry is (1+1/3+1/5+1/3)/4=7/15. The opposite site has 2/15, while either neighbor has 1/5. The curve returns upward after the opposite site because distance is measured forward around the ring.

The trace is 4·7/15=28/15, equal to the sum of the four powers. Under the mean-zero constraint, subtract 1/4 from each covariance entry and remove S₀. Neighbor covariance becomes −1/20 despite positive nearest-neighbor coupling. This is a constraint effect, not evidence of an antiferromagnetic interaction.

## 9. Distinguish a mode index from a physical wavenumber

For array index ℓ, use signed integer q=ℓ when 2ℓ≤n and q=ℓ−n otherwise. The physical wavenumber is k=2πq/(n a). Conjugate partners have indices ℓ and (n−ℓ) modulo n. For even n we label the Nyquist mode with positive q=n/2; it is its own partner.

For n=8, index 7 means q=−1, not a physical wave seven times faster than index 1. If the ring is refined to n=16 at fixed length, the same negative wave moves to index 15. Keeping index 7 would compare different physical waves.

These labels describe spatial phase variation. Temperature rescales the covariance, not the physical wavenumber or the geometry.

## 10. Compare spectra with compatible normalization

Unitary Fourier coordinates have raw power Sℓ. The quantity compatible with the continuum operator is a Sℓ:

$$
a S_\ell=\frac{T}{m^2+(4c^2/a^2)\sin^2(ka/2)}
\longrightarrow\frac{T}{m^2+c^2 k^2}\quad(ka\to0).
$$

For independent sites with m²=T=1, a=1/2 gives K=I/2, hence Sℓ=2 but a Sℓ=1. A doubled raw power here is a normalization change, not a doubled continuum fluctuation spectrum.

The approximation is for a fixed physical low wave as spacing decreases. Modes near Nyquist do not satisfy small ka merely because there are more sites. Nor does increasing n at fixed a perform a continuum limit: it enlarges the box.

## 11. Obtain decay from the infinite-chain equation

For m²,c²>0, translation invariance on an infinite chain gives a recurrence away from the source. Try C∞(d)=A∞r^|d| with 0<r<1. The recurrence and source jump yield

$$
u=\frac{a^2m^2}{c^2},\qquad
r=\frac{2}{2+u+\sqrt{u(u+4)}},\qquad
A_\infty=\frac{T}{\sqrt{M(M+4\kappa)}}.
$$

The root satisfies κr²−(M+2κ)r+κ=0. At the origin, (M+2κ)A∞−2κA∞r=T fixes the amplitude. The reciprocal expression for r avoids subtracting almost equal positive numbers when the mass is small.

The spatial decay length is

$$
\xi=-\frac{a}{\log r}
=\frac{a}{\operatorname{acosh}(1+u/2)},\qquad
\xi_{\mathrm{cont}}=\sqrt{\frac{c^2}{m^2}}.
$$

For a=m²=c²=1, ξ is approximately 1.039, while the continuum approximation is 1. Temperature changes A∞ but not either length when the coefficients are held fixed. This model does not include temperature-dependent material coefficients.

## 12. A finite ring sees periodic images

Summing the infinite covariance over all periodic images gives the unconstrained finite-ring result

$$
C_n(d)=A_\infty\frac{r^d+r^{n-d}}{1-r^n},\qquad d=0,\ldots,n-1.
$$

There are two routes around a ring. Near the opposite side they contribute comparably, so the curve is not a single decaying exponential. For massive mean-zero rings, subtract T/(nM), the contribution of the removed constant mode.

The image expression is specific to periodic geometry. It is not the covariance of an open or fixed-zero chain. Compare it with the exact finite matrix before interpreting a curve as an infinite-system result. A larger ring approximates a local infinite profile only when its circumference is large relative to the decay length and the chosen separations avoid wraparound effects.

## 13. Massless finite fields can exist without an infinite stationary field

For m²=0,c²>0 on a mean-zero ring, the exact finite covariance is

$$
C_n(d)=\frac{T}{\kappa}
\left[\frac{n^2-1}{12n}-\frac{d(n-d)}{2n}\right].
$$

With n=8,a=c²=T=1, the first entries are 21/32,7/32,−3/32. Positive short-range covariance crosses zero because every row sums to zero. The variance grows with box size, so this family does not converge to a stationary infinite field with finite point variance.

A pinned massless chain also has a finite covariance. Neither finite existence establishes an infinite stationary covariance. We therefore leave the infinite covariance and finite decay length undefined in this case. A divergent correlation length in a Gaussian limit also does not establish the critical exponents of a general interacting system.

## 14. Zero coupling does not remove a global constraint

With c²=0,m²>0 on the full space, C=(T/M)I. Distinct sites are independent and the infinite reference has ξ=0. Under the mean-zero constraint,

$$
C=\frac{T}{M}\left(I-\frac{\mathbf1\mathbf1^T}{n}\right).
$$

For n=8,M=T=1, diagonal entries are 7/8 and off-diagonal entries are −1/8. If one site rises, the others must collectively compensate. The interaction graph has no edges with positive weight, yet the ensemble is dependent.

This is a useful warning when centering training data or enforcing conservation: preprocessing and constraints can create correlations. On a pinned massive chain, subtracting each sample's mean generally differs from conditioning on a zero sum because its covariance does not have uniform row sums.

## 15. A local ratio is a diagnostic, not a definition

For two positive, strictly decreasing adjacent covariance values x,y, one can form a local estimate a/log(x/y). It equals ξ for an exact single exponential at that spacing. Periodic images and constant-mode subtraction violate that assumption.

The massless eight-site example gives a positive first estimate 1/log(3), despite having no infinite stationary field covariance. Later entries change sign, making the logarithm inadmissible. Equal or increasing positive values also do not define a positive decay estimate by this rule.

For periodic profiles read sites (origin+d) modulo n at forward distance d a. For nonperiodic profiles retain signed displacement (j−origin)a and the actual origin. Their ratios are classified as boundary-nonuniform instead of being promoted to an infinite decay length.

## 16. Refine the grid while fixing the physical question

Use four symbol-comparison levels with nℓ=n·2^ℓ and aℓ=a/2^ℓ. Hold box length n a, signed q, temperature and physical coefficients fixed. Compare λ/a with m²+c²k² and a S with T/(m²+c²k²).

$$
\frac{4}{a^2}\sin^2(ka/2)
=k^2-\frac{a^2 k^4}{12}+O(a^4)
\quad\text{at fixed }k.
$$

The operator gap is second order in spacing for a fixed nonzero physical mode. A constant mode has zero gap; a constrained constant mode is excluded, so do not divide by its zero eigenvalue. The refinement panel compares periodic symbols, not newly inverted larger covariance matrices.

The finite calculator accepts 3–16 sites; four symbol levels can reach 128 sites. This distinction prevents a symbolic resolution comparison from being presented as a large-system simulation.

## 17. Reproduce the four-site result in Python

Run this independent small floating-point exercise in a Python environment with NumPy. It constructs the edge energy before transforming the inverse.

```python
import numpy as np

n = 4
a, mass2, speed2, temperature = 1.0, 1.0, 1.0, 1.0
incidence = np.zeros((n, n))
for j in range(n):
    incidence[j, j] = -1
    incidence[j, (j + 1) % n] = 1
K = a * mass2 * np.eye(n) + speed2 / a * incidence.T @ incidence
C = temperature * np.linalg.solve(K, np.eye(n))
indices = np.arange(n)
F = np.exp(-2j * np.pi * np.outer(indices, indices) / n) / np.sqrt(n)
spectrum = F @ C @ F.conj().T
assert np.allclose(C[0], np.array([7, 3, 2, 3]) / 15)
assert np.allclose(np.diag(spectrum).real, [1, 1/3, 1/5, 1/3])
assert np.allclose(spectrum, np.diag(np.diag(spectrum)))
assert np.allclose(K @ C, temperature * np.eye(n))
print(C[0], np.trace(C), a * np.diag(spectrum).real)
```

For the constrained case, replace the inverse by B(BᵀKB)⁻¹Bᵀ and check PKC=TP. Do not run a full-space solve on a singular massless ring. This exercise demonstrates the equations; its floating tolerances are not rigorous error enclosures.

## 18. Six prediction-led comparisons

| Comparison | Predict before calculating | Inspect afterward |
|---|---|---|
| T=1 versus T=2 | Powers and covariance double; ξ stays fixed | Covariance, powers, infinite amplitude and length |
| Ring versus open versus pinned | DFT mixing appears away from periodic geometry | Entire transformed covariance, not just its diagonal |
| Full versus mean-zero ring | Constant power disappears | Row sums and the constant covariance subtraction |
| Positive versus zero mass | Finite constrained field can survive | Massless curve, sign changes and infinite-field status |
| c²=0 with and without constraint | Independence survives only without the constraint | Zero or negative off-diagonal covariance |
| Fixed-length refinement with q=−1 | The array index changes while k stays fixed | Alias mapping, operator gap and a-normalized power |

For each comparison, write one prediction, record one exact value or interval, and explain any disagreement. Rotate or replay a drawing only after identifying the ensemble; a display animation is not a new equilibrium simulation.

## 19. Choose examples to isolate assumptions

| Example IDs | What they isolate |
|---|---|
| `massive-ring`, `four-site-hand-check`, `sixteen-site-ring` | Baseline, exact small example and finite circumference |
| `hotter`, `soft-mass`, `stiff-gradient` | Temperature, soft constant mode and gradient penalty |
| `independent-sites`, `independent-constrained` | Local independence versus global conditioning |
| `open-chain`, `pinned-chain`, `massless-pinned` | Boundary geometry and normalizability |
| `massless-ring-constrained`, `massive-ring-constrained`, `massless-pinned-constrained` | Constraints in three distinct settings |
| `fixed-box-refined`, `odd-ring`, `shifted-origin` | Spacing, parity and spatial origin |
| `negative-alias`, `nyquist` | Signed wavenumber and self-conjugate mode |
| `equivalent-spelling`, `maximum-coprime` | Equivalent rational inputs and large exact output |

Input coefficients use nonnegative integer or fraction notation with at most 12 digits per numerator and denominator. Their upper bound is 16; spacing lies between 1/16 and 16, and temperature is positive. Origin and selected mode are valid site indices. These are calculator limits, not physical universality claims.

## 20. What numerical evidence establishes

Finite matrices, covariances, reduced LDLᵀ pivots and projected residuals use exact rational arithmetic. General Fourier phases, square roots and logarithms use outward intervals. The numerical engine uses 768-bit working scale and 512-bit output quantization; its square-root helper uses 512 bits. These are arithmetic settings, not a claim of 512 correct relative bits for arbitrarily tiny values.

An independent reference constructs incidence matrices and bordered inverses, then evaluates the full complex Fourier transform at 300 decimal digits. Cross-checking 30 inputs against these point references supports the implementation but is not a proof that the reference values have certified error bounds. Trace, Hermitian symmetry and inverse transformation check different aspects of consistency.

An attractive plot is weaker evidence than a verified matrix. A finite example is weaker evidence than a theorem about every system size. Before using the model in AI research, state the Gaussian assumption, boundary, constraint, source normalization and observation noise. A non-Gaussian field can share this covariance while differing in higher connected moments.

## 21. Worked questions

1. **Why does spacing multiply raw Fourier power in the continuum comparison?** The energy precision is K=a A. Thus T/λ is the variance of a unitary discrete coordinate, while a T/λ corresponds to T divided by the differential-operator eigenvalue. At a=1/2,m²=T=1,c²=0, the two values are 2 and 1.

2. **Check the four-site variance without matrix inversion.** Average the inverse eigenvalues: (1+1/3+1/5+1/3)/4=7/15. Multiplying by four gives trace 28/15, which equals the sum of powers.

3. **Does negative covariance prove a negative local coupling?** No. For eight independent sites conditioned to sum zero, Cᵢⱼ=−1/8 for i≠j at M=T=1. The global constraint creates dependence despite c²=0.

4. **Can the periodic eigenvalue formula describe a pinned chain?** Generally no. The Fourier basis remains complete but does not diagonalize the pinned precision. Compute the full FCF* or use the actual boundary eigenbasis.

5. **Does the positive ratio 21/7 establish a finite massless decay length?** No. It gives the local number 1/log(3), while subsequent covariance becomes negative. The finite mean-zero ensemble exists; an infinite stationary field with finite point variance does not.

6. **Where does negative mode index 7 move under an eight-to-sixteen-site fixed-length refinement?** Its signed wave is −1, so it moves to index 15. Keeping index 7 would change the physical wavenumber.

7. **If temperature doubles, does the correlation length double?** No. With fixed coefficients, C, S and A∞ double, while r and ξ are unchanged. This follows directly from the recurrence and its amplitude jump.

8. **Is a finite-ring semilog slope an exact measurement of ξ?** Not generally. Its covariance contains both periodic paths and possibly a constant subtraction. Use the infinite-chain equation to define ξ, then assess finite-size corrections before fitting data.

## 22. Research connections and further reading

This example connects local quadratic regularization to nonlocal Gaussian kernels, Fourier preconditioning, mesh consistency and constraints in inverse problems. A research extension should compare full operators and noise models, rather than infer a physical mechanism from a covariance plot alone. Interactions, higher dimensions and genuine critical behavior require further analysis beyond this Gaussian chain.

For the continuum Gaussian Green function and correlation-length discussion, read [David Tong, Statistical Field Theory, sections 2.2.1–2.2.3](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2). The finite-spacing factors, constrained finite examples and numerical conventions above are derived for this chapter's declared model. They are not a transcription of a continuum formula into every finite boundary geometry.
