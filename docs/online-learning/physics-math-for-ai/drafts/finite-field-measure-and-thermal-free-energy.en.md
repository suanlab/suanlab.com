## 1. A landscape needs a rule for counting its volume

Imagine a height map representing energy. A low valley is attractive, but a broad valley can carry more probability than a narrow valley of the same depth. To turn the landscape into a statistical model, specify both its energy and how volume is counted. The partition function sums weighted volume, not just minimum heights.

Earlier chapters computed Gaussian field means and correlations. This chapter asks what additional information is required for an **absolute free energy** and what happens when a model's coefficients change with temperature. The prerequisites are multivariable Gaussian integration, matrices and differentiation. We set Boltzmann's constant to one, so temperature has energy units.

This manuscript is a preparation draft. Its finite mathematical core has independent checks; a reader, interactive controls and browser verification are still pending. No general interacting continuum field theory is being solved here.

## 2. A spatial field is many coordinates with geometric meaning

A real field on a finite lattice is a vector φ=(φ1,…,φN). Unlike an arbitrary feature vector, its coordinates have positions and neighboring pairs. Use a one-dimensional chain or a rectangular two-dimensional grid. If the spacing is a and spatial dimension is d, one cell has volume v=a^d and the modeled region has volume V=Nv.

The field value might represent a local magnetization, a displacement or an approximation error. A sitewise external source j couples to that value. Changing a while holding N fixed changes the physical size. Increasing N while holding a fixed increases volume. Holding physical size fixed while reducing a is refinement. These are different experiments.

The finite implementation supports two or more sites per axis and at most 32 sites total. The theory below explains the general finite construction; that implementation limit is not a physical law.

## 3. From a gradient penalty to a lattice matrix

An energy density containing rφ²/2 penalizes field amplitude, while κ|∇φ|²/2 penalizes neighboring differences. Approximate each directional derivative by a difference divided by a and integrate using cell volume v. With an explicitly chosen boundary rule,

$$
E(\varphi,T)=E_0(T)+\frac12\varphi^\mathsf{T}K(T)\varphi-J(T)^\mathsf{T}\varphi,
\quad K=v\left[rI+\frac{\kappa}{a^2}L\right],\quad J=vj,\quad E_0=Vf_0.
$$

L is the graph Laplacian, including the stated boundary contributions. Source density j and vector J differ by cell volume. Omitting that factor silently changes the physical source when the grid changes. The background density f0 does not move the preferred field configuration, but it contributes to absolute free energy and its derivatives.

## 4. Boundaries determine which differences exist

For an open chain, include internal neighboring differences only. For periodic boundaries, every site contributes a forward difference in each spatial direction, including the wraparound difference. For fixed-zero exterior boundaries, add differences from each boundary site to its neighboring exterior zero.

For two periodic sites the forward differences are φ2−φ1 and φ1−φ2. They are two terms in this finite-difference convention, giving L=[[2,−2],[−2,2]]. Replacing this by a simple graph with a single edge changes the model. On a 2×2 grid with fixed-zero exterior values, each site's diagonal receives four directional contributions: two internal and two exterior.

These are choices of finite model. Fixed boundary values and removal of the mean mode are also different operations.

## 5. State the reference measure before normalizing

With q independent real field coordinates, choose a positive, temperature-independent reference field scale φ*. On the full space, the integration measure is the ordinary volume divided by φ*^q. On a linear constraint space, use its induced Euclidean volume divided by the same power.

$$
Z(T)=\int_{\mathcal H}\frac{d\mathrm{vol}_{\mathcal H}(\varphi)}{\varphi_*^q}
\exp[-E(\varphi,T)/T].
$$

This makes Z dimensionless. Its absolute value depends on the declared reference measure. Multiplying all volume weights by a constant changes Z even when every normalized probability remains the same. Thus a plot of normalized field fluctuations alone cannot determine an absolute free energy.

Throughout the thermal derivatives, keep spacing, boundaries, constraints and reference scale fixed. A temperature-dependent measure would contribute additional derivative terms.

## 6. Normalizability is a condition on the integration space

The quadratic precision must be positive definite on the space being integrated. A positive mass r is sufficient when κ≥0, but it is not necessary with fixed boundaries or a mean-zero constraint. A moderately negative r can be stabilized by positive spatial eigenvalues.

A periodic massless field has a flat uniform direction. Its full Gaussian integral diverges. Restricting to mean zero removes that direction and defines another ensemble; it does not give a finite value to the original divergent integral. A negative direction also makes a full-space Gaussian integral diverge, regardless of a finite linear source.

The core checks exact positive LDL pivots. A failed check means the stated Gaussian is not normalizable. A zero first pivot does not by itself prove that the full matrix is singular: [[0,−1],[−1,0]] is invertible and indefinite.

## 7. The mean-zero plane has its own volume element

Write a mean-zero field as φ=Bz with columns ei−eN. This basis is convenient but not orthonormal. Its Gram determinant is N. If A=BᵀKB, the restricted precision determinant with respect to orthonormal volume is det(A)/N.

$$
d\mathrm{vol}_{\mathcal H}=\sqrt{\det(B^\mathsf{T}B)}\,dz=\sqrt N\,dz,
\qquad \det K_{\mathcal H}=\frac{\det(B^\mathsf{T}KB)}{N}.
$$

For two sites, φ=(z,−z) travels a distance √2 dz in the plane. Forgetting that factor leaves normalized moments unchanged but gives the wrong absolute partition function. A covariance test alone therefore misses this error.

## 8. Complete the square and keep every normalization factor

Let G=B(BᵀKB)^−1Bᵀ, using B=I for the full space. The mean is μ=GJ and the covariance is Cφ=TG. The minimum energy on the declared space is m=E0−JᵀGJ/2.

$$
\log Z=\frac q2\log(2\pi T)-\frac12\log\det K_{\mathcal H}
-q\log\varphi_*-\frac{m}{T},\qquad F=-T\log Z.
$$

The logarithmic determinant records fluctuation volume. A sharp direction has a small width and contributes less volume. A source moves the center and changes m. The measure scale fixes the reference for that volume. Keeping only exp(−m/T) discards the entire Gaussian volume contribution.

The core returns the rational operands of this expression, not a purported rational value for π or logarithms.

## 9. Worked example: two coupled sites

Take a=1, T=2, r=κ=1, open boundaries, J=(1,0), E0=0 and φ*=1. Then K=[[2,−1],[−1,2]], det K=3 and G=[[2,1],[1,2]]/3. The mean is (2/3,1/3), and the covariance is twice G.

$$
m=-\frac13,\qquad \log Z=\log(4\pi)-\frac12\log3+\frac16.
$$

For this fixed Hamiltonian, U=m+qT/2=5/3 and the total heat capacity at fixed volume is q/2=1. The negative minimum energy is allowed because the source shifts the energy reference. Each independent real quadratic coordinate contributes T/2 above that minimum.

## 10. A temperature-dependent effective energy needs a different observable

An effective model may already include the entropy of hidden microscopic variables. Its coefficients can depend on T even when a microscopic Hamiltonian does not. Differentiating exp[−E(φ,T)/T] then differentiates both the explicit inverse temperature and the energy coefficients.

$$
\mathcal A=E-T\partial_T E,\qquad
U=T^2\partial_T\log Z=\langle\mathcal A\rangle.
$$

The thermal internal energy U is generally not the average effective energy 〈E〉. To define the derivative, specify a path of coefficients as T varies. Giving only their values at one temperature is insufficient. The foundation input supplies values, first derivatives and second derivatives at that temperature, equivalent to a local quadratic Taylor path.

This is also relevant to machine learning: a family of energy models retrained at different temperatures is not automatically a single fixed Hamiltonian. Differentiating across that family includes how its parameters change.

## 11. Derive the corrected heat-capacity identity

For any observable O on a fixed measure space, differentiate the normalized density. Its logarithmic derivative is (A−〈A〉)/T². Hence the derivative of 〈O〉 is 〈O_T〉+Cov(O,A)/T². Apply this with O=A and note that A_T=−T E_TT.

$$
C_V=\frac{dU}{dT}=\frac{\operatorname{Var}(\mathcal A)}{T^2}
-T\langle\partial_T^2 E\rangle.
$$

For a fixed Hamiltonian the second term vanishes and A=E, recovering a nonnegative variance formula. For an arbitrary effective family the second term can make this derivative negative. That does not demonstrate a negative canonical heat capacity for a fixed microscopic Hamiltonian.

## 12. Gaussian moments compute both terms exactly

Define Q=K−TK_T, b=J−TJ_T and c=E0−TE0_T. Then A=c+φᵀQφ/2−bᵀφ. Gaussian moment identities give

$$
U=c+\frac12\operatorname{tr}(QC_\varphi)+\frac12\mu^\mathsf{T}Q\mu-b^\mathsf{T}\mu,
\quad \operatorname{Var}(\mathcal A)=\frac12\operatorname{tr}[(QC_\varphi)^2]
+(Q\mu-b)^\mathsf{T}C_\varphi(Q\mu-b).
$$

The remaining expectation is E0_TT+tr(K_TT Cφ)/2+μᵀK_TTμ/2−J_TTᵀμ. These formulas remain valid with a linear constraint when the covariance and mean are those of that constrained Gaussian.

The independent checker does not reuse these formulas. It differentiates the determinant and completed-square energy in a second-order Fraction Taylor ring, then differentiates log Z itself.

## 13. Two counterexamples that expose the wrong formula

Consider q independent real coordinates, zero source, zero background and φ*=1.

| Coefficient path | Normalized covariance | Average effective energy | Thermal U | Total Cv |
|---|---|---|---|---|
| K=I | T I | qT/2 | qT/2 | q/2 |
| K=T I | I | qT/2 | 0 | 0 |
| K=T² I | I/T | qT/2 | −qT/2 | −q/2 |

For K=T I the distribution is unchanged with temperature: its exponent is −φᵀφ/2. Therefore log Z is constant and U=0. Substituting Var(E)/T² would instead give q/2 for the heat capacity. The missing coefficient derivatives cause that error.

For K=T² I, log Z=(q/2)log(2π)−(q/2)log T. Its first two thermal derivatives give the negative values in the table. This is a valid positive Gaussian family; identifying it with a fixed microscopic Hamiltonian is an additional, unsupported claim.

## 14. Thermodynamic entropy and field-distribution entropy differ

Thermodynamic entropy is S=−∂T F=log Z+U/T. The Gibbs entropy of the field density relative to the declared reference measure is SG=log Z+〈E〉/T. Combining them yields

$$
S=S_G-\langle E_T\rangle.
$$

They coincide for a fixed Hamiltonian. They need not coincide for an effective family. Hidden microscopic entropy can be represented inside E(φ,T), so the entropy of the visible field distribution alone does not describe every thermodynamic contribution.

Continuous entropy also depends on the declared reference scale. Changing units without changing the associated volume measure is not a harmless relabeling of an absolute entropy.

## 15. Constants can matter even when probabilities do not change

Multiply φ* by c>0 while leaving field coordinates and energy fixed. Then log Z decreases by q log c. F increases by Tq log c, and S decreases by q log c. U and Cv do not change because this measure factor is temperature independent.

In contrast, change coordinates φ=c z. The same model now has Kz=c²K, Jz=cJ and φ*z=φ*/c. Its determinant gains c^(2q), exactly canceling the reference-scale contribution. Its absolute Z is unchanged, while the mean and covariance transform with the coordinate units.

Add a field-independent total background a+bT+cT². All normalized field probabilities stay the same, but F gains that background, U gains a−cT², S gains −b−2cT, and Cv gains −2cT. A common energy offset cancels in probabilities; it cannot be discarded indiscriminately from absolute thermodynamic derivatives.

## 16. Real Fourier modes do not double the integration dimension

For a periodic chain, a unitary Fourier transform diagonalizes the precision. A real field obeys φ̃(N−k)=φ̃(k)*. The zero mode is real; for even N the Nyquist mode is also real. Every other conjugate pair contains two real coordinates in total.

For one pair, use qc=√2 Re φ̃(k) and qs=−√2 Im φ̃(k). Then |φ̃(k)|²+|φ̃(N−k)|²=qc²+qs² and dqc dqs=2 dRe dIm. An orthonormal real Fourier basis preserves ordinary volume. The Gaussian determinant has one square-root factor per independent real coordinate.

On a one-dimensional periodic lattice, λk=v[r+4κ sin²(πk/N)/a²]. The full determinant is the product over all N eigenvalues: paired eigenvalues occur twice because the pair has two real coordinates. Integrating independent complex variables for every k instead creates 2N real coordinates and gives the wrong answer.

## 17. Worked example: a constrained massless pair

Take two periodic sites, a=κ=1, r=0, T=2 and zero source. The uniform eigenvalue is zero, so the full integral diverges. Restricting to φ1+φ2=0 leaves a single mode with eigenvalue 4.

With B=(1,−1)ᵀ, det(BᵀKB)=8 and the Gram determinant is 2. Thus the restricted determinant is 4, Z=√(4π)/2=√π, and the covariance is [[1,−1],[−1,1]]/4. Using det A=8 directly would incorrectly divide Z by √2.

This demonstrates three separate checks: normalizability on the declared space, correct induced volume, and moments of the normalized constrained law. None can replace the others.

## 18. A runnable thermal-path experiment

This NumPy example uses diagonal precision so the entire calculation fits on one screen. It checks a fixed model, K=T I and K=T² I using the corrected identity. NumPy performs floating-point arithmetic; the repository's independent foundation checks use exact fractions instead.

```python
import numpy as np

def thermal(q, T, r, r1, r2):
    C = (T / r) * np.eye(q)
    Q = (r - T * r1) * np.eye(q)
    mean_E = 0.5 * r * np.trace(C)
    U = 0.5 * np.trace(Q @ C)
    var_A = 0.5 * np.trace((Q @ C) @ (Q @ C))
    mean_E2 = 0.5 * r2 * np.trace(C)
    Cv = var_A / T**2 - T * mean_E2
    return mean_E, U, Cv

q, T = 2, 2.0
print(thermal(q, T, 1.0, 0.0, 0.0))       # (2, 2, 1)
print(thermal(q, T, T, 1.0, 0.0))         # (2, 0, 0)
print(thermal(q, T, T*T, 2*T, 2.0))       # (2, -2, -1)
```

Before running, predict which distribution stays unchanged with T. Then compare the printed average effective energy with U. Change q and check the degree-of-freedom scaling. This experiment does not simulate dynamics or estimate an interacting partition function.

## 19. Questions with explanations

**1. Does multiplying every integration weight by the same constant change the mean?** No. The constant cancels between numerator and Z. Absolute F can still change because it depends on log Z.

**2. Can a positive diagonal prove normalizability?** No. [[1,2],[2,1]] has positive diagonal and a negative eigenvalue. Check positivity on the full declared integration space.

**3. Is fixing the mean to zero the same as choosing a smaller source?** No. It changes the support and number of integrated degrees of freedom. A uniform source cannot remove the Gaussian massless divergence.

**4. Why are first and second coefficient derivatives part of an input?** U and Cv concern a temperature path, not just a single probability distribution. Two families can pass through the same K(T0) while having different derivatives and thermal observables.

**5. Does negative Cv in the K=T² family violate the fixed-Hamiltonian variance formula?** No. Its hypotheses do not hold: the effective energy explicitly depends on T.

**6. Can identical normalized covariance determine entropy?** Not absolute thermodynamic entropy. The reference measure and temperature-dependent background matter, and the visible Gibbs entropy may differ from the thermodynamic entropy.

**7. Why does a conjugate Fourier pair count twice in a determinant but not as two independent complex variables?** It has two real coordinates. Two independent complex variables would have four.

**8. Does a finite Gaussian calculation establish a critical exponent?** No. Interactions, limits and control of neglected terms require additional work. A finite determinant is a useful diagnostic rather than a general critical theory.

## 20. Research connection and next steps

An energy-based AI model specifies a score landscape. Absolute likelihood additionally requires a normalizer relative to a reference measure. Changing coordinates, discretization or hidden variables changes the bookkeeping. A normalized prediction may remain the same while an absolute evidence or entropy claim changes.

Temperature derivatives also require a declared parameter path. Treating a temperature-tuned model family as a fixed Hamiltonian can turn an algebraically correct variance calculation into the wrong scientific conclusion. Report the energy convention, measure, spatial scale, constraints and what is held fixed.

Next, construct local order parameters from microscopic states, including multiplicities and hidden-state entropy. Then examine interacting scalar fields and distinguish constrained probability free energies from Legendre effective actions. Criticality, renormalization and scale separation receive their own treatments.

## 21. Sources and scope

[Tong §1.1.1 and §1.3](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S1) supplies the microscopic-to-field context and temperature-dependent effective free-energy warning. [Tong §2 and §2.1](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2) supplies the quadratic fluctuation and determinant context. The finite reference-measure conventions, thermal identities and counterexamples above are derived explicitly for this chapter.

The accompanying model contract and independent Fraction/Taylor-ring checks cover finite Gaussian fields. They do not certify numerical logarithm displays, continuum measures, interacting integral errors, phase transitions or the completion of this unit's reader interface.
