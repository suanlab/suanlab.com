## 1. Searching for a state means choosing an energy landscape

A Hamiltonian assigns energy statistics to quantum states. A variational calculation asks which admissible state gives the smallest mean energy. The central guarantee is a bound on an expectation, not a promise that an optimizer will find the best state. Keeping these two statements separate prevents a large class of mistakes in quantum simulation and machine learning.

Picture a landscape drawn over possible state directions. Multiplying a vector by a nonzero complex number does not change its physical direction or energy. The landscape therefore lives on normalized rays rather than on every coordinate of an unrestricted array. A trial family explores only part of that landscape. A good coordinate system can make exploration easier, but it cannot create directions absent from the family.

| Object | What it changes | What it does not establish |
| --- | --- | --- |
| A normalized trial state | The energy expectation being evaluated | That the state minimizes the energy |
| A restricted family of states | Which directions a search can explore | That the exact ground space is accessible |
| A numerical optimization algorithm | How parameters are searched | A global minimum or an error certificate |

The route through this chapter is **state direction → energy average → variation and residual → spectral information → restricted approximation → research checks**. Prerequisites are complex inner products, Hermitian operators and elementary differentiation. No physical time evolution is specified by an energy landscape.

## 2. State the Hamiltonian and its domain first

Begin with a finite complex inner-product space and a Hermitian matrix H. For a nonzero raw vector z, divide by its squared norm when computing expectations. This permits integer or otherwise convenient raw representatives without changing the state. The energy unit is the common unit used by the entries of H; vector normalization does not remove that unit.

$$
N=z^\dagger z>0,\qquad
|\psi\rangle=\frac{z}{\sqrt N},\qquad
E(z)=\frac{z^\dagger Hz}{z^\dagger z},\qquad H=H^\dagger.
$$

Hermiticity makes the numerator real. The zero vector is excluded because it represents no normalized state. A program should report that missing definition instead of treating its energy as zero. With complex entries, every inner product uses conjugation in its first slot. Replacing a conjugate transpose by a transpose can produce a complex number that is not an energy expectation.

In an infinite-dimensional space, “Hermitian matrix” must be replaced by an appropriate self-adjoint operator with a specified domain, or by its closed quadratic form. A ground-energy variational bound needs a lower spectral bound and admissible trial functions. Differentiating a formal expression without checking boundary conditions does not supply these assumptions. The finite numerical family below does not verify domains of differential operators.

## 3. The lower bound is a weighted-average theorem

Let distinct energy values be epsilon j and let Pj be their orthogonal spectral projectors. A degenerate energy has one projector onto its entire eigenspace. The normalized state supplies nonnegative spectral probabilities summing to one. The mean cannot lie below the smallest value being averaged.

$$
H=\sum_j\epsilon_jP_j,\qquad
p_j=\frac{z^\dagger P_jz}{N},\qquad
\sum_jp_j=1,\qquad
E-\epsilon_0=\sum_{j>0}(\epsilon_j-\epsilon_0)p_j\ge0.
$$

This proves the finite variational principle. Equality requires zero probability outside the ground eigenspace. If that space has dimension two, any normalized superposition inside it is a ground state. Comparing fidelity with one arbitrarily selected ground vector would misdescribe other exact ground states.

For a self-adjoint operator bounded below, the corresponding statement uses the spectral measure or the quadratic form: the infimum of the spectrum bounds all admissible expectations. That infimum need not be an attained eigenvalue. A free particle on the whole line illustrates why a limiting trial sequence and an actual normalizable ground state are different objects. Do not replace every spectral infimum by a ground projector when no such eigenspace exists.

## 4. Differentiate along complex directions using a real parameter

Choose an arbitrary complex direction eta and a real parameter t. Both the numerator and denominator change along z+t eta. Differentiating the quotient produces a residual. This is often the clearest way to derive the stationary equation without mixing conventions for complex gradients.

$$
r=(H-E)z,\qquad
\left.\frac{dE(z+t\eta)}{dt}\right|_{t=0}
=\frac{2\operatorname{Re}(\eta^\dagger r)}{N},\qquad
E'(0)=0\ \text{for every }\eta\ \Longleftrightarrow\ Hz=Ez.
$$

For the reverse implication, use eta=r: a zero derivative then forces the squared residual norm to vanish. Every eigenstate is stationary, including the highest-energy state. Stationarity is therefore weaker than minimization. On the unit sphere, the residual is tangent because z dagger r is zero. Up to the chosen real-inner-product convention, it is the constrained gradient direction.

Two variations must be flat: eta=z rescales the representative, and eta=i z changes its global phase to first order. These checks catch normalization and conjugation errors. A single direction with zero derivative does not prove stationarity in all directions. The supplied calculation also returns the exact second derivative of the real path, including derivatives of the normalization denominator. Explicitly, with every quantity evaluated at t=zero,

$$
N'=2\operatorname{Re}(z^\dagger\eta),\qquad
E''=\frac{2\eta^\dagger(H-E)\eta}{N}-\frac{2E'N'}{N}.
$$

The second term matters because the raw path need not stay on the unit sphere.

## 5. Variance is a residual certificate, with a limited conclusion

For a normalized state, energy variance is the squared norm of its eigenvalue-equation residual. This is a useful bridge between a physical statistic and numerical linear algebra. It supplies more information than an energy value alone, but its meaning depends on which eigenvalue is being sought.

$$
\operatorname{Var}(H)=\frac{\|r\|^2}{N}
=\langle H^2\rangle-E^2
=\sum_jp_j(\epsilon_j-E)^2,\qquad
\min_j|\epsilon_j-E|\le\sqrt{\operatorname{Var}(H)}.
$$

The last inequality follows because an average of squared distances is at least the smallest squared distance. It certifies that some spectral value is nearby. It does not identify that value as the ground energy. An exact excited eigenstate has zero variance and zero residual while being completely outside the ground eigenspace.

A small residual is also an absolute statement in an energy unit. Rescaling H rescales the residual and the standard deviation. For an eigenvector-accuracy estimate, a separation from competing eigenvalues is needed. Near a degeneracy, individual eigenvectors can be sensitive even when the combined invariant subspace is accurately determined. This is a reason to present spectral subspaces rather than a preferred numerical basis inside a degenerate space.

## 6. A positive gap turns excess energy into a ground-space bound

Suppose the ground eigenspace exists and the next distinct energy is separated by a known positive gap Delta. Every probability outside the ground space costs at least Delta in mean energy. Applying that observation to the weighted average gives a leakage bound.

$$
\Delta=\epsilon_1-\epsilon_0>0,\qquad
1-p_0\le\frac{E-\epsilon_0}{\Delta},\qquad
L_0\le\epsilon_0,\quad 0<d\le\Delta
\ \Longrightarrow\ 1-p_0\le\frac{E-L_0}{d}.
$$

The assumptions are operationally important. A variational upper estimate U0 for the ground energy cannot replace epsilon zero in the numerator: doing so can make the claimed error artificially small. A certified lower bound L0 is safe, though it may be loose. Likewise, an optimistic gap estimate is unsafe; the denominator must be an established positive lower bound if an exact gap is unavailable.

The final probability bound can be capped at one. A constant Hamiltonian has no distinct excited energy, so the positive-gap formula is not defined; all states already belong to its ground space. In a gapless infinite system, a finite-size gap cannot automatically certify infinite-system ground-state fidelity. The chapter's matrix calculations describe the supplied finite Hamiltonian.

## 7. A two-level example separates three questions

Take H=diag(0,2). A ground vector, an excited vector and a balanced coherent state provide three distinct diagnostics. The phase i in the last state is retained, even though this diagonal Hamiltonian makes its energy depend only on populations.

| Normalized state | Mean energy | Energy variance | Ground-space probability |
| --- | --- | --- | --- |
| (1,0) | 0 | 0 | 1 |
| (0,1) | 2 | 0 | 0 |
| (1,i)/sqrt(2) | 1 | 1 | 1/2 |

$$
|\psi(\theta,\phi)\rangle=
\begin{pmatrix}\cos\theta\\e^{i\phi}\sin\theta\end{pmatrix},\qquad
E=2\sin^2\theta,\qquad
\operatorname{Var}(H)=4\sin^2\theta\cos^2\theta.
$$

Moving theta transfers population between the energy levels. The energy is stationary at the ground and excited endpoints, but one is a minimum and the other a maximum. Moving phi leaves these particular energy statistics fixed. That invariance is a property of this diagonal H, not a rule that relative phases never matter. An off-diagonal Hamiltonian couples amplitudes and can distinguish those phases.

An instructive picture uses an energy-level diagram beside the trial-state populations and a residual arrow. A zero residual arrow must not be colored “ground state” without checking the spectral location. For a degenerate ground level, draw the combined ground sector rather than selecting one coordinate and hiding the others.

## 8. Preserve tiny quantities and keep units visible

Set H=diag(0,epsilon) and use the raw state (M,1). Taking M large gives a small but nonzero excited population. The energy and variance become much smaller than a usual display tolerance while their exact algebra remains simple.

$$
E=\frac{\epsilon}{M^2+1},\qquad
\operatorname{Var}(H)=\frac{\epsilon^2M^2}{(M^2+1)^2},\qquad
1-p_0=\frac1{M^2+1}.
$$

For epsilon=10 to the power minus six and M=one million, the variance is approximately ten to the power minus twenty-four. Calling it zero would incorrectly classify the state as an eigenstate. The gap bound is tight in this two-level example. Its small energy scale must not be confused with an arbitrary numerical rounding rule.

A shift H+aI adds a to E but preserves residuals and variance. Multiplying H by a real b multiplies residuals by b and variance by b squared. A negative b reverses the spectral order: the old highest-energy space becomes the new ground space. These are useful independent checks and reminders that energy minimization is not unchanged under a sign reversal.

## 9. A continuous trial function that can be solved by hand

Consider the one-dimensional harmonic oscillator with positive mass m and frequency omega. The normalized Gaussian with alpha positive is an admissible rapidly decaying trial function. Its parameter controls inverse squared width, so alpha has the unit of inverse length squared.

$$
H=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+\frac{m\omega^2x^2}{2},\qquad
\psi_\alpha(x)=\left(\frac{\alpha}{\pi}\right)^{1/4}e^{-\alpha x^2/2},\qquad
\langle x^2\rangle=\frac1{2\alpha}.
$$

Differentiate the Gaussian twice and integrate using its normalization. The kinetic term grows with alpha because narrower functions vary more sharply; the potential term decreases because the position distribution becomes narrower. This competition gives an intuitive explanation of a nonzero optimal width.

$$
E(\alpha)=\frac{\hbar^2\alpha}{4m}+\frac{m\omega^2}{4\alpha},\qquad
\alpha_* =\frac{m\omega}{\hbar},\qquad
E(\alpha_*)=\frac{\hbar\omega}{2},\qquad
\operatorname{Var}(H)=\frac{(m\omega^2-\hbar^2\alpha^2/m)^2}{8\alpha^2}.
$$

The optimum follows by differentiation or the arithmetic-geometric mean inequality. The variance follows from the fourth position moment, which is three divided by four alpha squared. Here the trial family contains the exact ground state, so the optimum variance vanishes. That favorable result is special; optimizing a Gaussian in a nonquadratic potential does not generally produce an exact eigenstate. This continuous example is an analytic derivation, not an additional differential-operator mode of the finite matrix calculator.

Minimizing within Gaussians alone would not prove a global lower bound. For the oscillator domain, integration by parts with vanishing boundary terms gives the factorization below. Positivity of the squared annihilation-operator norm establishes the lower bound hbar omega/2 for admissible states, and the optimizing Gaussian attains it.

$$
a=\sqrt{\frac{m\omega}{2\hbar}}x+\sqrt{\frac{\hbar}{2m\omega}}\frac{d}{dx},\qquad
H=\hbar\omega\left(a^\dagger a+\frac12\right),\qquad
a\psi_{\alpha_*}=0.
$$

## 10. Restricting the family changes what a zero gradient means

Let a normalized family depend on real parameters theta. Its parameter derivative probes only the state directions produced by the family's Jacobian. A vanishing parameter gradient can mean that the residual is orthogonal to those accessible directions, not that the full residual vanishes.

$$
\frac{\partial E}{\partial\theta_a}
=2\operatorname{Re}\left\langle
\frac{\partial\psi}{\partial\theta_a},(H-E)\psi\right\rangle.
$$

There can also be a singular parameterization. For example, use real amplitudes with angle pi/4 plus theta cubed in the two-level example. At theta zero the parameter derivative is zero because the Jacobian is zero, although the state is not an energy eigenstate. A flat parameter direction and a physically stationary state are different claims.

Linear trial spaces lead to the nonorthogonal Ritz problem in the next chapter. Nonlinear wavefunctions, neural quantum states and quantum circuits need separate analysis of accessible directions and optimization. Record approximation error from the family, optimization error within it and numerical or statistical error in estimating the objective separately. None is removed by the variational principle alone.

## 11. Research connections require an error budget

Variational quantum eigensolvers estimate energy expectations from repeated measurements. If the prepared state is physical and the observable is the same exact H, its true mean still obeys the variational bound. A finite noisy estimate can fall below the true ground energy without contradicting that theorem. Readout bias, a miscalibrated Hamiltonian or an error-mitigation procedure that yields a nonphysical effective state needs a separate audit.

Variational Monte Carlo writes the energy as an average of the local energy H psi divided by psi under the normalized squared amplitude. This representation requires admissible functions and finite expectations. At exact zero-amplitude points the ratio is undefined; assigning a version on a set of zero sampling probability does not change the expectation. This does not justify replacing large finite near-node values or failed evaluations by zero, which generally changes the estimator and can bias it. Sampling variance, autocorrelation, rare events and optimizer adaptation must be reported. The variance of an energy estimator is not automatically the quantum energy variance of the state.

The same Rayleigh structure appears in principal-component analysis, which maximizes a covariance quotient, and in spectral methods with orthogonality constraints. The optimization direction and constraints matter. In research comparisons, keep H, units, boundary conditions, admissible space and spectral sector fixed. Use energy, residual, a justified gap estimate and sensitivity to increasing the trial family as complementary diagnostics, rather than treating one attractive loss curve as a proof.

Research entry points include [Peruzzo and collaborators on a hybrid variational eigensolver](https://arxiv.org/abs/1304.3061) and [Carleo and Troyer on neural variational states](https://arxiv.org/abs/1606.02318). These motivate the research connections; the finite derivations and examples above are developed here.

## 12. Read the finite experiment as a certified comparison

The prepared numerical family accepts general complex Hermitian matrices of dimension one through four, a nonzero state and an arbitrary direction. Its thirty-six bounded integer fields specify matrix entries, a common energy denominator and a rational real path parameter. The fixed examples are intended to be compared with freely edited preparations; they do not infer an unknown physical Hamiltonian.

Moments, path derivatives and residuals are exact complex-rational calculations. Square-free/Sturm isolation retains signed spectral values and exact multiplicities at 1024 root bits. Whole spectral-space projectors and weights use 768-bit directed work and outward 512-bit observations. No particular numerical eigenvector basis is returned. An interval is a certificate of uncertainty in a reported value, not a simulated sampling distribution.

Follow this sequence: predict the energy sector, inspect variance, identify the ground-space probability, then check whether a positive-gap bound is applicable. Compare a phase direction with an amplitude-transfer direction. Finally choose a path that reaches the zero vector and verify that its moments become undefined rather than zero. Optional motion should guide the reading of comparisons; it must not suggest physical time evolution that the input never specified.

## Research extension: an excited-spectrum cutoff turns variance into a ground certificate

### A residual becomes informative after the spectral sector is identified

The original residual bound locates some nearby energy. Now assume that all energies outside the ground space lie above a separately justified cutoff β. Write λ1 for the next distinct energy above λ0, allowing a degenerate ground space. For a normalized trial state:

$$
\lambda_0<\beta\le\lambda_1,\quad E<\beta,\quad v=\operatorname{Var}(H),\quad d=\beta-E>0,\qquad
 E-\frac{v}{\beta-E}\le\lambda_0\le E.
$$

The left inequality is the ground-energy form of the Temple bound. Its denominator is an energy separation and v has squared energy units. It needs E<β, not merely a small variance. For finite Hermitian H it follows by averaging a polynomial that is nonnegative at every spectral value:

$$
\langle(H-\lambda_0)(H-\beta)\rangle
 =v+(E-\lambda_0)(E-\beta)\ge0,\qquad
 (E-\lambda_0)(\beta-E)\le v.
$$

At λ0 the polynomial vanishes; above β both factors are nonnegative. Expanding its expectation gives the identity and division by β−E gives the bound. No ground eigenvector has to be computed for this algebra, but the cutoff hypothesis must come from evidence outside the trial loss. See [Zhu, Argentati and Knyazev, §2](https://arxiv.org/pdf/1207.3240) for the residual/spectral-separation setting.

### Bound ground-space population without a known ground energy

Let ℓ be the total excited probability. If ℓ>0, its conditional mean μe is at least β. Splitting the energy distribution into the ground level and the excited group gives:

$$
\ell=1-p_0,\quad \mu_e=\mathbb E[H\mid H\ne\lambda_0]\ge\beta,\qquad
 v\ge\frac{\ell}{1-\ell}(\mu_e-E)^2\ge\frac{\ell}{1-\ell}d^2,\qquad
 \ell\le\frac{v}{v+d^2},\quad p_0\ge\frac{d^2}{v+d^2}.
$$

To obtain the first bound, keep the variance between the two groups and drop the nonnegative variance within the excited group. Mean balance expresses the ground group's displacement in terms of μe−E. The assumption E<β implies ℓ<1. At ℓ=0 the desired inequality holds directly without defining μe. This is the one-sided variance, or Cantelli, geometry of this spectral distribution; it bounds the entire ground space, not a selected basis vector in it.

One possible independent cutoff uses a reference matrix and a rigorously bounded perturbation. In the next display eigenvalues are ordered with multiplicity and the reference ground state is nondegenerate:

$$
H=H_{\rm ref}+V,\quad\|V\|_2\le\eta,\quad
 \lambda_j(H_{\rm ref})-\eta\le\lambda_j(H)\le\lambda_j(H_{\rm ref})+\eta,\qquad
 \lambda_1(H_{\rm ref})-\lambda_0(H_{\rm ref})>2\eta
 \ \Longrightarrow\ \beta=\lambda_1(H_{\rm ref})-\eta\le\lambda_1(H).
$$

The ordered bounds follow from min–max because −ηI≤V≤ηI. The strict reference-gap condition preserves a nondegenerate ground level. This is one finite-matrix construction, not a claim that every many-body problem has an easily available norm bound. With a degenerate reference, its cluster and next distinct level need a separate multiplicity-aware argument.

### Test sharpness, unsafe substitutions and uncertain inputs

In a two-level system the estimates can be exact even when the trial is mostly excited:

$$
H=\operatorname{diag}(0,2),\quad p_1=q,\quad0\le q<1,\quad
 E=2q,\quad v=4q(1-q),\quad\beta=2,\qquad
 E-\frac{v}{\beta-E}=0,\quad\frac{v}{v+(\beta-E)^2}=q.
$$

At q=3/4 only one quarter of the state is ground, and the leakage bound correctly reports3/4. At q=1 the state is an exact excited eigenstate, E=β and v=0; the certificate is unavailable rather than a zero-error conclusion. Approaching this endpoint makes the denominator sensitive to input uncertainty.

A higher Ritz estimate of an excited energy is an upper bound, not the required lower cutoff. Substitution can give a false theorem:

$$
H=\operatorname{diag}(0,1,9),\quad V_{\rm trial}=\operatorname{span}\{e_0,e_2\},\quad
 \mu_1^{\rm Ritz}=9,\quad\psi=e_1,\quad E=1,\quad v=0,\qquad
 1-\frac0{9-1}=1>\lambda_0=0.
$$

The proposed β=9 misses the actual excited level1. The displayed result would put the true ground energy at least1, so it is invalid. A gap inferred only from a restricted subspace cannot exclude a missing low-energy direction in the full problem.

For uncertain E and v, propagate directions of inequalities instead of evaluating a formula at convenient midpoints. A conservative enclosure is:

$$
E\in[E_L,E_U],\quad 0\le v\le v_U,\quad\lambda_1\ge\beta_L>E_U,\quad d_L=\beta_L-E_U,\qquad
 E_L-\frac{v_U}{d_L}\le\lambda_0\le E_U,\qquad
 1-p_0\le\frac{v_U}{v_U+d_L^2}.
$$

Indeed, E≥EL while the subtraction v/(β−E) is at most vU/(βL−EU). The leakage expression grows with variance and decreases with the positive separation. This does not assume that the Temple expression is monotone in E. If EU≥βL, this particular certificate cannot be issued. Negative estimated variance is not a physical input to repair by silently setting it to zero.

For the state q=1/4, the following valid but deliberately widened input enclosures illustrate a safe calculation:

$$
E\in[49/100,51/100],\quad v\in[74/100,76/100],\quad\beta_L=199/100,\quad
 d_L=37/25,\qquad
 -87/3700\le\lambda_0\le51/100,\quad 1-p_0\le475/1844.
$$

The true λ0=0 and leakage1/4 are inside these bounds. Statistical intervals would additionally need their justified joint confidence coverage; deterministic interval arithmetic alone is not that sampling guarantee.


```python
from fractions import Fraction as Q
for q in [Q(0), Q(1, 4), Q(3, 4), Q(99, 100)]:
    energy, variance = 2*q, 4*q*(1-q)
    distance = 2-energy
    assert distance > 0
    lower = energy-variance/distance
    leakage = variance/(variance+distance*distance)
    assert lower == 0 and leakage == q
    print(q, energy, variance, lower, leakage)
```

| excited probability q | energy | variance | Temple lower bound | leakage upper bound |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1/4 | 1/2 | 3/4 | 0 | 1/4 |
| 3/4 | 3/2 | 3/4 | 0 | 3/4 |
| 99/100 | 99/50 | 99/2500 | 0 | 99/100 |

![A justified excited cutoff separates the trial mean from the excited sector; the one-sided variance bound is sharp for a two-level source and remains unavailable at the excited eigenstate endpoint.](/assets/learning/quantum-variational-research-excited-cutoff-certificates.en.svg)

The table uses exact fractions. The figure compares the sharp one-sided leakage bound with the weaker capped bound v/(β−E)². These are standalone cutoff and uncertainty calculations. The existing finite reader still reports its exact supplied-H diagnostics; it has no new external-cutoff or sampling-confidence input.

#### Research check 1: the same variance, different ground weights

At q=1/4 and3/4 the variance is3/4 in both cases, but leakage is1/4 and3/4. The mean and the cutoff separation identify which interpretation applies. Variance by itself cannot tell the two preparations apart.

#### Research check 2: a wrong excited cutoff

For the three-level counterexample, setting β=1 makes E<β fail. That is the correct refusal for the excited vector. Setting β=9 only makes the division possible by making a false spectral assertion; an internally consistent arithmetic result is not a certificate of the missing hypothesis.

#### Research check 3: an unresolved denominator

If an energy enclosure reaches the certified cutoff, do not divide by a tiny positive replacement. Report this certificate as unavailable or refine the input evidence. An optimizer's loss curve cannot resolve that inequality by itself. The exact Rayleigh principle still holds for every physical trial state.

## 13. Exercises with explanations

**1. Raw normalization.** For H=diag(0,2) and z=(2,2i), compute N, E and variance.

**Explanation.** The squared norm is eight. The energy numerator is eight, so E=one. The second-moment numerator is sixteen, giving second moment two and variance one. The raw norm changes the representative, while the normalized populations remain one half each. Dividing by a raw coefficient count instead of the squared norm would fail this calculation.

**2. A perfect residual in the wrong sector.** Evaluate the same diagnostics for z=(0,1). Can zero variance certify ground-state accuracy?

**Explanation.** E=two, second moment=four and variance=zero. The state is an exact excited eigenstate, with zero ground probability. The residual correctly certifies the eigenvalue equation but supplies no ground-sector identification. This is a counterexample to an unconditional “small variance means ground state” rule.

**3. A gap bound with a known energy origin.** Use H=diag(-3,1) and z=(2,1). Compute energy, leakage and the gap bound.

**Explanation.** N=five and E=(-12+1)/5=-11/5. The ground probability is four fifths and leakage is one fifth. The gap is four, so (E+3)/4=one fifth, exactly the leakage. The variance is sixteen times four twenty-fifths, or 64/25. An approximate upper ground energy cannot safely replace minus three in this equality. For example, using the valid upper estimate minus two gives a supposed leakage bound of minus one twentieth, an immediate contradiction.

**4. Check a directional derivative.** Take H=diag(0,2), z=(1,1) and eta=(1,-1). Find the first two derivatives at t=zero.

**Explanation.** The path energy is one minus 2t divided by (one plus t squared). Its first derivative at zero is minus two and its second derivative is zero. The residual at the starting point is (-1,1), which gives the same first derivative through the complex-direction formula. At t=one the path reaches the ground direction, and at minus one it reaches the excited direction; the path never becomes the zero vector.

**5. A missing path state.** Instead set eta=-z and evaluate at t=one. What should a visualization show?

**Explanation.** The raw path vector is zero, its squared norm is zero and no normalized energy or variance exists there. An unavailable marker and an explanation are appropriate. Connecting neighboring finite values through an invented zero-energy point would add information the model does not define.

**6. A Gaussian away from the optimum.** Set alpha to twice its optimal oscillator value. Find the energy and variance in oscillator units.

**Explanation.** The kinetic contribution is one half hbar omega and the potential contribution is one eighth hbar omega. Their sum is five eighths hbar omega, above the exact ground energy. Substitution into the variance expression gives nine thirty-seconds of hbar squared omega squared. Correct dimensions provide an additional check on both answers.

## 14. What is prepared and what comes next

Use the fixed concept comparisons to distinguish energy from variance, then predict the outcome before changing the saved free experiment. Compare ground and excited states, a degenerate ground space, a tiny positive variance and a path that reaches the zero vector. Worker calculations keep exact and interval certificates available alongside display coordinates. The entry point is **Book → Interactive Book**.

Continue with nonorthogonal Ritz spaces to distinguish changing coordinates from enlarging an approximation space. Later many-body, tensor-network and quantum-information material must supply its own assumptions and convergence or resource claims. A finite calculator and one Gaussian example do not complete those subjects. The variational principle becomes useful for new research when its exact guarantee, restricted search and evidence of approximation quality are all visible.
