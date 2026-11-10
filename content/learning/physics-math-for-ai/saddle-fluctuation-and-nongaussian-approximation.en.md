## 1. The deepest point does not determine the integral

Imagine two bowls with the same depth. One is narrow and the other broad. A small amount of material settles near the bottom of either bowl, but the broader bowl accommodates more material close to that depth. An exponentially weighted integral has the same distinction: the smallest action determines the largest local weight, while the surrounding volume determines how much contributes.

In the preceding Euclidean-kernel chapter the action was quadratic, so Gaussian integration gave the entire answer. We now add a quartic position cost. Finding the best path is still important, but it no longer performs the integral.

We will derive the Gaussian volume factor, the first correction beyond it, a useful bound in a special case, and the need to include multiple minima. We begin with finite-dimensional integrals. Gaussian integration, matrix inverses and elementary Taylor expansion are sufficient prerequisites.

## 2. Keep the different mathematical objects separate

A stationary point is a configuration where the action has zero derivative. For a positive Euclidean weight, a nondegenerate minimum supports a local Gaussian approximation. The term saddle-point method is conventional; a maximum of the action does not become a positive Gaussian contribution by taking an absolute determinant.

| Object | Meaning | Caution |
|---|---|---|
| Minimum path x* | Most highly weighted interior configuration at fixed endpoints | Generally not the mean of a non-Gaussian law |
| Action Hessian H | Curvature of S around a specified configuration | Not the Hamiltonian operator |
| Inverse R=H⁻¹ | Curvature inverse, independent of ℏ at x* | Gaussian covariance is ℏR |
| Gaussian kernel K_G | Minimum-action exponential times fluctuation volume | Not the exact quartic kernel |
| First coefficient c₁ | Coefficient of ℏ after dividing by K_G | Not a probability or a complete error bound |

All comparisons below state which physical parameters, endpoints and grid are fixed. In particular, reducing ℏ is different from refining the time grid.

## 3. Specify the finite action before approximating it

Let N≥2 be the number of time intervals, δ=τ/N their duration, and d=N−1 the number of integrated coordinates. Fix x₀=a and x_N=b. Take m,τ,ℏ>0, ω²≥0 and λ≥0. Our convention puts a factor of one quarter in the quartic potential.

$$
V(x)=\frac{m\omega^2x^2}{2}+\frac{\lambda x^4}{4},\qquad
S_N(x)=\frac{m}{2\delta}\sum_{i=0}^{N-1}(x_{i+1}-x_i)^2
+\delta\left[\frac{V(a)}2+\sum_{i=1}^{N-1}V(x_i)+\frac{V(b)}2\right].
$$

The endpoint half weights match the preceding chapter. They are constant with respect to interior variations, but they remain part of the action value and kernel normalization. Dropping them preserves some derivative equations while changing the integral.

$$
K_N(b,a)=\left(\frac{m}{2\pi\hbar\delta}\right)^{N/2}
\int_{\mathbb R^d}e^{-S_N(x)/\hbar}\,d^d x.
$$

S_N has units of action, and λ has units of energy divided by length to the fourth power. K_N has units of inverse length. Dividing the joint interior weight by K_N defines a pinned probability density; K_N itself need not integrate to one over b.

The quartic restriction λ≥0 matters. A negative leading quartic coefficient makes the real-coordinate integral diverge. The separate double-well example later retains a positive quartic term.

## 4. Derive the nonlinear stationary equations

An interior coordinate x_i appears in two adjacent kinetic penalties and one interior potential. Differentiating both edges gives a discrete acceleration term. Differentiating the quartic term gives a cubic force.

$$
g_i(x)=\frac{\partial S_N}{\partial x_i}
=\frac{m}{\delta}(2x_i-x_{i-1}-x_{i+1})
+\delta(m\omega^2x_i+\lambda x_i^3),\qquad
g_i(x^*)=0.
$$

At N=2 the single interior coordinate touches both fixed endpoints. Their contributions add; one endpoint must not overwrite the other. At λ=0 these are the linear harmonic equations from the preceding chapter. At λ>0 a matrix inverse alone generally cannot solve for x*.

The physical interpretation is a balance of smoothing and position cost. Neighboring time steps discourage a jagged path. The potential encourages positions near the origin. A path with positive endpoints can bend below both endpoint values.

## 5. Curvature is a matrix, not just a second derivative at one time

Differentiating again produces a symmetric tridiagonal Hessian. The quartic potential increases its diagonal according to the position about which we expand.

$$
H_{ii}(x)=\frac{2m}{\delta}+\delta m\omega^2+3\delta\lambda x_i^2,\qquad
H_{i,i+1}=H_{i+1,i}=-\frac m\delta,\qquad
H_{ij}=0\ \ (|i-j|>1).
$$

A diagonal-only picture misses coupling: neighboring fluctuations influence one another. If one time coordinate is pushed upward, the kinetic cost favors a corresponding response at nearby times. The inverse Hessian records these relationships.

H is the Hessian of the dimensional action S_N. The Gaussian precision in the exponent is Q=H/ℏ. Thus R=H⁻¹ and C_G=ℏR are different quantities. In this chapter C_G always denotes the Gaussian reference covariance, not a computed covariance of the full quartic law.

## 6. Why the main model has one minimum

For a fluctuation u with zero endpoints, the kinetic quadratic form is a sum of squared differences. It is positive unless all differences vanish; zero endpoints then force u=0. The nonnegative harmonic and quartic curvature can only add to it.

A convenient conservative estimate follows by writing each u_i as a sum of preceding differences and using Cauchy–Schwarz. Summing the resulting bounds gives Σu_i²≤N²Σ(Δu)². Therefore

$$
H(x)\succeq\mu I,\qquad
\mu=\frac{m}{\delta N^2}+\delta m\omega^2>0,\qquad
\min(a,b,0)\le x_i^*\le\max(a,b,0).
$$

The uniform curvature bound makes S_N strongly convex. Its growth at large interior coordinates guarantees a minimum exists, and strong convexity makes it unique. The position bracket follows from a discrete maximum principle: an interior maximum above both endpoints and zero cannot balance the restoring terms. A negative interior minimum has the corresponding contradiction.

This bracket includes zero. Replacing it by the interval between a and b would incorrectly exclude paths that bend toward the origin.

## 7. A solver proposes a path; a residual checks it

A practical Newton step solves a tridiagonal linear system H(x)p=−g(x). A line search can shorten the step until the action decreases. A straight path is a reasonable initial candidate, but initialization and convergence criteria are computational choices rather than proofs of an exact answer.

Do not form an inverse merely to take a Newton step: solve the linear system. Do not confuse the last step size with the residual. A tiny update can reflect numerical rounding or poor scaling.

For the hand calculations we will manufacture exact stationary points and verify g=0. For a general numerical candidate x̂, we instead keep a bound on its distance from the true minimum. This distinction remains necessary even if a plot looks perfectly smooth.

## 8. Turn the residual into a position and action certificate

Let ρ be the largest absolute component of g(x̂). Strong convexity gives ||x̂−x*||₂≤||g(x̂)||₂/μ. Using ||g||₂≤√dρ and √d≤d produces an easily computed, conservative componentwise bound.

$$
\|x^*-\widehat x\|_\infty\le e=\frac{d\rho}{\mu},\qquad
0\le S_N(\widehat x)-S_N(x^*)\le\frac{d\rho^2}{2\mu}.
$$

For the action bound, minimize the strong-convexity quadratic lower model centered at x̂. Its minimum is S_N(x̂)−||g(x̂)||₂²/(2μ). The free straight-path kinetic action m(b−a)²/(2τ) supplies another lower bound because every potential term is nonnegative.

In the N=2 example below, x*=1/2 and μ=1 in the chosen reference units. The nearby candidate x̂=51/100 has residual 97651/2000000≈0.04883. The bound e≈0.04883 safely contains the actual position error0.01, but is deliberately loose. A certificate can be valid without being sharp.

Certificates must also be propagated when curvature is evaluated at x̂. A useful Hessian error bound and resolvent estimate are

$$
\eta=3\delta\lambda e(2\|\widehat x\|_\infty+e),\qquad
\left|(H(\widehat x)^{-1})_{ij}-R_{ij}\right|\le\frac{\eta}{\mu^2},
\qquad R=H(x^*)^{-1}.
$$

Indeed, the Hessian difference is diagonal and bounded by η, and A⁻¹−B⁻¹=A⁻¹(B−A)B⁻¹. Both inverse norms are at most 1/μ. If η<μ, the positive matrices H(x̂)±ηI bracket H(x*) and their determinants bracket detH(x*). If this sufficient condition fails, a determinant certificate has not been obtained; the failure is not zero uncertainty.

## 9. Expand the action around the minimum

Write x=x*+ξ, with ξ₀=ξ_N=0. The first derivative vanishes. Because the potential is quartic, the remaining finite-dimensional expansion stops exactly at fourth degree.

$$
S_N(x^*+\xi)=S_*+\frac12\xi^{\mathsf T}H\xi
+\delta\lambda\sum_{i=1}^d\left(x_i^*\xi_i^3+\frac{\xi_i^4}{4}\right),
\qquad S_*=S_N(x^*),\quad H=H(x^*).
$$

This identity contains three different contributions: the value at the minimum, the quadratic fluctuation cost, and the non-Gaussian remainder. Around a candidate x̂ rather than x*, an additional linear term g(x̂)ᵀξ remains. Removing it without accounting for the residual changes the problem.

A positive quartic potential does not make the shifted remainder nonnegative. Its cubic term can have either sign. That observation will determine where a simple global Taylor bound is valid.

## 10. Integrate the Gaussian volume factor

Discard the cubic and quartic remainder temporarily. Diagonalizing the positive Hessian turns the fluctuation integral into d independent Gaussian integrals. Each curvature eigenvalue contributes an inverse square root.

$$
\int_{\mathbb R^d}e^{-\xi^{\mathsf T}H\xi/(2\hbar)}\,d^d\xi
=\frac{(2\pi\hbar)^{d/2}}{\sqrt{\det H}},\qquad
K_G=\sqrt{\frac{m^N}{2\pi\hbar\,\delta^N\det H}}\,e^{-S_*/\hbar}.
$$

The second expression retains the N free-step prefactors and uses d=N−1. With λ=0 the expansion has no higher terms, so K_G equals the exact finite harmonic kernel. With λ>0 it is a Gaussian approximation.

A single scalar bowl S=S_*+Hξ²/2 has Gaussian integral proportional to H⁻¹ᐟ². Changing H from 1 to 4 with the same minimum action halves the integral and the fluctuation width. The peak exponential alone predicts neither change.

For a numerically located irrational minimum, the action, determinant and Gaussian covariance require enclosures or explicitly labeled approximations. An exact rational representation of a nearby candidate does not turn that candidate into the exact minimum.

## 11. Expose the small parameter by rescaling fluctuations

Set ξ=√ℏ y. The Gaussian law of y then has covariance R and no remaining ℏ dependence. The physical fluctuation ξ has covariance ℏR.

$$
y\sim\mathcal N(0,R),\qquad
B_3(y)=\delta\lambda\sum_i x_i^*y_i^3,\qquad
B_4(y)=\frac{\delta\lambda}{4}\sum_i y_i^4,\qquad
\frac{K_N}{K_G}=\mathbb E_G e^{-\sqrt{\hbar}B_3-\hbar B_4}.
$$

The identity for the ratio is exact when x* and H are exact. The small-ℏ approximation enters only when expanding the remaining expectation.

This scaling explains why the minimum path stays fixed while the reference band shrinks. Reducing ℏ narrows fluctuations; it does not solve a different stationary equation. Changing λ generally does change the minimum. At zero endpoints, however, x*=0 for every nonnegative λ, so the Gaussian Hessian is unchanged even as the non-Gaussian integral changes.

## 12. Compute moments rather than guessing a correction

For a centered Gaussian, odd total-degree moments vanish. Even moments are sums over pairings, each pair contributing its covariance. One way to derive the rule is Gaussian integration by parts: the expectation of y_i times a polynomial equals Σ_j R_ij times the expectation of that polynomial's derivative with respect to y_j. Repeating reduces the degree by two until the constant moment 1 remains.

The particular moments needed here are

$$
\begin{aligned}
\mathbb E_G y_i^4&=3R_{ii}^2,\\
\mathbb E_G(y_i^3y_j^3)&=9R_{ii}R_{jj}R_{ij}+6R_{ij}^3,\\
\mathbb E_G(y_i^4y_j^4)&=9R_{ii}^2R_{jj}^2+72R_{ii}R_{jj}R_{ij}^2+24R_{ij}^4.
\end{aligned}
$$

For the six-variable product, either one pair crosses between the two sites or all three do. There are 3×3=9 choices in the first case and 3!=6 in the second. For the eight-variable product, zero, two or four cross pairs give 9, 72 and 24 pairings.

These formulas remain valid at i=j. They then reduce to the familiar sixth and eighth moments 15R_ii³ and 105R_ii⁴. Replacing R by only its diagonal would lose the correlations in a coupled path.

## 13. Derive the first correction beyond the Gaussian

Expand the last exponential to the order needed after Gaussian expectation. The term linear in B₃ has odd degree and vanishes, but its square has even degree and survives.

$$
\frac{K_N}{K_G}=1+\hbar c_1+O(\hbar^2),\qquad
c_1=\frac12\mathbb E_G B_3^2-\mathbb E_G B_4
=\frac{(\delta\lambda)^2}{2}\sum_{i,j}x_i^*x_j^*
\left(9R_{ii}R_{jj}R_{ij}+6R_{ij}^3\right)
-\frac{3\delta\lambda}{4}\sum_i R_{ii}^2.
$$

The coefficient c₁ has reciprocal-action units, so ℏc₁ is dimensionless. The positive cubic-square contribution and negative quartic contribution compete. Dropping the cubic term simply because a Gaussian has zero third moment gives the wrong first correction.

This is a fixed-N asymptotic statement under the smooth, coercive, nondegenerate-minimum assumptions already established. One justifies it by separating a neighborhood of the minimum from the strongly suppressed exterior, rather than exchanging an infinite series with the entire integral without a remainder argument. Odd local terms cancel under the centered Gaussian.

An enclosure for c₁ only encloses this coefficient. It does not supply a numerical bound on all higher terms in K_N/K_G. The O(ℏ²) notation has not provided that missing constant.

## 14. Work through three one-coordinate stationary points

Take N=2 and m=τ=ω²=λ=1, so δ=1/2, and set a=0. The stationary equation and curvature become

$$
g(x)=\frac92x+\frac12x^3-2b,\qquad
b=\frac{9q+q^3}{4}\ \Longrightarrow\ x^*=q,\qquad
H(q)=\frac92+\frac32q^2.
$$

Choosing q first manufactures an exact stationary point without pretending that a general nonlinear equation has a rational solution. The main action is strongly convex, so verifying g(q)=0 identifies its unique minimum.

| b | x* | H | Cubic-square contribution | Quartic contribution | c₁ |
|---|---|---|---|---|---|
| 0 | 0 | 9/2 | 0 | −1/54 | −1/54 |
| 37/32 | 1/2 | 39/8 | 80/19773 | −8/507 | −232/19773 |
| 13/2 | 2 | 21/2 | 20/3087 | −1/294 | 19/6174 |

At q=1/2 the Gaussian inverse is R=8/39. Keeping the endpoint half potential gives S*=17277169/16777216. The cubic-square contribution is 80/19773, while the quartic contribution is −312/19773; their difference is−232/19773.

At q=2 the total coefficient is positive. This does not contradict the fact that adding a positive quartic cost reduces a kernel relative to the same model with that cost removed: K_G is a different reference, evaluated at the quartic model's own minimum and Hessian.

The moved remainder at q=1/2 also gives a useful counterexample. Substituting ξ=−1/2 into δλ(qξ³+ξ⁴/4) yields−3/128. Positivity of the original quartic potential has not made this shifted remainder nonnegative.

## 15. A minimum is a mode, not automatically a mean

In the one-coordinate example, the normalized pinned density has its mode at q. Its Gaussian approximation is symmetric about q. The cubic term makes the exact density asymmetric.

For any r>0, compare equally distant points on both sides of a positive stationary point q. The common quadratic and quartic terms cancel in their density ratio:

$$
\frac{p(q+r)}{p(q-r)}
=\exp\left(-\frac{2\delta\lambda q r^3}{\hbar}\right)<1
\qquad(q>0,\ \lambda>0).
$$

The left side of the mode has greater weight at every equal distance. Pairing positive and negative displacements in the mean integral shows that the exact mean is less than q in this one-coordinate case. The minimum remains q.

A sensible drawing would therefore distinguish the stationary marker, a symmetric Gaussian reference and the actual one-coordinate quartic density. A band computed from ℏR describes that Gaussian reference. It is neither a sampled path envelope nor a certified covariance of the full non-Gaussian chain.

## 16. Obtain a global ratio bound when both endpoints are zero

When a=b=0, the unique minimum is exactly zero. The shifted cubic term vanishes, and the Hessian is the harmonic one even if λ>0. Introduce W=Σy_i⁴≥0 and ε=ℏδλ/4.

$$
\frac{K_N}{K_G}=\mathbb E_G e^{-\varepsilon W},\qquad
M_1=\mathbb E_G W=3\sum_iR_{ii}^2,\qquad
M_2=\mathbb E_G W^2
=\sum_{i,j}\left(9R_{ii}^2R_{jj}^2+72R_{ii}R_{jj}R_{ij}^2+24R_{ij}^4\right).
$$

For every nonnegative scalar u, 1−u≤e⁻ᵘ≤1−u+u²/2. Taking expectations preserves the inequalities. Positivity of the exact ratio and e⁻ᵘ≤1 improve the raw endpoints:

$$
\max(0,1-\varepsilon M_1)\le \frac{K_N}{K_G}
\le\min\left(1,1-\varepsilon M_1+\frac{\varepsilon^2M_2}{2}\right).
$$

This is a finite-dimensional bound on the actual ratio, not just on its first coefficient. At ε=0 the ratio is exactly1. For large ε the interval can widen to nearly[0,1]; validity does not imply useful accuracy.

The condition is essential. In a general shifted problem B₃ can be negative, and the preceding nonnegative-u Taylor bound does not apply to its full remainder. Do not copy the zero-endpoint error bars to a nonzero-endpoint graph.

## 17. Check the zero-endpoint bound by hand

For the first row of the table, H=9/2 and R=2/9. The needed Gaussian moments are E[y⁴]=4/27 and E[y⁸]=560/2187. With ε=ℏ/8, substitution gives

$$
\max\left(0,1-\frac{\hbar}{54}\right)
\le\frac{K_2}{K_G}
\le\min\left(1,1-\frac{\hbar}{54}+\frac{35\hbar^2}{17496}\right).
$$

At ℏ=1 this is[53/54,17207/17496]. No sampling or decimal quadrature is needed to establish these rational endpoints. Dividing a measured difference by the Gaussian reference is important: an error in this ratio is not automatically the same absolute error in K₂.

The upper correction is proportional to ℏ², while the first reduction is proportional to ℏ. Keeping physical parameters fixed and decreasing ℏ therefore makes the interval narrower in a controlled way for this special model.

## 18. More terms need not mean a better answer

Consider a scalar action Hz²/2+λz⁴/4 with H>0. Gaussian moments produce the formal expansion coefficients

$$
a_n=\frac{(-\hbar\lambda/4)^n(4n-1)!!}{n!\,H^{2n}},\qquad
\left|\frac{a_{n+1}}{a_n}\right|
=\frac{\hbar\lambda(4n+1)(4n+3)}{4(n+1)H^2}.
$$

For fixed positive ℏλ the ratio grows without bound, so the terms eventually grow rather than tend to zero. This is not a convergent infinite sum. At a fixed truncation order, however, decreasing ℏ can make a finite-order remainder small.

A practical comparison records the truncation order together with the parameters and a relevant bound. Neither “keep every term” nor “the next term looks small” is a universal accuracy certificate. The finite Taylor inequality above gives an actual bound under its stated positivity condition; a generic asymptotic symbol alone does not.

## 19. Separate time refinement from semiclassical scaling

There are at least three independent changes: increase N at fixed τ, decrease ℏ at fixed action, or decrease λ at fixed ℏ. The first changes the finite quadrature rule and number of coordinates; the second narrows fluctuations about a fixed minimum; the third generally changes the minimum itself.

For zero endpoints and ω²=0, the free curvature inverse is R_ii=δi(N−i)/(mN). Use Σi²(N−i)²=(N⁵−N)/30, obtained by expanding N²Σi²−2NΣi³+Σi⁴. Summing the squared diagonal then gives an exact first-correction check:

$$
-c_1=\frac{3\delta\lambda}{4}\sum_{i=1}^{N-1}R_{ii}^2
=\frac{\lambda\tau^3}{40m^2}\left(1-\frac1{N^4}\right).
$$

The factor1−N⁻⁴ approaches1 from below. This statement concerns one finite-grid coefficient; it is not a theorem that every quartic observable has fourth-order discretization error.

Raw Hessian eigenvalues also change their grid scaling when N changes. Compare consistently normalized physical quantities rather than diagnosing a physical soft mode from an unscaled matrix eigenvalue on a different mesh.

## 20. What changes when there are several minima?

The main chain is strongly convex and has one minimum. To study several contributions separately, introduce a one-coordinate integral in fixed reference units:

$$
A_\kappa(z)=\frac{z^4}{4}-\frac{\kappa z^2}{2},\qquad
I(\hbar,\kappa)=\int_{\mathbb R}e^{-A_\kappa(z)/\hbar}\,dz,\qquad
A_\kappa'(z)=z(z^2-\kappa).
$$

| κ | Minima | Curvature there | Point requiring care |
|---|---|---|---|
| Negative | z=0 | −κ>0 | One nondegenerate minimum |
| Zero | z=0 | 0 | Gaussian approximation fails |
| Positive | z=±√κ | 2κ>0 | Both minima contribute; z=0 is a maximum |

For a general finite integral with several isolated nondegenerate minima, the leading Laplace approximation sums their local Gaussian contributions. Their action values set exponential weights, and their Hessians set volumes. Unequal actions do not in general give equal contributions.

This scalar example is not a computation of the complete double-well quantum path integral. It illustrates multiple-minimum summation and coalescence without claiming a tunneling rate or an instanton calculation.

## 21. Sum the two local contributions before judging accuracy

For κ>0 the two minima have equal action −κ²/4 and curvature2κ. Each supplies one local Gaussian integral. Adding both gives

$$
I_G(\hbar,\kappa)=2\sqrt{\frac{\pi\hbar}{\kappa}}\,
e^{\kappa^2/(4\hbar)},\qquad
\frac{I}{I_G}=1+\frac{3\hbar}{4\kappa^2}+O(\hbar^2)
\quad(\kappa>0\ \text{fixed}).
$$

Keeping only one minimum loses a factor of approximately 2 as ℏ becomes small. The central stationary point has negative curvature and cannot provide a normalized real Gaussian. Taking the absolute value of its curvature would introduce a spurious contribution.

The symmetric full law has mean zero; neither minimum is that mean. A single local Gaussian at one well also loses this global symmetry. A mean at zero does not mean that most mass is near zero.

The scalar first-correction formula follows from the sixth and fourth Gaussian moments:

$$
c_1^{\rm scalar}
=\frac{5(A'''(z_*))^2}{24H^3}
-\frac{A''''(z_*)}{8H^2},\qquad H=A''(z_*)>0.
$$

For the double well, A'''=6z*, A''''=6 and z*²=κ give3/(4κ²). For κ<0, z*=0 and H=−κ instead give −3/(4κ²). The positive and negative cases do not share the same local expansion.

The relevant dimensionless parameter is ℏ/κ². Minima that are far apart on a plot can still overlap on the fluctuation scale if this ratio is not small.

## 22. At zero curvature, use the correct scale

At κ=0 the quadratic term disappears. The quartic integral remains finite, but a Gaussian formula dividing by √H is unavailable. Rescale z=ℏ¹ᐟ⁴u rather than z=√ℏ y:

$$
I(\hbar,0)=\hbar^{1/4}
\int_{\mathbb R}e^{-u^4/4}\,du.
$$

The width is proportional to ℏ¹ᐟ⁴, not ℏ¹ᐟ². Reducing ℏ by 16 therefore halves this quartic width; it would divide a fixed-curvature Gaussian width by 4.

A useful drawing shows a broad quartic bottom rather than inventing an infinitely wide Gaussian probability. At any fixed positive ℏ, the scalar integral stays finite as κ crosses zero. The change in the stationary-point pattern is not by itself a thermodynamic phase transition.

This example explains a failure mechanism: when quadratic curvature becomes too small relative to the retained higher terms, local Gaussian reasoning stops being a uniformly useful approximation.

## 23. Numerical integration is a separate source of evidence

The one-coordinate examples can be checked by deterministic quadrature. This is practical because only one integration coordinate remains. It is not an exact quadrature of the general N−1 dimensional quartic chain.

For positive κ, factor out the minimum-action exponential so the integrand does not grow exponentially large:

$$
J=\int_{\mathbb R}e^{-(A_\kappa(z)-A_{\min})/\hbar}\,dz,\qquad
\log I=-\frac{A_{\min}}{\hbar}+\log J.
$$

The shifted integrand is at most 1. A grid must still resolve each narrow peak and cover its tails. Two grids can agree while both miss a peak; grid agreement alone is not a certified error interval. Nor does increasing decimal precision remove a domain-truncation error.

We used independent exact Fraction identities for algebra and 80-digit Decimal quadrature for selected comparisons. These are different forms of evidence. The zero-endpoint Taylor bound is algebraic; the general shifted and double-well decimal comparisons are not advertised as certified integral bounds.

## 24. Reproduce the small cases with exact Python arithmetic

This exercise uses only the Python standard library. Gaussian integration by parts computes moments recursively, so the cubic-square term is obtained from a sixth moment rather than guessed from the sign of the potential. Repeated subproblems are cached.

The code verifies three exact stationary points, computes a rigorous zero-endpoint ratio bracket, keeps the endpoint half potential in S*, and checks a coupled two-coordinate Gaussian inverse. It does not implement a general nonlinear stationary solver or integrate the full quartic chain.

~~~python
from fractions import Fraction as F
from functools import lru_cache

def moment(R, powers):
    @lru_cache(None)
    def evaluate(e):
        if sum(e) == 0:
            return F(1)
        if sum(e) % 2:
            return F(0)
        i = next(i for i, degree in enumerate(e) if degree)
        remaining = list(e)
        remaining[i] -= 1
        total = F(0)
        for j, count in enumerate(remaining):
            if count:
                lower = remaining.copy()
                lower[j] -= 1
                total += count * R[i][j] * evaluate(tuple(lower))
        return total
    return evaluate(tuple(powers))

def correction(R, q, delta, lam):
    d = len(q)
    cubic = F(0)
    quartic = F(0)
    for i in range(d):
        powers = [0] * d
        powers[i] = 4
        quartic -= delta * lam * moment(R, powers) / 4
        for j in range(d):
            powers = [0] * d
            powers[i] += 3
            powers[j] += 3
            cubic += (delta * lam)**2 * q[i] * q[j] * moment(R, powers) / 2
    return cubic, quartic

delta = F(1, 2)
lam = F(1)
for q in [F(0), F(1, 2), F(2)]:
    b = (9*q + q**3) / 4
    assert F(9, 2)*q + q**3/2 - 2*b == 0
    H = F(9, 2) + F(3, 2)*q*q
    c3, c4 = correction([[1/H]], [q], delta, lam)
    print(f"q={q} H={H} c3={c3} c4={c4} c1={c3+c4}")

R0 = [[F(2, 9)]]
epsilon = F(1, 8)  # hbar=1, delta=1/2, lam=1
M1 = moment(R0, (4,))
M2 = moment(R0, (8,))
lower = max(F(0), 1-epsilon*M1)
upper = min(F(1), 1-epsilon*M1+epsilon**2*M2/2)
print(f"bounds={lower},{upper}")

q, b = F(1, 2), F(37, 32)
candidate = F(51, 100)
residual = F(9, 2)*candidate + candidate**3/2 - 2*b
assert abs(candidate-q) <= abs(residual)  # d=1, mu=1
print(f"residual={residual}")

def V(x):
    return x*x/2 + x**4/4

def action_one(x):
    return x*x + (b-x)**2 + delta*(V(x)+V(b)/2)

minimum_action = action_one(q)
for offset in [F(1, 4), F(1, 2), F(2)]:
    assert action_one(q+offset)-action_one(q-offset) == 2*delta*lam*q*offset**3
shift = F(-1, 2)
assert delta*lam*(q*shift**3+shift**4/4) == F(-3, 128)
print(f"S*={minimum_action}")

# N=3, time=3, mass=lam=1, omegaSquared=0.
# Endpoints 5/8 and interior positions (1/2, 1/2).
path = [F(5, 8), F(1, 2), F(1, 2), F(5, 8)]
for i in [1, 2]:
    assert 2*path[i]-path[i-1]-path[i+1]+path[i]**3 == 0
diagonal = F(11, 4)
determinant = diagonal**2 - 1
H = [[diagonal, F(-1)], [F(-1), diagonal]]
R = [[diagonal/determinant, 1/determinant],
     [1/determinant, diagonal/determinant]]
for i in range(2):
    for j in range(2):
        assert sum(H[i][k]*R[k][j] for k in range(2)) == F(i == j)
print(f"N=3 det={determinant} R11={R[0][0]} R12={R[0][1]}")
~~~

The first three total coefficients are −1/54,−232/19773 and19/6174. The ratio bracket is [53/54,17207/17496]. The coupled inverse has R₁₁=44/105 and R₁₂=16/105. These exact checks should be correct before trusting a decimal curve.

## 25. Six prediction-led comparisons

Predict the effect before calculating. State whether you are examining an exact identity, a bound, an asymptotic coefficient or a decimal approximation.

| Change | Prediction to test | Explanation |
|---|---|---|
| Decrease ℏ at fixed N and physical action | Keep x* fixed; shrink Gaussian width as √ℏ | The stationary equation does not contain ℏ |
| Increase λ with both endpoints zero | Keep x*=0 and H fixed, but reduce the exact kernel ratio | The quartic remainder changes beyond the unchanged Gaussian reference |
| Move b from 37/32 to 13/2 in the N=2 case | Change the sign of c₁ | The positive cubic-square contribution can dominate |
| Omit the cubic-square term, then restore it | Fail the nonzero-endpoint hand calculations | A zero third moment does not imply a zero squared-cubic contribution |
| Refine N at fixed τ with ω²=0 and zero endpoints | Approach λτ³/(40m²) in the first reduction coefficient | Time refinement changes a coefficient independently of small-ℏ scaling |
| Move κ through negative, zero and positive values | See one, degenerate and two minima | At zero curvature use quartic scaling; at positive κ sum both minima |

For a residual experiment, perturb the exact root 1/2 to 51/100 and compare its true displacement with the certified radius. For a drawing experiment, overlay the local Gaussian and normalized one-coordinate quartic shape around the same root. The reference covariance and the actual non-Gaussian spread need different labels.

## 26. Worked questions

1. **If two actions have the same minimum value, must their integrals agree?** No. Different Hessians give different nearby volumes. In one dimension, increasing curvature from 1 to 4 halves a Gaussian integral with the same minimum value and measure.

2. **Why does reducing ℏ not move the main model's stationary path?** The equation g(x*)=0 is obtained from S_N, which contains no ℏ. The factor1/ℏ changes concentration. Varying λ changes S_N itself and generally moves x*.

3. **Can a tiny Newton step certify an exact minimum?** No. Inspect the residual and a curvature lower bound. The certificate dρ/μ bounds position error; a small update by itself can result from rounding or numerical scaling.

4. **Why does the cubic term contribute when its Gaussian expectation is zero?** The first correction contains one half of E[B₃²], a sixth-degree even moment. Its one-cross and three-cross pairings give the factors9 and6. E[B₃]=0 does not remove this term.

5. **Does λ≥0 guarantee K_N/K_G≤1 for arbitrary endpoints?** No. The shifted cubic remainder can be negative. The q=2 hand case even has positive first coefficient 19/6174. The simple upper bound 1 applies to the zero-endpoint positive-remainder case.

6. **If c₁ is enclosed accurately, is the full kernel error certified?** No. It certifies that coefficient's uncertainty, not the discarded higher-order terms. A bound for the actual ratio requires a separate remainder argument, such as the special nonnegative-quartic Taylor bound.

7. **Why does a one-minimum approximation miss the symmetric double-well integral at small ℏ?** Both minima have the same action and curvature. Each contributes the same leading Gaussian volume, so taking one loses approximately a factor of 2. The negative-curvature central maximum is not a third positive Gaussian contribution.

8. **Does zero Hessian curvature at κ=0 mean the integral diverges?** No. The positive quartic term still confines the integral. Its width scales as ℏ¹ᐟ⁴. The failed Gaussian approximation must be replaced by an appropriate local scaling, rather than interpreted as a divergent exact law.

## 27. Connect this calculation to AI without importing false guarantees

A Laplace approximation to a finite statistical model has the same basic structure: expand a negative log weight near a mode and retain its Hessian. An evidence integral depends on the local volume as well as the best objective value. Two fitted modes with the same loss can therefore contribute different amounts to the normalization.

In the statistical analogy, define the negative joint log weight, prior and integration measure before assigning a temperature-like small parameter. The physical ℏ in our benchmark has units of action; identifying an ML tuning parameter with it requires a declared rescaling. A mode-based Gaussian covariance is an approximation to posterior uncertainty, not an automatic calibration guarantee.

Neural objectives need not have the uniqueness or strong-convexity properties proved for our chain. Symmetries and poorly identified parameters can create flat directions, while several modes can invalidate a single local fit. Adding a ridge or taking an absolute Hessian determinant changes the calculation and needs a stated interpretation. A coordinate change also changes a density and its measure Jacobian; the determinant alone is not a coordinate-free evidence score.

The benchmark is useful for checking whether an implementation distinguishes optimization, curvature estimation, normalization and neglected higher-order behavior. It does not prove that Laplace approximations work for an arbitrary trained network.

## 28. State the limits before continuing

The exact action and polynomial identities belong to a finite grid. The general chain's Gaussian approximation and first coefficient do not evaluate its complete quartic integral. The certified ratio bracket requires zero endpoints. The scalar double well illustrates multiple minima and degeneracy; it does not supply a quantum tunneling calculation. Decimal quadrature checks are not certified tail or interval analyses.

The next path-integral treatment concerns real-time phases and the classical limit, where oscillation and stationary phase introduce different issues. Continuum interacting limits, multiple quantum classical paths, instantons and renormalization require additional analysis.

For background, see [Wipf, Chapter 6 §6.3](https://www.tpi.uni-jena.de/~wipf/lectures/pfad/pfad6.pdf#page=5) on semiclassical expansions, [Tong, Statistical Field Theory §2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2) on fluctuations around stationary configurations, and [Grosche's introduction](https://arxiv.org/pdf/hep-th/9302097#page=4) on the classical exponential and fluctuation prefactor. The finite quartic-chain identities and hand examples here are derived with the conventions stated in this chapter.
