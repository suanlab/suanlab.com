## 1. A finite answer can have a divergent recipe

In the previous chapter, a Taylor expansion stopped converging outside a finite disk even though the real equilibrium remained unique. We now study a more striking example: the underlying integral is positive and finite, but its expansion about zero diverges at every positive coupling.

This does not make every finite approximation useless. At weak coupling, the first few terms can be excellent, while adding too many terms makes the answer worse. The goal is to explain that behavior with explicit assumptions, a proved remainder bound and an independently enclosed integral.

Think of estimating the average response of many fluctuations. Solving for the most likely configuration is not enough: all configurations contribute to the average. In our symmetric model the energy minimum stays at zero while the integral changes with the coupling.

## 2. Define the object before expanding it

Let X be a standard normal variable. Define the normalized finite integral

$$
Z(g)=\mathbb E\!\left[e^{-gX^4/4}\right]
=\frac1{\sqrt{2\pi}}\int_{-\infty}^{\infty}
e^{-x^2/2-gx^4/4}\,dx,\qquad g\ge0.
$$

Here g is dimensionless. Our quartic convention is g x⁴/4; changing it to g x⁴/4! changes the expansion coefficients. Keep this normalization when comparing formulas from other sources.

Since the extra exponential lies between zero and one, Z(0)=1 and 0<Z(g)≤1. The integral is finite because the Gaussian already provides an integrable upper bound. For positive g, the quartic term suppresses large fluctuations further.

This is an ordinary one-variable integral, often used as a zero-dimensional field-theory model. It is not a definition of a continuum path-integral measure, a time-evolution calculation or a model of multiple wells. Negative and complex couplings are outside this experiment.

## 3. Gaussian moments supply every coefficient

The even Gaussian moments follow from integration by parts:

$$
\mathbb E[X^{2m}]=(2m-1)!!,
\qquad \mathbb E[X^{2m}]=(2m-1)\mathbb E[X^{2m-2}].
$$

The boundary term vanishes because a polynomial times the Gaussian tends to zero at infinity. Starting with E[1]=1 gives E[X⁴]=3, E[X⁸]=105 and E[X¹²]=10395.

These moments describe how strongly rare large fluctuations contribute to high powers. They grow quickly. That growth will eventually defeat any fixed positive coupling, however small it initially looks.

## 4. Construct a formal perturbation series

Expand the quartic exponential to a finite order and average each term. The resulting coefficients are

$$
c_n=\frac{(-1)^n}{4^n n!}(4n-1)!!,
\qquad S_N(g)=\sum_{n=0}^{N}c_ng^n.
$$

The n=0 double factorial is understood as (−1)!!=1. An independent factorial formula is c_n=(−1)ⁿ(4n)!/[2⁴ⁿ(2n)!n!].

The first terms are

$$
Z(g)\sim1-\frac34g+\frac{105}{32}g^2
-\frac{3465}{128}g^3+\cdots.
$$

The symbol ∼ declares an asymptotic expansion, not a convergent sum. We will prove what each finite truncation means before discussing the infinite series.

## 5. Why not exchange an infinite sum and the integral?

For every fixed x, the exponential series converges. It is tempting to integrate the infinite series term by term and conclude that the resulting series equals Z. That exchange requires justification over the entire real line.

The absolute series reconstructs e^{+g x⁴/4}. Multiplying by the Gaussian gives e^{−x²/2+g x⁴/4}, which is not integrable for g>0. An integrable bound for the infinite absolute series is therefore unavailable in this argument.

The problem occurs in the large-x region. A small coupling does not make g x⁴/4 uniformly small over an unbounded domain. Finite-order integration remains legitimate, because every finite Gaussian moment exists and we can control its remainder explicitly.

## 6. A signed Taylor remainder gives the missing control

For u≥0, write the exponential remainder in integral form:

$$
r_N(u)=e^{-u}-\sum_{n=0}^{N}\frac{(-u)^n}{n!}
=\frac{(-1)^{N+1}}{N!}\int_0^u e^{-t}(u-t)^N\,dt.
$$

The integral is nonnegative. Replacing e^{−t} by its upper bound 1 gives

$$
0\le(-1)^{N+1}r_N(u)\le\frac{u^{N+1}}{(N+1)!}.
$$

This is stronger than inspecting the last retained term. It proves both the direction and a magnitude bound for the remainder for every nonnegative u, even where a truncated polynomial becomes negative.

This particular sign structure is essential. A general divergent expansion does not inherit the same bound merely because its coefficients alternate.

## 7. Average the remainder, not an unjustified infinite series

Substitute u=gX⁴/4 and average the finite remainder inequality. Let A_n=|c_n|. Then

$$
0\le(-1)^{N+1}\bigl(Z(g)-S_N(g)\bigr)
\le A_{N+1}g^{N+1}.
$$

Every quantity on the right is a finite Gaussian moment. For fixed N, this proves that the approximation error is O(gᴺ⁺¹) as g approaches zero through nonnegative values.

The calculator evaluates coefficients, terms, partial sums and this first-omitted-term bound as exact fractions. A floating-point cancellation between large partial terms is not used to certify an error.

## 8. Odd and even partial sums enclose the answer

An even N gives S_N≥Z; an odd N gives S_N≤Z. Consequently, any available odd and even pair supplies an enclosure:

$$
\max\!\left(0,\max_{n\text{ odd}}S_n\right)
\le Z(g)\le
\min\!\left(1,\min_{n\text{ even}}S_n\right).
$$

The positivity bounds zero and one remain valid even when a perturbative lower bound becomes negative or a higher-order upper bound becomes enormous. At g=0 every partial sum equals one exactly.

For a requested maximum retained order N, the calculator also computes the next coefficient. It needs that coefficient to bound the remainder of S_N. The adjacent S_{N+1} can therefore contribute to the reported asymptotic enclosure even though the selected partial sums stop at N.

## 9. Prove divergence at every positive coupling

The ratio of successive absolute evaluated terms is

$$
\frac{A_{n+1}g^{n+1}}{A_ng^n}
=g\frac{(4n+1)(4n+3)}{4(n+1)}
\longrightarrow\infty\qquad(g>0).
$$

The terms eventually increase and cannot approach zero. Thus the series diverges at every fixed positive g, and its radius of convergence is zero.

Nothing in this argument says that Z is infinite. We already proved that 0<Z≤1. The failure belongs to the infinite series representation at fixed coupling, not to the well-defined integral.

## 10. Keep the two limits separate

Fix N and send g to zero: the remainder bound tends to zero at the next power. Fix g>0 and send N to infinity: the terms eventually grow. Both statements are true simultaneously.

More generally, asymptotic coefficients alone need not identify a unique function. For g>0, a term such as e^{−1/g} is smaller than every fixed power as g→0. Adding it can leave the same formal power expansion while changing the function.

Here the integral definition, positivity and remainder estimates specify which function we mean. We are not assigning an arbitrary value to a divergent sum, and we are not implementing a general resummation procedure.

## 11. Choose a truncation rule and name what it optimizes

Our rule minimizes the proven first-omitted bound among available retained orders. For S_N the bound is A_{N+1}g^{N+1}. It compares omitted terms starting at index one, not the constant term.

The ratio above increases with its term index. We can therefore find the first order where the omitted-term sequence stops decreasing. Equality gives two equally small bounds; the calculator chooses the earlier retained order.

At weak coupling, the omitted index is roughly 1/(4g); the retained order is one less. This estimate is not an exact integer formula. More importantly, the order minimizing a bound need not minimize the actual error. A certificate can be conservative.

The experiment distinguishes the best bound within the order budget from the globally smallest first-omitted bound. If the global order lies beyond the maximum 32, it reports that limitation instead of calling the last available order globally optimal.

## 12. Work through g=1/10 by hand

The first partial sums and their next-term bounds are:

| Retained order | Partial sum | First-omitted bound |
|---|---:|---:|
| 0 | 1 | 0.075 |
| 1 | 0.925 | 0.0328125 |
| 2 | 0.9578125 | 0.0270703125 |
| 3 | 0.9307421875 | 0.032991943359375 |

The smallest listed bound belongs to S₂. A separate high-precision numerical evaluation gives Z≈0.94450180167417135. Thus the S₂ error is about 0.01331, smaller than its guaranteed bound 0.02707.

The S₃ error has the opposite sign. Adding one more term does not have to improve the answer. By much higher orders the partial sums can leave the physical interval (0,1] altogether while the integral remains inside it.

## 13. An independent integral enclosure

To compare with the series, compute the integral using a separate construction. Symmetry and Gaussian normalization give

$$
I(g)=\int_0^\infty e^{-S_g(x)}\,dx,
\qquad S_g(x)=\frac{x^2}{2}+\frac{gx^4}{4},
\qquad Z(g)=\frac{I(g)}{I(0)}.
$$

This ratio cancels the normalization constant without inserting a rounded square root of π. The numerical construction splits [0,16] into 256 panels and encloses the exponential on each panel.

On a panel x=a+δu with 0≤u≤1, separate e^{−S_g(a)} from e^{−ΔS(u)}. The polynomial ΔS has nonnegative coefficients. The even and odd exponential Taylor polynomials at degrees 32 and 33 therefore give integrated upper and lower bounds. Directed 192-bit arithmetic retains the rounding uncertainty.

These are local exponential expansions on bounded panels. They are not the divergent expansion of the full integral in g. Confusing those two expansions would obscure why the quadrature can remain useful at stronger coupling.

## 14. Bound the tail and combine certificates carefully

Convexity gives S_g(L+t)≥S_g(L)+S'_g(L)t. Integrating the resulting exponential majorant gives

$$
\int_L^\infty e^{-S_g(x)}\,dx
\le\frac{e^{-S_g(L)}}{S'_g(L)},\qquad L>0.
$$

The reference interval includes panel, rounding and tail uncertainties. An independently computed mpmath decimal and a Bessel-K expression serve as numerical cross-checks; the decimal itself is not a proved interval.

The display keeps three intervals separate: the raw quadrature interval, the asymptotic odd/even envelope and their intersection. The intersection legitimately uses both sources of information and must not be described as an independent quadrature result.

For extremely small coupling, a finite-order asymptotic bound can be much narrower than the quadrature resolution. A reference interval that straddles a partial sum does not establish zero error. Keep the exact next-term certificate and report the unresolved distance honestly.

## 15. Visual experiments and failure cases

At g=0.1, compare partial sums with the integral interval as order increases. Plot absolute term sizes and error bounds on logarithmic axes, while marking exact zeros separately. The downward-then-upward pattern explains useful truncation more clearly than a single final number.

Next use g=0.01 and then g=0.001. The useful order moves upward. In the latter case the globally best bound lies outside the available order budget. At g=1 or g=16, the positive finite integral survives even when ordinary perturbation theory gives weak information.

The integrand plot should show how increasing g suppresses the tails. Its vertical scale and normalization must be stated. A three-dimensional surface can show displacement, coupling and integrand height; it does not represent physical time or a continuum field configuration.

## 16. Connection to AI and statistical physics

An energy-based model needs a normalization integral, not merely its minimum. A perturbative baseline can reveal how a learned approximation behaves near a solvable Gaussian model and where higher-order corrections stop helping.

In this example even the curvature at the minimum stays equal to one for all g. Neither the minimum nor its Hessian alone determines the full normalization. Differentiating the integral gives a useful link to observables:

$$
\frac{dZ}{dg}=-\frac14\mathbb E\!\left[X^4e^{-gX^4/4}\right]<0,
\qquad \frac{d\log Z}{dg}=-\frac14\langle X^4\rangle_g.
$$

The first expectation uses the original standard normal distribution. The second uses the normalized density proportional to e^{−x²/2−gx⁴/4}. They are different averages. The first derivative at zero is −3/4, agreeing with the first perturbative coefficient. This connects normalization sensitivity to moments without treating an equilibrium location as an average.

Report the integral definition, coupling convention, retained order, remainder guarantee and independent reference uncertainty. When comparing methods, distinguish a small observed error from a certified bound and a budget-limited selection from a globally optimized rule.

This finite example prepares concepts used in correlation functions and statistical field theory. It does not by itself prove that a continuum path integral exists or that a resummation method succeeds for an interacting field theory.

## 17. Explained exercises

1. **Why does the integral remain finite at positive g?** Its integrand is bounded by the Gaussian integrand. Normalization then gives 0<Z≤1.
2. **Find the first correction.** E[X⁴]=3, so the coefficient is −3/4 and the evaluated correction is −3g/4.
3. **Which partial sums are upper bounds?** Even retained orders. The integrated exponential remainder has sign (−1)ᴺ⁺¹ for Z−S_N.
4. **Why does an alternating sign pattern not prove convergence?** At every fixed positive g the absolute term ratio eventually exceeds one and grows without bound; the terms do not tend to zero.
5. **What is optimized at g=0.1 by choosing N=2?** The first-omitted remainder bound, not a universal claim that N=2 minimizes the actual error for all inputs.
6. **Does a global recommended order beyond 32 mean the integral is unavailable?** No. It limits this truncation budget; the separate integral enclosure still exists.
7. **Can a reference interval containing S_N establish exact equality?** No. It may simply be too wide to resolve the error. At g=0 equality instead follows analytically.
8. **Why retain a separate label for the interval intersection?** It uses both quadrature and asymptotic information. Calling it an independent numerical reference would hide that dependence.

## 18. References and the next question

[NIST DLMF §2.1](https://dlmf.nist.gov/2.1) supplies asymptotic definitions; [§2.11](https://dlmf.nist.gov/2.11) discusses remainder control. [Mera, Pedersen and Nikolić](https://arxiv.org/pdf/1802.06034) use finite quartic models in the study of divergent-series resummation. Our normalization and certificates are stated explicitly above.

Next we turn to singular perturbations: a small coefficient can multiply the highest derivative, so setting it to zero can remove a boundary condition. Boundary layers require a change of variables rather than simply more terms in the same expansion.
