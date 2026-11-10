## 1. A small coefficient can control a whole boundary condition

Imagine a profile pinned at both ends. Most of it changes slowly, but a narrow region near one end must turn sharply to meet the pin. Removing a small smoothing term can erase that turn entirely. The term is small as a coefficient; its product with a large derivative need not be small.

This chapter studies a solvable boundary-value problem so that every approximation can be compared with an analytic solution. We will distinguish pointwise accuracy, accuracy over the entire interval, derivative accuracy and a small differential-equation residual. These are different questions.

The preceding chapters dealt with a nonlinear equilibrium and a finite integral. Here the difficulty comes from a singular spatial limit rather than increasing the number of terms in a series.

## 2. State the equation, signs and boundary data

Our dimensionless problem is

$$
\varepsilon u''(x)+a u'(x)=f,\qquad 0<x<1,\quad
u(0)=A,\quad u(1)=B,\qquad \varepsilon>0.
$$

Here a and f are constant, and primes denote derivatives in x. The interactive family uses rational inputs with 0<ε≤16, |a|,|f|,|A|,|B|≤16, a probe in [0,1] and 4–256 equal grid panels. Zero drift a=0 has its own formula. Zero ε is a limiting equation, not a legal numerical input.

Be careful with transport conventions. Multiplying our equation by −1 gives the common stationary form −εu″+v u′=s with v=−a and s=−f. A positive a in our convention places the decaying layer on the left. Do not infer the layer side from a remembered sign convention.

## 3. Why setting ε to zero loses information

For nonzero a, deleting the highest derivative gives a first-order equation a u′=f. Its solution has one integration constant, while the original problem has two prescribed end values. Unless those values happen to fit its slope, it cannot satisfy both.

For a>0, the outer solution selected by the far boundary is

$$
 u_{\mathrm{out}}(x)=B+\frac{f}{a}(x-1).
$$

Its left value is B−f/a. Define D=B−A−f/a. Then u_out(0)−A=D. If D is independent of ε and nonzero, the missing condition produces an order-one error at the left endpoint for every ε. This is not repaired by announcing that ε is “very small.”

## 4. Use one orientation for both drift signs

Let b=|a|. For a>0 set y=x, near value N=A and far value F=B. For a<0 set y=1−x, N=B and F=A. Reflection changes the sign of the first derivative, so the transformed equation always has positive drift:

$$
 \varepsilon v_{yy}+b v_y=f,\qquad v(0)=N,\quad v(1)=F.
$$

Write k=f/b, D=F−N−k and δ=ε/b. The coordinate y measures distance from the layer boundary, not always distance from x=0. The physical derivative gains a minus sign under reflection; the second derivative does not. The calculator retains these signs when reporting probe derivatives.

## 5. Derive the exact solution first

Let w=v_y. Solving εw_y+bw=f gives a constant contribution k and a decaying exponential. Integrate once and impose both end values:

$$
 v(y)=N+ky+D\frac{1-e^{-y/\delta}}{1-e^{-1/\delta}}.
$$

The denominator is positive for every finite δ>0. This formula is valid even when δ is not small. At y=0 the ratio is exactly zero; at y=1 it is exactly one. The implementation returns exact rational boundary values there instead of subtracting nearly equal floating-point quantities.

For the basic example ε=1/10, a=1, f=0, A=0, B=1, this becomes (1−e^(−10x))/(1−e^(−10)). At x=1/20 the stretched coordinate is 1/2, and the value is approximately 0.39349. The whole transition is visible near the left end even though the outer solution is the constant 1.

The outer value is 1 at all three positions below. Near the boundary it is far from the exact answer, while the composite already has a small error. The maximum composite error occurs at the far endpoint, rather than in the middle of the layer. These rounded decimals illustrate the comparison; use the interval tables for certified values.

| x | Exact solution | Composite | Exact minus composite |
|---|---:|---:|---:|
| 0.05 | 0.3934872046 | 0.3934693403 | 0.0000178643 |
| 0.1 | 0.6321492584 | 0.6321205588 | 0.0000286995 |
| 0.5 | 0.9933071491 | 0.9932620530 | 0.0000450961 |

## 6. Find the layer width by dominant balance

Suppose the profile changes by order one over a distance ℓ. Its first derivative scales as 1/ℓ and its second as 1/ℓ². To retain both derivative terms, balance ε/ℓ² with b/ℓ:

$$
 \ell\sim\delta=\frac{\varepsilon}{b},\qquad
 X=\frac{y}{\delta},\qquad v_{XX}+v_X=\delta k.
$$

For fixed b>0 and fixed data as ε→0, δ→0 and δk is small. The leading inner equation is v_XX+v_X=0. This scaling argument also tells us when the interpretation fails: small ε alone is insufficient if b is still smaller. For ε=0.1 and b=0.001, δ=100; there is no narrow layer relative to the unit interval.

## 7. Match the inner and outer descriptions

The outer profile in canonical coordinates is F+k(y−1). Its value as y approaches the near endpoint is F−k. The leading inner solution satisfies its near boundary condition and tends to that common value:

$$
 v_{\mathrm{in}}(X)=F-k-D e^{-X}.
$$

At X=0 this is N. At large X it approaches F−k. Matching does not mean forcing the leading inner approximation to satisfy the opposite boundary at X=1/δ. Each local approximation is designed for a different region.

The overlap is the constant F−k. There are points satisfying δ≪y≪1 only in the asymptotic regime δ→0. When δ is large, the formulas remain evaluable but this overlap interpretation is not justified.

## 8. Build a composite without double counting

Add the two descriptions and subtract their shared part:

$$
 v_{\mathrm{comp}}(y)=v_{\mathrm{out}}(y)+v_{\mathrm{in}}(y/\delta)-(F-k)
 =F+k(y-1)-D e^{-y/\delta}.
$$

The composite obeys v_comp(0)=N. Its far boundary residual is v_comp(1)−F=−Dq, where q=e^(−1/δ). It is a uniform approximation with a small boundary defect, not the exactly normalized solution.

For the basic example the composite is 1−e^(−10x). Its error is not inferred from a plot. We can derive it explicitly.

## 9. Prove a whole-interval error certificate

Subtract the composite from the exact solution:

$$
 v(y)-v_{\mathrm{comp}}(y)
 =Dq\frac{1-e^{-y/\delta}}{1-q},\qquad q=e^{-1/\delta}.
$$

The fraction increases from 0 to 1 over [0,1]. Therefore the maximum absolute error is attained at the far endpoint:

$$
 \|v-v_{\mathrm{comp}}\|_\infty=|D|q.
$$

This is an equality for this finite model. It is stronger than taking the largest error at a finite set of plot points. It is also conditional: the claim follows from this constant-coefficient equation and its boundary conditions, not from a universal rule that matched expansions always have exponentially small error.

## 10. Pointwise convergence can hide an order-one boundary error

The outer error is

$$
 v(y)-v_{\mathrm{out}}(y)
 =-D\frac{e^{-y/\delta}-q}{1-q},\qquad
 \|v-v_{\mathrm{out}}\|_\infty=|D|.
$$

For every fixed y>0 it tends to zero as δ→0, while at y=0 it remains −D. Thus the outer approximation converges pointwise away from the layer boundary but fails uniformly on the closed interval when D≠0. The exact endpoint limit and the limit reached by moving inward do not commute.

An average error can also become small while the worst error stays large. Squaring the outer-error formula and bounding its numerator by e^(−y/δ) gives an L² norm at most |D|√(δ/2)/(1−q). A small average loss need not certify a uniformly accurate boundary profile.

## 11. Small value errors and large gradients

Differentiate the exact canonical solution:

$$
 v_y=k+\frac{D}{\delta}\frac{e^{-y/\delta}}{1-q},\qquad
 v_{yy}=-\frac{D}{\delta^2}\frac{e^{-y/\delta}}{1-q}.
$$

With fixed nonzero D, the layer slope and curvature scale as 1/δ and 1/δ². The small coefficient ε multiplies a large curvature, so dropping that term near the boundary destroys the balance.

The composite derivative error has supremum |D|q/[δ(1−q)], attained at the near endpoint. This is a different certificate from |D|q for the values. Under reflection, u_x=−v_y and u_xx=v_yy; reporting unsigned slopes would conceal the direction of the sharp transition.

## 12. A zero interior residual need not solve the boundary problem

Both the outer solution and the composite have identically zero interior residual εv_yy+bv_y−f. The outer solution still misses the near boundary by D, and the composite misses the far boundary by −Dq. Differential residual alone does not determine the solution of a boundary-value problem.

For an AI surrogate or PINN, inspect boundary data, interior residual, value error and derivative error separately. A network sampled mostly outside the layer may look successful while missing the transition. The example provides a controlled test case; it does not prove that a particular architecture or optimizer will fail.

## 13. A small parameter does not always produce a visible layer

If D=0, the exact solution is N+ky and the outer and composite agree with it everywhere. The singular structure of the equation permits a layer, but these compatible data do not excite its exponential mode.

For a=1, f=1, A=0, B=1, D=0 and u=x for every positive ε. Compare this with a=1, f=1, A=B=0, where D=−1 and a layer remains. The sign of D reverses the correction; it does not change which side hosts the decaying mode.

## 14. Zero drift is a different degeneration

If a=0, divide by neither a nor b. Direct integration gives

$$
 u(x)=A+(B-A)x+\frac{f}{2\varepsilon}x(x-1).
$$

When f≠0 and ε→0, interior values generally grow like 1/ε. When f=0, the exact profile is the line joining the endpoints. The reduced equation at ε=0 is either inconsistent (0=f with f≠0) or underdetermined (0=0). It does not specify the same outer problem as nonzero drift.

A very small nonzero a is not rounded to zero in the calculator. The rational input preserves that distinction, and directed exponential intervals support the calculation despite cancellation in a naive formula.

## 15. Separate grid resolution from asymptotic validity

For m equal panels, h=1/m and h/δ=b/(mε). This ratio tells us how much stretched distance one physical grid cell crosses. The fraction of the exponential layer change occurring in the first cell is 1−e^(−h/δ).

The display reports whether mδ≥4, meaning at least four panels per e-fold distance. That is an explicit visualization heuristic, not a finite-difference convergence theorem. The flag δ≤0.1 is likewise a display convention for a relatively thin layer.

A physical 65-point drawing may miss a very thin layer. A separate 65-point inner-coordinate drawing samples 0≤X≤min(8,1/δ). These are complementary views. The inner window ending at X=8 does not mean the exponential is exactly zero there, and a uniform-error certificate does not depend on either drawing mesh.

## 16. Read numerical intervals and exact identities correctly

The arithmetic keeps rational parameters and uses directed dyadic enclosures for negative exponentials at a 192-bit output grid. Rational interval operations propagate rounding. That working grid is not a promise of 192 correct bits in the final solution: division by a small normalizer and large coefficients can widen the output interval.

Zero at the lower endpoint of an exponential enclosure can mean “smaller than the representable quantum,” not an exactly vanishing exponential. The classification still recognizes positive ε and nonzero drift. Exact boundary values, the zero-drift polynomial and the compatible-data identity are handled algebraically.

Independent references use the original signed-coordinate integrating-factor solution with high precision expm1, while the production model uses reflection and decaying exponentials. Numerical agreement is a check; the analytic identities and directed bounds explain the certificates.

## 17. Worked comparisons and explained exercises

1. **Basic left layer.** With ε=0.1, a=1, f=0, A=0, B=1, find δ, D and the uniform composite error. They are 0.1, 1 and e^(−10), approximately 0.00004540. The outer uniform error remains 1.
2. **Reverse drift.** Change a to −1 and retain the same end values. The near boundary is x=1, N=1, F=0 and D=−1. The layer moves to the right; the physical slope is positive because reflection reverses the negative canonical slope.
3. **Compatible data.** Set a=f=B=1 and A=0. D=0, so u=x. A thin nominal δ alone does not establish a nonzero layer amplitude.
4. **A residual trap.** For the first case, substitute u=1 into the differential equation. Its residual is zero, but u(0)=1 violates the prescribed 0. Boundary checks are indispensable.
5. **Resolve a thinner layer.** With δ=0.01 and m=4, h/δ=25. Almost the whole exponential adjustment lies in one cell. With m=256, h/δ=0.390625; there are 2.56 panels per e-fold, still below the display target 4. Neither flag proves a solver's error.
6. **Weak drift.** Take ε=0.1 and a=0.001. δ=100, not a narrow layer. Applying a leading thin-layer interpretation solely because ε<1 is unjustified.
7. **Zero drift with forcing.** With a=0, f=1, ε=0.1, A=0, B=1, the midpoint is 1/2−1/(8ε)=−0.75. The negative dip is an exact consequence of the stated sign convention, not numerical instability.
8. **A derivative question.** Why can a term εu″ matter? At a layer of width δ=ε/b, εu″ and bu′ both scale as b²/ε for order-one amplitude. Their cancellation is the very mechanism that the reduced equation omits.

## 18. What this prepares, and what remains separate

Use the experiments to predict the side and width of a layer before viewing the exact profile. Then compare the fixed physical view with the stretched view, the outer error with the composite certificate, and the residual with both boundary defects. Change one input at a time and record which assumption changed.

This chapter covers a constant-coefficient linear boundary-value family. Variable coefficients, turning points, multiple or internal layers, nonlinear matching, WKB, and rigorous stability bounds for general PDE solvers require further work. The next perturbation chapter addresses a different nonuniform limit: secular terms and long-time phase error.

For background on layer scaling and matching, see [John K. Hunter, §4.3](https://math.ucdavis.edu/~hunter/notes/asy.pdf). For the definition of uniform asymptotic behavior, see [NIST DLMF §2.1(iv)](https://dlmf.nist.gov/2.1#iv). The equation and whole-interval certificates above are derived explicitly here; Hunter's illustrative equation includes a different zeroth-order term.
