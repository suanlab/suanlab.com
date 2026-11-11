## A flow maps initial data into later states

Fix a differential equation and let S_t(a) denote its state at time t from initial value a. A central question is whether adding initial values before evolution gives the same result as adding evolved states. The lab compares S_t(a+b) in blue with S_t(a)+S_t(b) in orange, keeping the equation and its forcing fixed.

This is a test of superposition, not simply a test of whether two plotted curves look similar. At time zero both expressions agree by construction. Later agreement must come from the structure of the equation or a special parameter choice.

## Homogeneous linear evolution preserves addition

For y'=r y,

$$
S_t(a)=ae^{rt},\qquad S_t(a+b)=S_t(a)+S_t(b).
$$

The same idea extends to matrix exponentials for linear systems. Exact arithmetic gives zero superposition defect. The lab also reports the expanded floating-point subtraction, which can contain a tiny rounding remainder even when the analytic identity is exact. Numerical noise should not be mistaken for mathematical nonlinearity.

## Constant forcing makes the flow affine

For y'=r y+f with fixed f,

$$
S_t(a)=ae^{rt}+f\phi(r,t),\qquad
\phi(r,t)=\begin{cases}(e^{rt}-1)/r&r\ne0,\\t&r=0.\end{cases}
$$

Consequently, the superposition defect is minus f times phi. Each separately evolved solution has received the forcing once, so adding them counts it twice. The equation is nonhomogeneous linear, but its map from initial value to final state is affine. Differences of two solutions still satisfy the homogeneous equation.

Near r=0, directly subtracting one from exp(rt) loses digits. The implementation uses expm1 and a small-argument series, including the limiting value t. This connects analytic solution formulas to numerical stability rather than treating exact mathematics and reliable computation as unrelated tasks.

## Logistic evolution introduces state-dependent saturation

For y'=r y(1-y),

$$
S_t(a)=\frac{a}{a+(1-a)e^{-rt}}.
$$

The controls restrict a and b so each and their sum remain strictly between zero and one. This avoids an unintended finite-time pole in the negative-rate, above-one regime. Within the specified domain the solution exists for the displayed times and is generally nonlinear in the initial value.

At r=1, a=0.2, b=0.3 and t=1, evolving 0.5 differs from adding the evolutions of 0.2 and 0.3. At r=0 every state stays constant, so this special logistic parameter happens to preserve addition without making the general family linear.

## Flow composition does not imply linear superposition

Every well-defined autonomous flow satisfies Sₜ(Sₛ(a))=Sₜ₊ₛ(a) on its common domain, including the logistic flow. This follows from uniqueness after shifting the initial time. Additivity Sₜ(a+b)=Sₜ(a)+Sₜ(b) is a different property. For the forced linear equation, variation of constants gives Sₜ(a)=eʳᵗa+fI(r,t), so the corrected identity is Sₜ(a+b)=Sₜ(a)+Sₜ(b)−Sₜ(0). Equivalently, augment the state by a constant component and use the homogeneous matrix [[r,f],[0,0]].

For logistic dynamics set E=eʳᵗ, q=expm1(rt), s=a+b. Clearing denominators gives an exact nonlinear defect:

$$
S_t(s)-S_t(a)-S_t(b)=
-\frac{Eabq(2+sq)}{(1+sq)(1+aq)(1+bq)}.
$$

For positive a,b,s<1 and t>0 its sign is opposite the sign of r. Near r=0 it is −2abrt+O((rt)²), although three separately rounded flows can have a zero difference. The live analytic diagnostic now uses this factorization; the expanded floating difference remains visible as an arithmetic comparison.

| Family | Composition in time | Initial-data additivity defect |
|---|---|---|
| Homogeneous linear | yes | 0 |
| Constant forced affine | yes | −fI(r,t) |
| Logistic | yes on common domain | factored expression above |

```python
from decimal import Decimal as D, localcontext
from math import exp, expm1
with localcontext() as ctx:
    ctx.prec = 90
    for rate in ('-1', '-1e-20', '1e-20', '1'):
        r, a, b = D(rate), D('.2'), D('.3')
        E = r.exp()
        flow = lambda x: x*E/(1+x*(E-1))
        reference = flow(a+b)-flow(a)-flow(b)
        q, s = expm1(float(r)), .5
        stable = -exp(float(r))*.2*.3*q*(2+s*q)/((1+s*q)*(1+.2*q)*(1+.3*q))
        assert stable != 0 and abs(D(stable)/reference-1) < D('2e-14')
        print(rate, stable, 'relative error:', float(abs(D(stable)/reference-1)))
```

High-precision Decimal subtraction independently checks both signs and small rates. A flow's semigroup property is useful for solver consistency, but cannot establish linearity of a learned vector field.

## Exercise: locate the extra forcing

Take r=1 and f=1/2. Write out the three forced solutions and cancel their initial-value terms. The defect is -(e^t-1)/2. At r=0 it becomes -t/2. Explain why subtracting two forced solutions removes f, while adding them does not. Finally compare with the homogeneous preset and distinguish its analytic zero from any displayed expanded rounding remainder.

Further reading: [MIT linear and nonlinear differential equations](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). These three flow comparisons are derived explicitly.
