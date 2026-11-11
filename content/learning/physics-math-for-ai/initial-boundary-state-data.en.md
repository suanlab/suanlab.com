## An equation is only part of a model

An equation describes a relation between a state and its derivatives. It usually does not select one trajectory or spatial field until initial or boundary data are supplied. Before calculating, identify the independent variables, the state components, the differential order and the domain. A time trajectory and a spatial boundary-value problem can use similar algebra while asking different questions.

The lab deliberately places two problems next to each other. Changing the plot selector changes its independent variable: time t for the initial-value problem, position x for the boundary-value problem. The blue curve is the selected solution and orange its derivative with respect to that variable; they need not have the same physical units.

## One initial value for a first-order time equation

For a fixed nonnegative decay rate lambda, solve

$$
y'(t)=-\lambda y(t),\qquad y(0)=y_0,
\qquad y(t)=y_0e^{-\lambda t}.
$$

The equation alone admits every value of the multiplicative constant y0. The initial datum selects one. At lambda zero the state stays constant, and a negative initial value remains negative while approaching zero for positive decay. Both behavior and sign follow from the formula rather than from a prescribed animation.

A second-order time equation would instead require a larger state, typically position and velocity. Converting it into a first-order system makes the required initial data explicit. Differential order counts state information, although constraints can change the number of independent components.

## Two endpoint values for a spatial equation

On the unit interval consider a constant source f:

$$
-u''(x)=f,\qquad u(0)=a,\quad u(1)=b,
\qquad u(x)=a+(b-a)x+\frac f2x(1-x).
$$

Integrating twice introduces two constants. The endpoint values determine them, leaving a straight boundary connector plus a zero-endpoint parabola. When f=0 the solution need not vanish: nonzero boundary data still produce a nontrivial affine field. A second derivative can therefore satisfy the equation while an added affine term changes the boundaries.

## Verify equation and data separately

The readout reports the initial residual, both boundary residuals and the spatial equation residual. Checking only the differential equation would miss incorrect endpoint values. Likewise, satisfying endpoints with an arbitrary interpolating curve does not establish the source equation. This separation is useful for physics-informed losses, numerical solvers and inverse models, where different residual terms represent different requirements.

## Residuals become error bounds only through the boundary problem

For two solutions of −u″=f with identical Dirichlet data, their difference w has zero endpoints and −w″=0. Multiplying by w and integrating by parts gives ∫|w′|²=0; hence w is constant, and the endpoints force w=0. This is an energy proof of uniqueness, rather than a count of integration constants.

For a twice differentiable candidate v, set r=−v″−f and e=v−u. The zero-boundary Green kernel is G(x,s)=min(x,s)(1−max(x,s)). It is nonnegative, vanishes at the endpoints, and its first derivative jumps by −1 across x=s. Therefore −∂ₓ²G=δₛ and e(x)=∫G(x,s)r(s)ds when the boundary values agree. Since ∫G ds=x(1−x)/2≤1/8,

$$
\|e\|_\infty\le \max(|e(0)|,|e(1)|)+\frac18\|r\|_\infty.
$$

The first term comes from the affine interpolation of the two boundary errors. Thus a pointwise residual certificate plus boundary errors gives a solution certificate for this operator. A residual measured at finitely many training points does not certify the supremum between them; a narrow missed bump can invalidate that inference.

| Constant residual r | Zero-boundary error at x=1/2 |
|---|---:|
| 1/10 | 1/80 |
| 1 | 1/8 |
| 2 | 1/4 |

```python
from fractions import Fraction as F
for residual in (F(1, 10), F(1), F(2)):
    x = F(1, 2)
    # Integrate s(1-x) on [0,x], and x(1-s) on [x,1].
    integral = (1-x)*x*x/2 + x*(F(1, 2)-x+x*x/2)
    error = residual*integral
    assert error == residual/8
    print(residual, error)
```

The exact rational integration below attains the bound at the midpoint. Changing the domain length to ℓ changes the residual constant to ℓ²/8. Neural PDE losses need this domain and boundary information before a small training loss can be interpreted as a physical error.

## Exercise: identify the missing information

Take f=2 and a=b=0. The solution is x(1-x), so its midpoint value is one quarter and its second derivative is minus two. Now add 3x+1: the differential equation is unchanged but both endpoint conditions fail. For the time equation set lambda one and y0 one; at t=1/2 the state is exp(-1/2). Explain why imposing a second endpoint value on this first-order trajectory can be incompatible rather than supplying a freely selectable second constant.

Further reading: [MIT differential-equations course](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). The paired examples and residual checks here are derived directly.
