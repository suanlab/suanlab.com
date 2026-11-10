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

## Exercise: identify the missing information

Take f=2 and a=b=0. The solution is x(1-x), so its midpoint value is one quarter and its second derivative is minus two. Now add 3x+1: the differential equation is unchanged but both endpoint conditions fail. For the time equation set lambda one and y0 one; at t=1/2 the state is exp(-1/2). Explain why imposing a second endpoint value on this first-order trajectory can be incompatible rather than supplying a freely selectable second constant.

Further reading: [MIT differential-equations course](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). The paired examples and residual checks here are derived directly.
