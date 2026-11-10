## An optimization step that includes the penalty

A gradient step linearizes a differentiable objective. A proximal step instead solves a small optimization problem containing the penalty exactly. For a proper closed convex function f and η>0,

$$
P(y)=\operatorname{prox}_{\eta f}(y)
=\arg\min_x\left[f(x)+\frac{(x-y)^2}{2\eta}\right].
$$

The quadratic term makes the objective strongly convex, so the minimizer is unique even when f has corners or an extended-valued indicator. Think of a spring pulling toward y while the penalty resists. Increasing η weakens the spring relative to f.

## Optimality through a subgradient

The necessary and sufficient condition is (y−P(y))/η∈∂f(P(y)). The lab displays this required subgradient, the actual subdifferential interval and their distance. A small objective value alone does not establish stationarity.

For f(x)=λ|x|, the zero-point subdifferential is [−λ,λ]. If |y|≤ηλ the spring force can be balanced at zero. Outside that region the penalty has constant slope of the appropriate sign. Consequently

$$
P(y)=\operatorname{sign}(y)\max(|y|-\eta\lambda,0).
$$

With y=2, λ=0.5 and η=1, the answer is 1.5 and the proximal objective is 0.875. The threshold includes η; using λ alone changes the method when the step size changes.

## Projection and ridge shrinkage

For an indicator of [−λ,λ], the penalty is zero inside and infinity outside. The proximal point is the Euclidean projection clip(y,−λ,λ), independent of η. Here λ means box radius. Infinite values are omitted from the finite-height plot, rather than drawn as zero.

For f(x)=λx²/2, stationarity gives P(y)=y/(1+ηλ). At y=2, λ=1 and η=0.5 the result is 4/3. An explicit gradient step would give one, so these operations are different. Ridge shrinkage is an implicit update using the derivative at the new point.

## The Moreau envelope smooths the value function

Define Eηf(y) as the minimized proximal objective. Despite the absolute-value corner, this value function is differentiable for proper closed convex f in finite dimensions:

$$
\nabla E_\eta f(y)=\frac{y-P(y)}{\eta}.
$$

This derivative is with respect to the input y, not the optimization variable x. The displayed proximal objective value is the envelope value; the required subgradient is also its derivative. For the box indicator the envelope is squared distance to the box divided by 2η.

Monotonicity of subgradients at two proximal points gives ‖P(y)−P(z)‖²≤(P(y)−P(z))(y−z). This firm nonexpansiveness implies a Lipschitz constant of one, but not a strict contraction in every case. An identity map is an immediate counterexample to strictness.

## Exercise and boundary cases

Set λ=0 in each mode. Absolute-value and ridge penalties disappear, so their proximal maps become identity. The indicator mode instead becomes projection onto the singleton {0}; its normal cone is the whole real line. Explain why the same numerical parameter has different meanings across modes.

Check the envelope derivative by a centered finite difference in y away from a threshold, then across a threshold with a smaller step. The envelope derivative stays continuous even when the proximal map changes its formula. This is one reason proximal methods connect nonsmooth optimization to smooth value functions.

Further reading: [Parikh and Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).
