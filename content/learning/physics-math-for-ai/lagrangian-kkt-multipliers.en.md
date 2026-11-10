## Put the inequality into the Lagrangian with a consistent sign

Consider min f(x)=(x−a)²/2 subject to x≤b. Write g(x)=x−b≤0 and choose λ≥0. The Lagrangian and its unconstrained infimum are

$$
L(x,\lambda)=\tfrac12(x-a)^2+\lambda(x-b),\qquad
\inf_xL=L(a-\lambda,\lambda)
=\lambda(a-b)-\lambda^2/2.
$$

For a feasible x, λg(x)≤0, so the dual function d(λ)=inf L is a lower bound on its objective. The infimum ranges over all real x, not only feasible x; confusing those domains changes the dual problem.

## Four conditions must hold together

KKT requires primal feasibility x≤b, dual feasibility λ≥0, stationarity x−a+λ=0, and complementary slackness λ(x−b)=0. The readout also uses the equivalent nonnegative slack product λ(b−x) for feasible pairs.

$$
x^*=\min(a,b),\qquad\lambda^*=\max(a-b,0).
$$

If a<b, the unconstrained center is feasible and λ*=0. If a>b, the boundary binds and λ*=a−b. At a=b the constraint is active with multiplier zero, so activity does not imply a strictly positive multiplier.

## Work the active example

Take a=2,b=1. The solution x*=1 has f*=0.5 and λ*=1. The unconstrained Lagrangian minimizer a−λ also equals one, giving d(1)=0.5. Stationarity, complementarity and primal-dual gap vanish.

Keep x=1 but set λ=0. Complementarity still vanishes because the constraint is active. Stationarity is now −1 and d(0)=0, leaving gap 0.5. Thus one vanishing residual cannot stand in for the entire KKT system. Blue shows the feasible objective window, while orange shows L over an unconstrained window. The orange minimizer can lie outside the primal feasible region for a nonoptimal multiplier.

## Which assumptions make KKT an optimality statement?

This quadratic is convex and differentiable, the inequality is affine, strict feasibility is available by choosing x<b, and the optimum is finite. Slater's condition ensures existence of optimal multipliers here. Convexity makes any feasible KKT pair sufficient; constraint qualification is needed for general necessity and multiplier-existence claims, not to rescue an already valid convex certificate.

Without convexity, stationarity may describe a maximum: min −x² on [−1,1] has an interior stationary point at zero but smaller values at the endpoints. Without a suitable qualification, even a convex optimum may have no KKT multiplier, as the next unit's x²≤0 example shows.

## Exercise and explained answer

Set a=0,b=1,x=0,λ=0. Slack is one, complementarity zero, stationarity zero and gap zero. The inactive constraint exerts no multiplier force. Then set a=b=1 and keep x=1,λ=0: the constraint becomes active while its multiplier remains zero.

## Research connection

Multipliers can encode sensitivity under appropriate perturbation and regularity assumptions, but the sign convention, constraint scaling and objective scaling affect their numerical values. When reporting a solver result, check feasibility, stationarity, complementarity and an actual bound separately. A small training gradient or an active constraint alone is not a global certificate.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
