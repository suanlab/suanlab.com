## Feasibility comes before stationarity

For unconstrained differentiable convex optimization over all of R, a zero gradient certifies a minimum. A constraint changes the allowed directions. A point outside the feasible set cannot become a constrained solution merely by making its unconstrained gradient zero.

The lab minimizes f(x)=(x−a)²/2 on a nonempty closed interval C=[l,u]. Bounds are chosen with l≤0≤u, so the domain never becomes empty. Strong convexity makes the solution unique, including when the interval collapses to the singleton {0}.

## Derive the correct first-order condition

For differentiable convex f on a convex set, x* is optimal exactly when it is feasible and

$$
f'(x^*)(y-x^*)\ge0\quad\text{for every }y\in C.
$$

Necessity follows by moving a small positive distance toward any feasible y and taking a directional derivative. Sufficiency follows from f(y)≥f(x*)+f'(x*)(y−x*). The convex assumptions matter; a stationary point of a nonconvex objective need not minimize anything.

In one dimension the minimum first-order product is easy to compute. If g=f'(x)≥0, the smallest product occurs at y=l; if g<0 it occurs at y=u. A negative result supplies a concrete feasible descent direction. The readout leaves this certificate undefined for infeasible candidates.

## Normal cones explain nonzero boundary gradients

The outward normal cone is N_C(x)={n:n(y−x)≤0 for all y∈C}. Optimality is equivalent to

$$
0\in f'(x)+N_C(x).
$$

For a nondegenerate interval the cone is {0} in the interior, (−∞,0] at the lower endpoint, and [0,∞) at the upper endpoint. At a singleton it is all of R. Outside C the normal cone is empty. These cases explain why the required normal −f'(x) can cancel a nonzero boundary gradient.

## Work the clipped quadratic

The solution is x*=clip(a,l,u). With a=2 and C=[−1,1], the optimum is one. Its gradient is −1, and the required normal +1 belongs to the upper-endpoint cone. Every feasible displacement y−1 is nonpositive, so their product with gradient −1 is nonnegative.

Blue draws the unconstrained objective; orange overlays its feasible portion. The candidate and optimum are marked separately. Setting candidate x=2 makes the gradient zero, but it violates the interval and receives no feasible objective-excess certificate.

## Exercise and explained answer

Set a=0.5,C=[−1,1],candidate zero. Gradient is −0.5 and the minimum product is −0.5 at y=1, revealing descent. The unique optimum is 0.5. Then set both bounds to zero and a=2: the only feasible point is optimal despite gradient −2, because every feasible displacement vanishes.

## Research connection

Projection methods, constrained parameterizations and geometric optimization rely on feasible directions rather than unconstrained gradient tests alone. Record the constraint set, differentiability and convexity assumptions before treating a residual as a global certificate. Strong convexity adds uniqueness and distance bounds, but convexity alone need not provide those stronger conclusions.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
