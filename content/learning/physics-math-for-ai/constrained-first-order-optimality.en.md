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

## Projected-gradient fixed points and a descent proof

For a nonempty closed convex C, the projection criterion is q=P_C(z) exactly when ⟨z−q,y−q⟩≤0 for all y in C. Substituting z=x−η∇f(x) yields the equivalence

$$
x=P_C(x-\eta\nabla f(x))
\quad\Longleftrightarrow\quad
\langle\nabla f(x),y-x\rangle\ge0\ (y\in C),\qquad\eta>0.
$$

For differentiable convex f this certifies global optimality. With an L-Lipschitz gradient on the relevant convex domain, let d=x_next−x. Projection gives ∇f(x)·d≤−‖d‖²/η. Combine it with the descent lemma to obtain f(x_next)≤f(x)−(1/η−L/2)‖d‖². For η≤1/L the guaranteed decrease is at least ‖d‖²/(2η). Feasibility and the step-size assumption are both part of the statement.

The exact-fraction code projects a two-dimensional quadratic onto a triangle by enumerating its three edges. This is a separate extension of the interval lab. At the boundary, a nonzero gradient may be balanced by the normal even when a gradient-only stopping test would reject the solution.

| Condition | What it establishes |
|---|---|
| Feasible fixed point, convex differentiable f | Global optimum |
| Feasible projected step, η≤1/L | Objective descent |
| Small objective difference from rounded heights | No certificate by itself |

The interval readout now evaluates objective excess from displacement to the optimizer. For a=1,C=[−1,0],x=−10⁻²⁰, both rounded objective heights equal 1/2 but true excess is approximately 10⁻²⁰.

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
c=[F(4,5),F(3,5)]; eta=F(1,2)
def norm2(v): return sum(x*x for x in v)
def clip(x): return max(F(0),min(F(1),x))
def project(z):
    t=clip((z[0]-z[1]+1)/2)
    candidates=[[F(0),clip(z[1])],[clip(z[0]),F(0)],[t,1-t]]
    if min(z)>=0 and sum(z)<=1: candidates.append(z)
    return min(candidates,key=lambda q:norm2([q[i]-z[i] for i in range(2)]))
def value(x): return norm2([x[i]-c[i] for i in range(2)])/2
x=[F(0),F(0)]
for k in range(8):
    nxt=project([x[i]-eta*(x[i]-c[i]) for i in range(2)])
    step=[nxt[i]-x[i] for i in range(2)]
    assert value(nxt) <= value(x)-norm2(step)/(2*eta)
    print(k,tuple(x),float(value(x)))
    x=nxt
```

## Exercise and explained answer

Set a=0.5,C=[−1,1],candidate zero. Gradient is −0.5 and the minimum product is −0.5 at y=1, revealing descent. The unique optimum is 0.5. Then set both bounds to zero and a=2: the only feasible point is optimal despite gradient −2, because every feasible displacement vanishes.

## Research connection

Projection methods, constrained parameterizations and geometric optimization rely on feasible directions rather than unconstrained gradient tests alone. Record the constraint set, differentiability and convexity assumptions before treating a residual as a global certificate. Strong convexity adds uniqueness and distance bounds, but convexity alone need not provide those stronger conclusions.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
