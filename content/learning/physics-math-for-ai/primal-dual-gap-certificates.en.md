## A feasible pair brackets an unknown optimum

An optimizer may return a point without knowing the exact optimum value. Duality supplies an independently checkable lower bound. If x is primal feasible and λ is dual feasible, weak duality gives d(λ)≤p*≤f(x). Therefore the computable gap f(x)−d(λ) bounds the unknown objective error f(x)−p*.

This certificate gap differs from the optimal-value duality gap p*−d*. Even a problem with strong duality can have a large certificate gap when the current pair is poor. The lab uses a quadratic family where both exact optimum and current certificate can be compared.

## Derive a diagnostic identity

For f(x)=(x−a)²/2 with constraint x≤b, the Lagrangian dual function is d(λ)=λ(a−b)−λ²/2. A feasible multiplier must be nonnegative. Completing the square yields

$$
f(x)-d(\lambda)
=\frac12(x-a+\lambda)^2+\lambda(b-x).
$$

The first term is half the squared stationarity residual. The second is multiplier times slack. Each is nonnegative for a feasible pair, which proves weak duality for this family without a numerical search.

The algebraic identity holds even outside the feasible domains. Nonnegativity of its terms does not. The lab displays the algebraic gap and feasibility flags separately, and withholds the certified gap when either feasibility condition fails. The curve at negative multipliers is an algebraic extension, not a collection of valid lower bounds.

## Worked certificates and failures

Take a=2 and b=1. The optimal pair is x=1, λ=1, with primal and dual value 0.5 and zero gap. At x=0, λ=0 the certified gap is two, while actual objective error is 1.5. The certificate is valid but conservative.

Now take a=0, b=1, x=0 and λ=−1. The algebraic gap is −0.5. This does not refute weak duality because the multiplier is infeasible. Alternatively a=2, b=1, x=2 and λ=1 produces the same negative gap through primal infeasibility. A valid dual lower bound does not make an infeasible x an upper bound.

## Convert objective accuracy into distance

The displayed objective is strongly convex with parameter one. At its constrained optimum x*, the first-order variational inequality combines with strong convexity to give

$$
\frac12|x-x^*|^2\le f(x)-p^*\le f(x)-d(\lambda).
$$

Hence a feasible pair supplies |x−x*|≤√(2gap). For the earlier x=0, λ=0 example the distance bound is two and the actual distance is one. For a general strong-convexity parameter μ the bound becomes √(2gap/μ). Without strong convexity, objective accuracy alone need not control distance to a particular optimizer.

## Exercise and numerical interpretation

Hold a=2,b=1,x=1 and vary λ from zero to two. The primal point stays optimal, but only the matching multiplier makes the gap vanish. Explain this using the squared stationarity term rather than saying the primal solver failed.

The readout also compares the expanded subtraction f−d with the decomposed gap. Near convergence, subtraction can lose relative accuracy when two objective values nearly agree. Decomposition avoids that cancellation for feasible pairs in this family. It does not repair infeasibility, justify silently clipping a negative number to zero, or supply a universal stable formula for every optimization model.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
