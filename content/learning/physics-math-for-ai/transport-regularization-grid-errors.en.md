## Three errors require three different checks

An entropic transport computation can differ from a desired continuum problem because of regularization, incomplete optimization and discretized marginals. Increasing the grid size does not automatically control the other two. This lab separates them in a family with an independently known continuum answer.

Let μ be uniform on [0,1] and ν its translation by s. Their W₂ distance is |s|. Use N=2ᵏ midpoint atoms for each law. Matching corresponding midpoints also gives discrete W₂=|s| for every N, so actual pair-distance discretization error is zero.

## A positive bound can coexist with zero actual error

For each individual marginal, direct integration over midpoint cells gives

$$
W_2(\mu,\mu_N)=\frac{1}{\sqrt{12}N},\qquad
|W_2(\mu,\nu)-W_2(\mu_N,\nu_N)|
\le\frac{2}{\sqrt{12}N}.
$$

The bound is positive while the actual pair error in this translation family is zero. It bounds perturbations; it does not claim equality. These quantities are distances, whereas the regularization certificate below is in squared-cost units.

## Restore feasibility before using a primal bound

A finite Sinkhorn matrix may have wrong marginals. The experiment converts it into a feasible plan with mass quantum 2⁻²⁴. Every row and column has integer capacity 2²⁴/N. First retain floored cell masses subject to remaining capacities, then distribute residual integer capacities.

All resulting entries are nonnegative integer multiples of the quantum. Their row and column sums are exactly 1/N in binary arithmetic. This operation is a feasibility correction, not another Sinkhorn convergence theorem. Its reported L1 correction includes both marginal repair and quantization effects.

## Combine an entropy bound with an optimization certificate

Let G be the regularized primal–dual Fenchel gap for the corrected feasible plan and the current potentials. For uniform N-point marginals, joint entropy lies between log N and 2log N. Comparing with the unregularized permutation plan yields

$$
0\le \langle C,\gamma_{feas}\rangle-OT
\le\epsilon\log N+G.
$$

The εlog N term bounds the transport-cost bias of the exact regularized optimizer. G separately controls regularized optimization error. Neither term is the marginal-discretization bound above. This worst-case entropy bound can grow with N without proving that actual bias grows.

Blue plots corrected excess transport cost and orange its certificate. The excess is evaluated as Σγᵢⱼ(xᵢ−xⱼ)², avoiding cancellation against the translation cost s². The feasible regularized objective and dual are also shown with the stable generalized-KL gap.

## Derive a distance budget from squared-cost certificates

For uniform marginals, H(P)=H(X)+H(Y|X)≥log N and H(P)≤H(X)+H(Y)=2log N. A feasible permutation optimizer P₀ has H(P₀)=log N. If the corrected feasible P has regularized gap G, then Jε(P)≤Jε(P₀)+G. Canceling the entropy convention's constant gives

$$
\langle C,P\rangle-OT_N
\le\epsilon[H(P)-H(P_0)]+G
\le\epsilon\log N+G=:B.
$$

Now suppose each marginal approximation has a verified W₂ error, whose sum is δ. Triangle inequalities give |W₂(μ,ν)−W₂(μ_N,ν_N)|≤δ. Since a feasible plan's squared cost c is at least OT_N, use |√c−√OT_N|≤√B to obtain

$$
|\sqrt{c}-W_2(\mu,\nu)|\le\delta+\sqrt B.
$$

This explicitly converts cost units to distance units. Adding δ directly to B would mix dimensions. For the midpoint family, δ=1/(√3N). To target distance tolerance τ, it is sufficient to choose δ≤τ/2, εlog N≤τ²/8 and G≤τ²/8. The last is a stopping requirement, not a statement that a chosen iteration count meets it.

| τ | N | ε budget |
|---|---:|---|
| 1/2 | 4 | τ²/(8log N) |
| 1/4 | 8 | τ²/(8log N) |
| 1/8 | 16 | τ²/(8log N) |

```python
from math import log, sqrt
for tau, N in ((.5, 4), (.25, 8), (.125, 16)):
    delta = 1/(sqrt(3)*N)
    epsilon_budget = tau*tau/(8*log(N))
    gap_target = tau*tau/8
    bound = delta+sqrt(epsilon_budget*log(N)+gap_target)
    assert delta <= tau/2 and bound <= tau
    assert epsilon_budget < .05
    print(tau, N, "epsilon budget:", epsilon_budget,
          "gap target:", gap_target, "distance bound:", bound)
```

The computed ε budgets lie below this lab's minimum ε=0.05; the program is a separate theoretical accuracy-budget example. It does not claim these tolerances are attainable with the unchanged controls. Actual translation-pair grid error is still zero, so the marginal bound can be conservative. General nonuniform marginals need their own entropy and feasibility analysis.

## Exercise and explained answer

At N=8,s=0.5,ε=0.1 and twenty iterations, each marginal quantization distance is approximately 0.036084, actual pair error zero and corrected excess cost approximately 0.038481. The regularization bound is approximately 0.207944 and the regularized gap about 5.71×10⁻⁸.

Double N: marginal quantization distance halves, but optimization residual and entropy effects must still be inspected. Set zero iterations and observe that exact feasibility after correction does not imply a small regularized certificate. Report all these accuracy notions before claiming that a grid computation approximates a continuum transport problem.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
