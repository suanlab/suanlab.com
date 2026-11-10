## Moving mass is different from matching labels

Let μ=Σaᵢδₓᵢ and ν=Σbⱼδᵧⱼ be probability measures. A transport model must preserve their masses, rather than simply connect each source to its closest target. The lab fixes source positions (0,1), target positions (0,2), and allows their left masses a and b to vary.

A deterministic Monge map T assigns one target to each source point and must satisfy T#μ=ν. Consequently, all mass of a positive source atom goes to a single target. If μ is one unit atom but ν has two positive atoms, no such map exists.

## Kantorovich allows a coupling

A plan γᵢⱼ describes how much source mass i goes to target j:

$$
\min_{\gamma\ge0}\sum_{ij}C_{ij}\gamma_{ij},\qquad
\gamma\mathbf1=(a,1-a),\quad
\gamma^T\mathbf1=(b,1-b).
$$

For two atoms every feasible plan is determined by z=γ₀₀:

$$
\gamma=\begin{pmatrix}z&a-z\\b-z&1-a-b+z\end{pmatrix},\qquad
\max(0,a+b-1)\le z\le\min(a,b).
$$

The plan-blend control moves through this interval. Nonnegative endpoint plans are mixed directly, preserving feasibility even when a marginal has zero mass. The graph's line widths illustrate transported mass; displayed edge lengths are not transport costs.

## Solve the entire feasible family

Absolute distance gives C=[[0,2],[1,1]], while squared distance gives C=[[0,4],[1,1]]. The objective slope in z is respectively −2 or −4. Thus the optimal plan uses z=min(a,b). The cost curve represents all feasible choices, while orange marks the optimum.

The Monge problem has only four assignments to enumerate. Its possible left-target masses are 0,a,1−a,1. Assignments differing only on zero-mass source points are still enumerated, although they describe the same transported measure.

## Existence of a map does not eliminate the relaxation gap

With a=b=0.5 and squared cost, the monotone map and optimal plan both cost 0.5. With a=1,b=0.5, the plan splits the single source atom, while a Monge map is impossible.

More subtly, choose a=0.4,b=0.6. A deterministic map exists, but it must swap the targets: cost 0.4·4+0.6·1=2.2. The optimal coupling is [[0.4,0],[0.2,0.4]], with cost 0.6. Splitting improves the atomic problem even though a map is feasible.

## Exercise and assumptions

Reproduce all three cases and check every row and column sum. Explain why sending every source to its nearest target fails when target capacities disagree.

Results guaranteeing optimal maps under quadratic cost use additional assumptions, notably an absolutely continuous source and suitable moment conditions. The atomic counterexample does not contradict those results. A balanced coupling also requires equal total mass; unbalanced transport is a different problem with different constraints.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
