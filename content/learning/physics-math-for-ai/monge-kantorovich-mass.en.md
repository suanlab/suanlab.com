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

## Existence of a plan and loss of determinism in a limit

For finite probability marginals, the product matrix aᵢbⱼ is feasible. Every feasible entry lies between zero and min(aᵢ,bⱼ); the equality constraints and nonnegativity define a closed bounded set. It is compact in finite dimensions, so a continuous linear cost attains its minimum. A deterministic map instead belongs to a finite list of assignments whose feasible subset may be empty. Existence of the relaxed optimum therefore requires no map-existence assumption here.

The atomic obstruction can appear in a limit even when every approximating source has a map. Let μₑ be uniform on [0,e], with e>0, and ν=(δ₀+δ₂)/2. Send the lower half interval to zero and the upper half to two. This monotone map is optimal for squared cost. Direct integration gives

$$
\int|x-T_e(x)|^2d\mu_e(x)
=\frac1e\left[\int_0^{e/2}x^2dx+\int_{e/2}^e(2-x)^2dx\right]
=2-\frac32e+\frac13e^2.
$$

As e→0, the source converges to δ₀, and the cost tends to two. The limiting optimal plan splits mass from zero equally between the targets. No deterministic map from δ₀ can do this. Convergence of optimal costs and plans need not preserve the property of being a map. The limiting random target choice is encoded in a coupling rather than a function of the collapsed source position.

| e | Squared cost |
|---|---:|
| 1/2 | 4/3 |
| 1/4 | 79/48 |
| 1/8 | 349/192 |

```python
from fractions import Fraction as F
for e in (F(1, 2), F(1, 4), F(1, 8)):
    low, high = e/2, e
    first = low**3/3
    antiderivative = lambda x: 4*x-2*x*x+x**3/3
    second = antiderivative(high)-antiderivative(low)
    value = (first+second)/e
    assert value == 2-F(3, 2)*e+e*e/3
    print(e, value)
```

This continuum-to-atom example is separate from the lab's fixed source positions. It motivates checking source regularity before transferring a map theorem to empirical atomic data.

## Exercise and assumptions

Reproduce all three cases and check every row and column sum. Explain why sending every source to its nearest target fails when target capacities disagree.

Results guaranteeing optimal maps under quadratic cost use additional assumptions, notably an absolutely continuous source and suitable moment conditions. The atomic counterexample does not contradict those results. A balanced coupling also requires equal total mass; unbalanced transport is a different problem with different constraints.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
