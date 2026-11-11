## A power cost becomes a distance only after taking a root

For probability measures with finite pth moments, the Wasserstein distance is

$$
W_p(\mu,\nu)=\left(\inf_{\gamma\in\Pi(\mu,\nu)}
\int|x-y|^p\,d\gamma(x,y)\right)^{1/p},\qquad p\ge1.
$$

The lab compares p=1 and p=2. Minimum squared cost is W₂², while the metric is W₂. Squaring a metric does not preserve its triangle inequality: three point masses at positions zero, one and two already give a counterexample.

## In one dimension, match quantile levels

Let Qμ and Qν be generalized inverse CDFs. The same uniform random level U generates X=Qμ(U),Y=Qν(U), yielding an optimal monotone coupling:

$$
W_p^p=\int_0^1|Q_\mu(u)-Q_\nu(u)|^p\,du.
$$

With atoms, a flat source quantile interval can overlap several target intervals. This is a coupling construction and need not define a deterministic map from the source atom alone. The shared probability level carries information lost when only the atom's location is retained.

The diagram plots quantiles against probability u, not densities against physical position. Vertical segments connect jumps for display; their intermediate heights do not represent extra atoms. Zero-mass atoms are omitted from the quantile graph.

## Compute interval overlaps exactly

Source atoms are (−1,1), with masses (a,1−a). Target atoms are (s−r,s+r), where s is a translation and r>0 a width scale. Each plan entry is the length of intersection between a source CDF interval and a target CDF interval. Multiplying this length by the corresponding power distance gives the cost table.

This finite calculation uses sorted positions and normalized nonnegative masses. In higher dimensions there is generally no single common quantile ordering that solves arbitrary transport problems.

## Three comparisons

For equal half masses, translating both atoms by one gives W₁=W₂=1. Doubling the width from atoms ±1 to atoms ±2 also gives distance one, although the means remain equal. Equal means therefore do not imply equal distributions.

Keep support ±1 but change left mass from 0.25 to 0.75. Half the mass must move distance two. W₁ is one, squared W₂ cost is two, and W₂ is √2. The distinction concerns how long moves are weighted, not a different normalization of total mass.

## Prove the metric inequality and identify a missing moment condition

In one dimension the quantile formula makes the triangle inequality an Lp Minkowski inequality: Qμ−Qλ=(Qμ−Qν)+(Qν−Qλ). Taking norms gives Wₚ(μ,λ)≤Wₚ(μ,ν)+Wₚ(ν,λ), assuming finite pth moments. The root in Wₚ is essential for this argument.

For squared cost, uncrossing ordered points explains monotonicity. If x₁<x₂ and y₁<y₂, replacing two crossed edges by ordered edges changes cost by

$$
(x_1-y_2)^2+(x_2-y_1)^2
-(x_1-y_1)^2-(x_2-y_2)^2
=2(x_2-x_1)(y_2-y_1)>0.
$$

Move the same positive mass on the two edges so marginals stay fixed. A crossed support cannot be optimal. For absolute cost the gain can be zero, so monotone optimality does not imply uniqueness.

Weak convergence alone does not control a Wasserstein distance. Let μₙ=(1−n⁻²)δ₀+n⁻²δₙ. For every bounded continuous f, |∫f dμₙ−f(0)|≤2‖f‖∞/n², proving weak convergence to δ₀. Yet

$$
W_1(\mu_n,\delta_0)=1/n,\qquad
W_2^2(\mu_n,\delta_0)=1.
$$

| n | Tail mass | W₁ | W₂² |
|---|---:|---:|---:|
| 2 | 1/4 | 1/2 | 1 |
| 4 | 1/16 | 1/4 | 1 |
| 8 | 1/64 | 1/8 | 1 |

```python
from fractions import Fraction as F
for n in (2, 4, 8):
    tail = F(1, n*n)
    w1, w2_squared = tail*n, tail*n*n
    bounded_test = tail*F(n, 1+n)  # f(x)=x/(1+x), x>=0.
    assert w1 == F(1, n) and w2_squared == 1
    assert bounded_test <= tail
    print(n, tail, w1, w2_squared, "bounded test:", bounded_test)
```

The rare far-away atom escapes bounded tests while retaining a second-moment cost. This separate family explains why weak distributional similarity is insufficient for squared-transport claims in empirical generative evaluation.

## Exercise and explained answer

Recompute the last example by integrating the two step quantiles. Their difference is two on a probability interval of length 0.5 and zero elsewhere. Verify the lower bound |EμX−EνY|≤Wₚ, obtained from Jensen and the coupling marginal constraints.

Use a=1 or zero to check degenerate one-atom laws. Coincident measures must have zero distance even when their array representation includes unused zero-mass positions. For empirical AI distributions, record the ground metric, moment assumptions, weights and whether a reported value is Wₚ or its pth power.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
