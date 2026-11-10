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

## Exercise and explained answer

Recompute the last example by integrating the two step quantiles. Their difference is two on a probability interval of length 0.5 and zero elsewhere. Verify the lower bound |EμX−EνY|≤Wₚ, obtained from Jensen and the coupling marginal constraints.

Use a=1 or zero to check degenerate one-atom laws. Coincident measures must have zero distance even when their array representation includes unused zero-mass positions. For empirical AI distributions, record the ground metric, moment assumptions, weights and whether a reported value is Wₚ or its pth power.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
