## Mixing laws and moving particles are different operations

The linear mixture μ_mix,t=(1−t)μ+tν changes weights on existing support points. Displacement interpolation instead selects a coupling γ and pushes each paired mass to its interpolated location:

$$
\mu_t=\bigl((x,y)\mapsto(1-t)x+ty\bigr)_\#\gamma.
$$

Both have mean (1−t)EμX+tEνY. Equal means do not make their variances, supports or transport distances equal. The upper diagram shows position horizontally and time vertically; orange points identify the current moving particles.

## Optimal coupling creates a Wasserstein geodesic

For squared Euclidean cost and an optimal finite-second-moment coupling,

$$
W_2(\mu,\mu_t)=tW_2(\mu,\nu),\qquad
W_2(\mu_t,\nu)=(1-t)W_2(\mu,\nu).
$$

The lab recomputes these distances using one-dimensional quantile overlaps rather than simply assigning them the right-hand values. This provides a separate numerical check of the geodesic identity.

The optimal mode uses monotone coupling. The product mode uses independent source and target draws, which always has the correct marginals but is generally not cost-optimal. Interpolating its positions is a legitimate path of measures, without automatically being a shortest constant-speed path.

## A midpoint example with point masses

Let μ be a point mass at −1 and ν a point mass at one. At t=0.5, the mixture keeps half the mass at each endpoint, with variance one. Displacement moves all mass to zero, with variance zero. Both means are zero.

Endpoint W₂ distance is two. The displacement midpoint is distance one from the source, but the mixture midpoint is distance √2 from it. The words interpolation and midpoint therefore need an explicit geometry.

## A nonoptimal path between identical endpoints

Use μ=ν=(δ₋₁+δ₁)/2 and the product coupling. A quarter of mass follows each of four pairs. At t=0.5 the moving law is 0.25δ₋₁+0.5δ₀+0.25δ₁. It differs from the unchanged endpoint law even though endpoint W₂ distance is zero.

The selected coupling costs two, while optimal cost is zero. Its intermediate distance from the source is √0.5, not t·0. Thus correct marginals and straight particle tracks are insufficient for a geodesic claim.

## Exercise and representation details

Compare the two midpoint presets and independently compute their atomic masses. Then set t=0 and t=1: all coincident atoms must combine back into the endpoint laws. The implementation merges equal positions and excludes zero masses, avoiding multiple probability stems that would misrepresent one atom.

For the unequal-weight optimal example, check the source and target distances sum to the endpoint distance. The particle diagram retains individual coupling tracks, while the probability table merges particles sharing a position. These views describe the same transported measure at different levels of detail.

In generative or distributional AI models, state which coupling, cost and interpolation are used before referring to a learned path as a transport geodesic.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
