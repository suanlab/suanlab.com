## Use range alone or also variance?

Compare deviations of the mean of n independent Bernoulli(p) observations. Hoeffding uses the range [0,1]; Bernstein also uses centered variance and an absolute bound. Constants differ across one-sided, two-sided, sum and mean conventions, so the displayed formulas define this lab's convention.

$$
P(|\bar X-p|\geq t)\leq\min(1,2e^{-2nt^2}).
$$

## Bernstein assumptions and scale

For Zᵢ=Xᵢ−p, use variance v=p(1−p) and absolute bound M=max(p,1−p). For interior p and positive t:

$$
P(|\bar X-p|\geq t)\leq\min\!\left(1,2\exp\left[-\frac{nt^2}{2(v+Mt/3)}\right]\right).
$$

Both denominator terms have squared-observation units. At p=0 or one, the variable is constant and every positive-deviation tail is zero. The implementation handles this separately with actual centered bound zero. The inclusive event at t=0 has probability one.

## A small-variance example

At n=30,p=.1,t=.2, v=.09 and M=.9. Hoeffding is about .18144; Bernstein is 2exp(−4)≈.03663. The exact event sums binomial counts K≥9, including the boundary count nine. Compare the exact tail with the bounds rather than treating a bound as the probability.

## No uniformly winning curve

At p=.5, v=.25 and M=.5. Bernstein can exceed Hoeffding at the same n,t. Variance sensitivity does not guarantee a smaller bound everywhere. Bounds must lie above actual probability; values above one are replaced by the trivial upper bound one.

Blue and orange are bound curves against t. Exact tails appear in the table and readout. Discrete binomial tails change in steps as event membership changes, even though these bound curves are smooth.

## Applying the formulas

Check independence, population boundedness, centering and variance information. The largest observed sample value does not certify a population-wide bound. Dependence or uncertainty in estimating range and variance calls for additional results. Here the Bernoulli law is known and both exact probability and bounds are calculated under that law.

## Exercises with explanations

1. Is v the variance of the mean? **No: it is single-observation variance; n appears separately.**
2. Are the curves exact tails? **No: they are upper bounds.**
3. Can a constant p=0 variable exceed a positive deviation? **No.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
