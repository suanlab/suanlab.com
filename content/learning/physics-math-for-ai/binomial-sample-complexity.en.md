## Sufficient data versus exact failure probability

Estimate p using an IID Bernoulli mean. Required data depend on error tolerance epsilon, failure probability delta and the desired guarantee. This experiment compares exact binomial event sums with Hoeffding's distribution-free upper bound. Blue is the finite exact tail and orange the bound clipped at probability one.

The event |K/N-p|≥epsilon includes its boundary. Integer success counts make that boundary important at small N. Log-domain probabilities avoid underflow at extreme p, while floating ties are handled at scaled machine precision.

## Compute the finite event

$$
P\left(\left|\frac KN-p\right|\ge\epsilon\right)
=\sum_{k\le\lfloor N(p-\epsilon)\rfloor\ \mathrm{or}\ k\ge\lceil N(p+\epsilon)\rceil}
{N\choose k}p^k(1-p)^{N-k}.
$$

N is at most 128 here. Endpoint probabilities zero and one are handled as point masses. No prescribed smooth curve replaces the computed sum. Exact within a model still has floating-point rounding limits, so independent factorial formulas and normalization are checked.

## Derive a sufficient Hoeffding count

For independent variables in [0,1],

$$
P(|\overline X-p|\ge\epsilon)\le2e^{-2N\epsilon^2}.
$$

Requiring this bound to be at most delta gives

$$
N\ge\left\lceil\frac{\log(2/\delta)}{2\epsilon^2}\right\rceil.
$$

This is sufficient for every p, not the exact necessary minimum for a particular distribution. The sufficient count can exceed the plotted range. Changing delta changes the confidence target, but not the actual tail at fixed N,p,epsilon.

## Worked example and exercise

For p=.5, epsilon=.2 and N=3, counts one and two have error 1/6 and are excluded. Only zero and three contribute, giving probability 2/8=.25. At N=4, counts one and three also qualify because their error is .25. The tail becomes (1+4+4+1)/16=.625. More samples do not force every finite-n tail to decrease monotonically.

Exercise: obtain a sufficient count for epsilon=.2 and delta=.05. Solution: log(40)/.08≈46.11, so choose 47. The bound controls every N≥47. In contrast, finding a first exact-curve crossing below .05 does not establish that every subsequent N remains below it.

Also distinguish an exact curve at known p from estimating an unknown p. All-zero observations do not prove p is zero. For example, at p=.1, ten zero observations occur with probability .9^10, approximately .349. The interface p is an oracle setting defining the world, not a true parameter inferred automatically from the data.

## Report AI data requirements precisely

Specify loss, deviation event, confidence, IID assumptions and the distribution range when discussing sample complexity. Dependence, selection bias and distribution shift can invalidate an unchanged application. Estimator variance, RMSE and deviation probability are different quantities, so their exponents cannot be compared indiscriminately. The next chapter separates theoretical expectations from finite-window slopes in observed learning curves.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
