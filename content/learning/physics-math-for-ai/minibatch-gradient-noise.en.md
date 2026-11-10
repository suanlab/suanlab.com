## A batch is a sampling design

Let four component gradients at the current scalar parameter be x+σzᵢ, where z=(−3,−1,1,3). The full gradient is x and the population variance, using denominator four, is 5σ². This setup isolates sampling variation from the shape of an entire training trajectory.

A batch mean can use independent draws with replacement, or distinct draws without replacement. These schemes have the same expectation but different covariance. Saying only that the batch has size B leaves an important assumption unspecified.

## Independent replacement sampling

For independent uniform draws, variances add while cross-covariances vanish:

$$
\mathbb E[\bar g_B]=x,\qquad
\operatorname{Var}(\bar g_B)=\frac{5\sigma^2}{B}.
$$

With B=2, there are sixteen equally likely ordered batches. The extreme pair (−3,−3) is possible, producing mean gradient x−3σ. Batch size four still allows repeated indices and does not force the full gradient. Its variance remains 5σ²/4.

## Distinct draws and finite-population correction

Without replacement, the covariance of two distinct draws is −5σ²/3. Expanding the variance of their average gives

$$
\operatorname{Var}(\bar g_B)
=\frac{5\sigma^2}{B}\frac{4-B}{3}.
$$

The denominator three follows from using the population variance with denominator N=4. If one instead defines sample variance with denominator N−1, the formula must be rewritten consistently. Mixing these conventions produces a spurious extra correction.

At B=2, twelve ordered batches are possible and variance becomes 5σ²/3. At B=4 all twenty-four permutations average the whole population exactly, so variance is zero. Order does not change a mean, but counting ordered batches preserves their equal sampling probabilities before aggregation.

## The diagram is an exact PMF

The code enumerates all ordered batches, groups equal integer offset sums and divides multiplicities by the total count. Blue stems show exact probability masses rather than a Monte Carlo histogram or a continuous probability density. Orange on the baseline marks the full gradient; it is not an additional probability atom.

Moving x translates every outcome without changing variance. Multiplying σ scales deviations linearly and variance quadratically. The table reports floating-point renderings of exact finite combinatorial probabilities. Symmetry makes the expected offset exactly zero; that structural cancellation is handled before numerical scaling.

## Exercise and explained answer

Use σ=1,x=0,B=2. With replacement, the mean zero outcome comes from (−3,3),(3,−3),(−1,1),(1,−1), giving probability 4/16. Without replacement those pairs still occur, but the denominator is twelve, giving probability 4/12. Calculate the remaining masses and verify total probability one and variances 2.5 versus 5/3.

Compare B=4 in both schemes. Explain why only the no-replacement mode removes noise entirely. Real epochs with random reshuffling also correlate successive batches, so this one-batch marginal calculation does not establish independence of an entire SGD trajectory. Sampling cost, parallelism and learning-rate choices remain separate considerations.

Further reading: [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
