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

## An epoch couples batches even when each marginal is unbiased

Shuffle all four offsets and split them into two disjoint size-two batches. Let their mean offsets be m₁,m₂. They obey m₂=−m₁ exactly, so Cov(m₁,m₂)=−5/3. Each marginal still has mean zero and variance 5/3. Marginal unbiasedness is weaker than fresh conditional unbiasedness after observing the first batch.

For two sequential updates with component gradients x+σm and q=1−η, the final state is

$$
x_2=q^2x_0-\eta\sigma(qm_1+m_2)
=q^2x_0+\eta^2\sigma m_1,
\qquad\operatorname{Var}(x_2)=\tfrac53\eta^4\sigma^2.
$$

By contrast, two independently redrawn size-two no-replacement batches give variance (5/3)η²σ²(1+q²). The difference comes from cross-batch covariance. Averaging all four gradients at one common point removes the noise, but sequential updates evaluate gradients at different points and do not cancel it completely.

| Scheme, η=1/5 and σ=1 | Two-step variance |
|---|---:|
| One reshuffled epoch | 1/375 |
| Independent size-two batches | 41/375 |
| One full batch at one point | 0 |

The code enumerates all 24 permutations exactly. This is a separate two-step experiment; the controls show one-batch marginal distributions. The epoch result depends on the quadratic gradients and should not be generalized to every reshuffling algorithm or dataset.

## Reproduce the calculation in Python

```python
from itertools import permutations
from fractions import Fraction as F
eta=F(1,5); outcomes=[]; covariance=F(0)
for p in permutations([-3,-1,1,3]):
    m1,m2=F(p[0]+p[1],2),F(p[2]+p[3],2)
    assert m2==-m1
    x=-eta*m1; x=(1-eta)*x-eta*m2
    outcomes.append(x); covariance+=m1*m2/24
variance=sum(x*x for x in outcomes)/24
independent=F(5,3)*eta*eta*(1+(1-eta)**2)
assert covariance==-F(5,3) and variance==F(1,375) and independent==F(41,375)
print('covariance:',covariance,'epoch variance:',variance,'independent:',independent)
```

## Exercise and explained answer

Use σ=1,x=0,B=2. With replacement, the mean zero outcome comes from (−3,3),(3,−3),(−1,1),(1,−1), giving probability 4/16. Without replacement those pairs still occur, but the denominator is twelve, giving probability 4/12. Calculate the remaining masses and verify total probability one and variances 2.5 versus 5/3.

Compare B=4 in both schemes. Explain why only the no-replacement mode removes noise entirely. Real epochs with random reshuffling also correlate successive batches, so this one-batch marginal calculation does not establish independence of an entire SGD trajectory. Sampling cost, parallelism and learning-rate choices remain separate considerations.

Further reading: [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
