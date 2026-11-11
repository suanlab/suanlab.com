## Expected rates versus finite observed slopes

Estimate true mean zero from independent-model Rademacher observations ±sigma, optionally adding fixed estimator bias b. Each repetition generates the longest sample and uses prefixes N=2,4,...,128. Points at different N within one repetition are coupled, not independent observations.

Blue averages realized squared estimator errors. Orange is the model-derived expectation. Enabling the prediction-noise floor adds exact independent test-noise variance conditional on each learned estimator. Those values are not presented as observed test samples.

## Derive the expected risk

The sample mean has mean zero and variance sigma²/N, while bias is fixed:

$$
E[(\overline X+b)^2]=b^2+\frac{\sigma^2}N.
$$

Predicting an independent noisy label adds sigma², giving total risk b²+sigma²/N+sigma². Conditional risk is evaluated exactly for each realized estimator and then averaged across actual repetitions. Observations are not forced to equal the expectation.

Metric choice changes exponents. If mean-estimation MSE is proportional to 1/N, RMSE is proportional to 1/sqrt(N). Comparing these directly with a confidence radius or deviation probability from the previous chapter mixes different definitions.

## Floors and observation windows

For R(N)=C+A/N, its continuous log-log tangent is

$$
\frac{d\log R}{d\log N}=-\frac{A/N}{C+A/N}.
$$

It is minus one without a floor, but approaches zero when positive C dominates. Thus bias, irreducible noise and the chosen N window can change apparent scaling for the same estimator. Finite-repetition slopes may even leave this theoretical interval because of sampling variation.

## Fit actual positive observations

Ordinary least squares is applied only to positive repetition-average risks:

$$
\widehat s=\frac{\sum_i(\log N_i-\overline{\log N})(\log \widehat R_i-\overline{\log\widehat R})}{\sum_i(\log N_i-\overline{\log N})^2}.
$$

Fewer than two positive points leaves the slope undefined. Zero risk is not replaced with an arbitrary small positive value to manufacture a fit. Prefix dependence also prevents automatically reporting an independent-residual standard error. Assess seeds, repetitions, fit windows and the estimation procedure together.

## Nested prefixes carry an exactly computable covariance

Set bias and floor to zero and use unit Rademacher observations. Let Z_n=(S_n/n)² for a single repetition. Expanding the fourth moment gives E[S_n⁴]=3n²−2n, since only terms with even sign multiplicities survive. Thus Var(Z_n)=2(n−1)/n³. For n≤m, write S_m=S_n+U with independent centered U; conditioning on the prefix leaves

$$
\operatorname{Cov}(Z_n,Z_m)=\frac{2(n-1)}{n\,m^2}.
$$

Averaging K independent repetitions divides this covariance by K, not by the number of plotted sample sizes. More prefix points do not create that many independent experiments. The exact enumeration below checks all joint paths up to m, keeping the shared observations rather than drawing a fresh prefix for each point.

| Nested sizes (n,m), one repetition | Exact covariance |
|---|---:|
| (2,4) | 1/16 |
| (2,8) | 1/64 |
| (4,8) | 3/128 |

```python
from itertools import product
from fractions import Fraction as F
for n,m in ((2,4),(2,8),(4,8)):
    pairs=[(F(sum(s[:n]),n)**2,F(sum(s),m)**2) for s in product((-1,1),repeat=m)]
    mean=lambda values:sum(values,F(0))/len(pairs)
    ez=mean([a for a,b in pairs]);ew=mean([b for a,b in pairs])
    covariance=mean([a*b for a,b in pairs])-ez*ew
    variance=mean([a*a for a,b in pairs])-ez*ez
    assert ez==F(1,n) and ew==F(1,m)
    assert variance==F(2*(n-1),n**3)
    assert covariance==F(2*(n-1),n*m*m)
    print(n,m,covariance)
```

Logarithms add another distinction: E[log Rhat] is not log E[Rhat]. Even a positive expected risk can have a zero realized estimate; every even prefix has positive probability of an exactly zero Rademacher sum. All K repetitions can be zero with positive probability, so a log-risk expectation can be minus infinity. The UI correctly leaves zero-risk points out of its descriptive slope and never adds an artificial floor, but that selection must not be interpreted as an unbiased population exponent estimate. These covariance formulas concern the ideal independent-sign model, not an exact probability distribution over the finite seeded generator.

## Worked example and exercise

With sigma one, zero bias and no prediction floor, expected risk at N=128 is 1/128=.0078125. Including independent prediction noise makes it 1.0078125. A nearly flat large-N curve in the latter case reflects its floor, not an estimator suddenly ceasing to learn.

Exercise: set sigma=b=0. What are risk and log slope? Solution: all risks are zero, so logarithms and the fitted slope are undefined. For sigma zero but bias one, all risks equal one, giving slope zero when at least two N values exist. These cases should not be conflated as the same flat success curve.

## Scaling claims in new AI research

A good finite-window power-law fit is not a universal neural scaling theorem. Record task, model, data distribution, metric, compute, floors and observation range, and compare alternative explanations. In this transparent example, the exact 1/N expectation follows from assumptions while the blue curve comes from actual repetitions. Preserving that distinction is essential to interpreting broader research claims.

References: [MIT learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/) and [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
