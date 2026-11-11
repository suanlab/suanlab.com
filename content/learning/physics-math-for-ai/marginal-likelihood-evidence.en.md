## A best parameter versus an average over parameters

MLE chooses a parameter maximizing likelihood. Evidence, or marginal likelihood, averages likelihood under the prior. Even an excellent best fit can receive a modest average if little prior mass lies near it. These operations answer different questions.

$$
Z=p(x)=\int p(x\mid\theta)p(\theta)\,d\theta.
$$

## Calculate a Beta-model example

With n=8 observations and k successes, a Beta(α,β) prior gives the evidence of one ordered sample as a Beta-function ratio. At k=6 and α=β=1, Z=1/252≈0.003968 and log Z≈−5.529, using natural logarithms.

$$
Z=\frac{B(\alpha+k,\beta+n-k)}{B(\alpha,\beta)}.
$$

For the event consisting of every ordering with the same count, multiply by C(8,k). In the default example this count-event probability is 1/9. The plot and readouts consistently use one ordered sample.

## Recover evidence from sequential predictions

Read observations one at a time. Multiply the probability of each observation under the posterior available just before it. After s successes and f failures, next-success probability is (α+s)/(α+β+s+f); the failure probability uses the other numerator.

$$
p(x)=\prod_{i=1}^{n}p(x_i\mid x_1,\ldots,x_{i-1}),\qquad \log p(x)=\sum_i\log p(x_i\mid x_{<i}).
$$

The table places successes first. Exchangeability means that reordering can change intermediate predictions while preserving the final product. Blue shows cumulative log predictive probabilities. Orange evaluates the same observations with the full-data MLE fixed from the beginning. That reference uses the whole dataset in advance; it is not an online prediction procedure.

## Prior sensitivity and scope

Change the prior shapes and observe the evidence. Because evidence averages over a prior, it can be sensitive to that choice. An improper unnormalized prior can leave an arbitrary multiplicative constant, preventing an ordinary evidence comparison. This lab uses only proper normalized Beta priors.

Products of tiny probabilities motivate computing sums of logarithms. Here a small integer example allows verification by both a function ratio and a sequential factorization.

## The same z statistic can give different evidence ratios

Switch from discrete masses to continuous-data densities in a separate example. With known observation variance one, let X_i|θ be iid N(θ,1), n≥1. M0 fixes θ=0; M1 uses a proper prior θ~N(0,τ²), τ²>0. Both models use the same observations and dominating measure. A density value is not the probability of one observed point.

The sample mean is N(0,1/n) under M0 and N(0,τ²+1/n) under M1. Gaussian residuals about that mean factor out with the same density contribution under both models, so they cancel in the ratio. The full-data Bayes factor BF_10 therefore equals the sample-mean density ratio. Writing z=√n times the observed mean and r=nτ² gives

$$
\operatorname{BF}_{10}
=\frac1{\sqrt{1+r}}\exp\left(\frac{z^2r}{2(1+r)}\right),\qquad
\log\operatorname{BF}_{10}=-\frac12\log(1+r)+\frac{z^2r}{2(1+r)}.
$$

The first factor comes from normalizing M1's wider predictive density. The exponential factor reflects improved fit to the observed mean. Retaining only the fit term would discard part of the evidence calculation.

## Distinguish fixed-statistic and fixed-effect limits

Keep τ²=1 and z=2 while increasing n. This requires the observed mean to shrink as 2/√n. The two-sided null z-test p-value remains approximately 0.04550, while the Bayes factor changes as follows.

| n | Mean | BF_10 |
|---|---|---|
| 1 | 2 | 1.92212 |
| 100 | 0.2 | 0.72082 |
| 10000 | 0.02 | 0.07387 |

For fixed finite z and τ²>0, the exponential tends to exp(z²/2), while the prefactor tends to zero. Hence BF_10→0. This calculation shows why a p-value cannot simply be converted into posterior model probability. Posterior odds also need model prior odds. The two procedures use different probability definitions and alternative assumptions.

By contrast, along a path with fixed observed mean c≠0, z²=nc² and log BF_10/n→c²/2>0. Evidence then grows toward M1. Do not generalize the fixed-z conclusion to say that adding observations always favors M0 under every signal.

## Prior width is an explicit analysis input

Holding n and z fixed while τ²→∞ also gives BF_10→0: the prior places less mass near the region fitting the observations. This is a limit of proper priors. It does not authorize replacing them by an improper flat prior with its normalizing constant deleted. Obtaining a proper parameter posterior and defining evidence between models are separate questions.

For research, specify scientifically plausible effect scales and prior-sensitivity ranges before inspecting the data. This Gaussian example compares the stated two models; it does not certify either model's real-world adequacy. The existing Beta controls are independent of this Gaussian analysis.

## Check the formula through density integration

The code directly multiplies the sample-mean likelihood density and prior density and numerically integrates their product. Changing only the integration coordinate to posterior center and scale avoids missing a narrow peak at large n. This finite-interval Simpson check is approximate and does not replace the algebraic limit proof.

```python
from math import exp, sqrt, pi, log1p, erfc, isclose

def normal(x, mean, sd):
    return exp(-.5*((x-mean)/sd)**2)/(sqrt(2*pi)*sd)

def integrated_bf(n, prior_sd, z):
    y = z/sqrt(n)
    variance = 1/(n+1/prior_sd**2)
    center = variance*n*y
    sd = sqrt(variance)
    def integrand(u):
        theta = center+sd*u
        return normal(y, theta, 1/sqrt(n))*normal(theta, 0, prior_sd)*sd
    steps, lo, hi = 4000, -10, 10
    h = (hi-lo)/steps
    total = integrand(lo)+integrand(hi)
    total += sum((4 if i % 2 else 2)*integrand(lo+i*h)
                 for i in range(1, steps))
    return total*h/3/normal(y, 0, 1/sqrt(n))

z = 2
print("Fixed-z two-sided p-value:", erfc(abs(z)/sqrt(2)))
for n in [1, 100, 10000]:
    for prior_sd in [.5, 1, 2]:
        r = n*prior_sd**2
        exact = exp(-.5*log1p(r)+z*z*r/(2*(1+r)))
        quadrature = integrated_bf(n, prior_sd, z)
        assert isclose(quadrature, exact, rel_tol=1e-10, abs_tol=1e-12)
        print("n, prior sd, observed mean, BF:", n, prior_sd, z/sqrt(n), exact)
print("The integration is approximate; the limiting arguments are in the text.")
```

## Questions for research use

1. Does the same p-value imply the same Bayes factor? **No: sample size and alternative prior scale also matter.**
2. Is the observed mean fixed along the fixed-z path? **No: it changes as z/√n.**
3. What belongs in the reproduction record? **The observation model, variance, prior scale, model odds and common data representation.**

For model dependence and prior sensitivity, see [PyMC, Bayes Factors and Marginal Likelihood](https://www.pymc.io/projects/examples/en/2022.01.0/diagnostics_and_criticism/Bayes_factor.html). The Gaussian formula, two limits and table above are computed directly for the stated model.

## Exercises with explanations

1. Can evidence exceed maximum likelihood? **A likelihood average under a proper prior cannot exceed its maximum.**
2. Does reordering change final evidence? **Not in this exchangeable model.**
3. Is evidence a parameter probability? **No: it is a data-event probability after integrating the parameter.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
