## A sample mean with known noise

Assume independent Gaussian observations with known σ. The sample mean has standard error σ/√n, giving an exact normal pivot even at finite n. This lab does not estimate σ or run a t procedure. Change observed mean and sample count to compare confidence intervals with a two-sided test of zero mean.

$$
Z=\frac{\bar X-\mu}{\sigma/\sqrt n}\sim N(0,1).
$$

## Confidence describes procedure coverage

A 95% confidence procedure produces random intervals covering a fixed μ in 95% of repeated samples under the model. It does not assign posterior probability 95% to the parameter lying in one observed interval. The defaults give standard error 0.25 and interval approximately (0.010,0.990). Check that the known-σ assumption actually holds before applying the procedure.

$$
C(X)=\left[\bar X-z_{1-\alpha/2}\frac\sigma{\sqrt n},\;\bar X+z_{1-\alpha/2}\frac\sigma{\sqrt n}\right],\quad P_\mu(\mu\in C(X))=1-\alpha.
$$

## Relate the two-sided test to the interval

The observed null statistic is two. Under a standard normal, the two-sided tail probability is approximately 0.0455. The observed |Z|=2 exceeds the 5% two-sided critical magnitude, so the test rejects and the corresponding interval excludes zero. A p-value measures data extremeness under the null, not the probability that the null is true. Failure to reject is not proof of the null either.

$$
p_{\mathrm{two}}=2\Phi(-|Z_{\mathrm{obs}}|),\qquad\text{reject if }|Z_{\mathrm{obs}}|>z_{1-\alpha/2}.
$$

## The plotted density and scope

Orange tails mark the rejection region, whose area is the nominal significance level. The observed point is drawn only inside the visible window [−4,4]; its numerical readout is never clipped. A 90% interval is narrower because its critical value is smaller. Unknown variance, nonnormality, selected samples, repeated testing or data-dependent hypothesis selection require reviewing the procedure’s guarantees.

## Audit coverage in a different sampling model

The exact Gaussian pivot above depends on known σ and Gaussian sampling. A Bernoulli proportion illustrates why inserting an estimated standard error into the same shape does not create an exact interval. Let K∼Binomial(n,p), with p fixed, and let p̂=K/n. The Wald interval uses p̂±z√(p̂(1−p̂)/n), optionally clipped to [0,1]. At K=0 it is the single point zero, so it misses every positive p. Clipping endpoints to the parameter range does not repair this defect.

For a concrete finite comparison, take n=10,p=1/20 and z=49/25=1.96. The two normal-based rules use this conventional approximate 95% critical value. Coverage is calculated from the entire binomial sampling law, rather than from the appearance of one interval:

$$
\operatorname{Coverage}_p(C)=\sum_{k=0}^n\mathbf1_{\{p\in C(k)\}}{n\choose k}p^k(1-p)^{n-k}.
$$

## Invert the score condition instead of plugging in its variance

For interior candidate p, a score test uses the null variance p(1−p), while the Wald rule uses the observed p̂(1−p̂). Inverting the score acceptance condition gives the Wilson interval. Its defining quadratic is

$$
n(\hat p-p)^2\le z^2p(1-p),\qquad
(n+z^2)p^2-(2n\hat p+z^2)p+n\hat p^2\le0.
$$

The quadratic also evaluates p=0 or one without dividing by zero variance. At K=0, its endpoints are 0 and z²/(n+z²), so it avoids the collapsed Wald interval. It remains a normal-based approximation, not a guarantee of at least 95% coverage for every finite n,p.

## Invert exact discrete tails and prove a coverage bound

Set α=1/20. The equal-tailed exact binomial interval, usually called Clopper–Pearson, retains p when both inclusive tail probabilities are at least α/2:

$$
C_{\mathrm{exact}}(k)=\{p:\ P_p(K'\le k)\ge\alpha/2,\ P_p(K'\ge k)\ge\alpha/2\}.
$$

For fixed p, the event that its lower-tail probability falls below α/2 is an initial segment of possible counts whose total probability is at most α/2. The upper-tail rejection event has the same bound. The union bound therefore makes the probability of excluding the true p at most α. This proves coverage **at least** 1−α; discreteness can make it larger. At k=0 use lower endpoint zero, and at k=n use upper endpoint one.

For the fixed n=10,p=1/20, the accepted counts and coverage are:

| Rule | Accepted k | Coverage |
|---|---|---|
| Wald | 1,2,3 | 0.400235 |
| Score | 0,1 | 0.913862 |
| Exact tails | 0,1,2 | 0.988496 |

The first rule misses the very probable zero count. The score rule improves that behavior here but still falls below nominal coverage. The exact-tail rule is conservative in this example. These are results for the explicitly stated rules and parameter; one favorable table does not establish a universal ranking of interval widths or performance.

## Enumerate the coverage with exact rational arithmetic

The code tests interval membership through the defining inequalities, so it needs no numerical root solver. Its finite coverage sums are exact fractions; table decimals are rounded. The 101-point parameter grid illustrates the exact-tail guarantee, while the tail argument supplies the general proof. This Bernoulli extension is separate from the page's known-σ Gaussian controls.

```python
from fractions import Fraction as Q
from math import comb

n, p, z, alpha = 10, Q(1, 20), Q(49, 25), Q(1, 20)

def binomial_law(p):
    mass = [Q(comb(n, k))*p**k*(1-p)**(n-k) for k in range(n+1)]
    assert sum(mass) == 1
    return mass

def exact_tail_accepts(k, mass):
    return sum(mass[:k+1]) >= alpha/2 and sum(mass[k:]) >= alpha/2

mass = binomial_law(p)
accepted = {name: [] for name in ['Wald', 'Score', 'Exact tails']}
for k in range(n+1):
    phat = Q(k, n)
    gap = n*(phat-p)**2
    if gap <= z*z*phat*(1-phat):
        accepted['Wald'].append(k)
    if gap <= z*z*p*(1-p):
        accepted['Score'].append(k)
    if exact_tail_accepts(k, mass):
        accepted['Exact tails'].append(k)
assert accepted == {'Wald': [1, 2, 3], 'Score': [0, 1], 'Exact tails': [0, 1, 2]}
for name, ks in accepted.items():
    coverage = sum(mass[k] for k in ks)
    print(name, "accepted counts:", ks, "coverage:", coverage, float(coverage))
for j in range(101):
    law = binomial_law(Q(j, 100))
    coverage = sum(law[k] for k in range(n+1) if exact_tail_accepts(k, law))
    assert coverage >= 1-alpha
print("Exact-tail coverage checked on 101 parameter values; general proof is in the text.")
```

## Further questions and research use

1. Does a realized interval reveal its procedure's coverage? **No: coverage is a probability over repeated data at a fixed parameter.**
2. Does “exact” necessarily mean coverage equals 95%? **No: the discrete guarantee can be conservative.**
3. What should a benchmark error interval specify? **The sampling law, independent unit, interval rule and any model selection or repeated testing.**

For Wilson inversion and exact binomial tails, see [NIST, §7.2.4.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm). The finite comparison and the explicit coverage-bound proof above use the stated binomial law.

## Exercises with explanations

1. What is the p-value at observed mean zero? **One.**
2. What happens to width when n quadruples? **It halves at fixed σ.**
3. Is this a posterior interval? **No: its construction and interpretation differ.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[NIST, Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) — For repeated-coverage interpretation and the unknown-σ t-interval counterpart.
