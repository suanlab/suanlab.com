## When the sampling law is unknown

A statistic’s sampling distribution depends on an often unknown population law. The nonparametric bootstrap substitutes the empirical distribution of observed data. For (0,1,2), each observed row gets mass one third. Resamples also have size three and use replacement, so a row may appear repeatedly.

$$
\widehat P_3=\frac13\sum_{i=1}^3\delta_{x_i},\qquad X_1^*,X_2^*,X_3^*\mid x\overset{\mathrm{iid}}\sim\widehat P_3.
$$

## Enumerate a small case completely

Three row choices at each of three positions give twenty-seven ordered resamples. The lab enumerates all of them. Duplicate observed values still correspond to separate row choices; equal statistic values are aggregated afterward. For the mean, outputs range from zero to two in steps of one third with counts (1,3,6,7,6,3,1). This law is exact conditional on the data, not the exact population sampling law.

$$
P^*(T^*=t\mid x)=\frac{\#\{\text{ordered resamples giving }t\}}{27}.
$$

## Bias, standard error and nonlinear statistics

For a mean, the conditional bootstrap expectation equals the original mean, giving zero bootstrap bias. Its standard error for (0,1,2) is √(2/9). With data (0,0,3) and the median, the observed statistic is zero but the bootstrap median equals three with probability 7/27. Its conditional expectation and bootstrap bias estimate are 7/9. This is an empirical-law calculation rather than exact knowledge of population bias.

$$
\widehat{\mathrm{bias}}^*=E^*[T^*]-T(x),\qquad\widehat{\mathrm{SE}}^*=\sqrt{\operatorname{Var}^*(T^*)}.
$$

## Limits of percentile intervals

Endpoints use the inverse empirical CDF of resampled statistics: choose the first sorted value reaching q, without interpolation. The selected 80% or 95% is a percentile level, not a demonstrated coverage guarantee for this tiny sample. Constant data give zero bootstrap standard error without proving a constant population. Time series or clustered data require considering resampling units that preserve dependence.

$$
[\,\widehat F_*^{-1}(\alpha/2),\;\widehat F_*^{-1}(1-\alpha/2)\,].
$$

## A statistic that the ordinary bootstrap cannot reproduce

Consider iid Uniform(0,θ) observations with θ>0 and maximum M_n. For 0≤t≤n, independence gives an exact survival probability for the endpoint-estimation pivot:

$$
R_n=\frac{n(\theta-M_n)}\theta,\qquad
P_\theta(R_n>t)=(1-t/n)^n\longrightarrow e^{-t}.
$$

Thus R_n converges to a standard exponential law, which has a continuous CDF and no atom at zero. At every finite n, M_n<θ almost surely, so P(R_n=0)=0 as well. This is a boundary-statistic problem, not a sample-mean CLT.

## Locate the nonvanishing atom in empirical resampling

With probability one the observed maximum is unique. In an ordinary n-out-of-n empirical bootstrap, every draw selects that observed row with probability 1/n. The resampled maximum M_n* equals M_n whenever at least one draw selects it. Consequently the bootstrap pivot has a large atom:

$$
R_n^*=\frac{n(M_n-M_n^*)}{M_n},\qquad
P^*(R_n^*=0\mid X)=1-(1-1/n)^n\longrightarrow1-e^{-1}.
$$

Both probability columns below give point mass at zero.

| n | Bootstrap | Population |
|---|---|---|
| 3 | 0.703704 | 0 |
| 10 | 0.651322 | 0 |
| 100 | 0.633968 | 0 |

Since the pivot is nonnegative, its CDF at zero equals this atom. The limiting population CDF is zero there, a continuity point. Therefore the conditional empirical-bootstrap law cannot consistently approximate the pivot's limiting law. Increasing the number of bootstrap replicates only reduces simulation error around this incorrect conditional distribution; it cannot remove the atom. The finite enumeration below has no Monte Carlo error and still displays the problem.

## A model-based resample can succeed for this pivot

If the Uniform(0,θ) model is justified, instead draw fresh independent observations from Uniform(0,M_n). Write these as M_n U_i*, with iid Uniform(0,1) base variables. The normalized resample pivot is n(1−max U_i*), so conditionally on any M_n>0 it has exactly the same law as R_n at finite n:

$$
P^*\!\left(\frac{n(M_n-M_{n,\mathrm{param}}^*)}{M_n}>t\mid X\right)=(1-t/n)^n\quad(0\le t\le n).
$$

This success uses the assumed uniform family and a scale-free pivot. It does not validate parametric resampling under a misspecified population law. Choosing a resampling mechanism is part of the statistical model, not merely a choice of the number of repetitions.

## Enumerate a three-point empirical example

For observed values (1/5,1/2,4/5), all 27 ordered empirical resamples produce pivot values 0,9/8,9/4 with counts 19,7,1. The code verifies these exact fractions and the model-based scale identity. Its fixed base values illustrate that identity and are not random draws. These maximum examples are separate from the existing controls for means and medians on three observed rows.

```python
from fractions import Fraction as Q
from itertools import product
from collections import Counter
from math import exp

sample = (Q(1, 5), Q(1, 2), Q(4, 5))
n, observed_max = len(sample), max(sample)
counts = Counter(n*(observed_max-max(draw))/observed_max
                 for draw in product(sample, repeat=n))
assert counts == Counter({Q(0): 19, Q(9, 8): 7, Q(9, 4): 1})
for value, count in sorted(counts.items()):
    print("empirical-bootstrap pivot, conditional mass:", value, Q(count, n**n))
for n in [3, 10, 100]:
    atom = 1-(1-Q(1, n))**n
    population_survival_at_one = (1-Q(1, n))**n
    assert atom+population_survival_at_one == 1
    print("n, empirical atom at zero, population CDF at zero:", n, float(atom), 0)
print("limiting empirical atom:", 1-exp(-1))
# Fixed base values illustrate the scale identity; these are not random draws.
u = (Q(1, 7), Q(2, 3), Q(4, 5))
for M in [Q(1, 10), Q(4, 5), Q(3)]:
    parametric_sample = tuple(M*x for x in u)
    pivot = len(u)*(M-max(parametric_sample))/M
    assert pivot == len(u)*(1-max(u))
print("Parametric normalized pivot is independent of the positive plug-in endpoint.")
```

## Further questions and research use

1. Does exact enumeration of the empirical bootstrap make it the true sampling law? **No: it removes simulation error, not the mismatch between empirical and population laws.**
2. Why does the maximum have an atom under empirical resampling? **Selecting the observed maximum even once is enough to retain it.**
3. What must a bootstrap report identify? **The statistic, resampling unit, resampling law and assumptions supporting the desired approximation.**

The maximum counterexample and model-based contrast are classical; see [Bickel–Freedman (1981), §6, Counter-example 2, p.1210](https://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/BickelFreedman81.pdf), [publication DOI](https://doi.org/10.1214/aos/1176345637). The finite three-point enumeration is calculated directly here.

## Exercises with explanations

1. Are identical-valued row choices counted separately? **Yes, then statistic outputs are aggregated.**
2. Is shuffling every observation once the same procedure? **No: this uses replacement.**
3. Is nominal percentile coverage automatic at small n? **No: additional theory and diagnostics are needed.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[Wasserman, All of Statistics — author’s page](https://stat.cmu.edu/~larry/all-of-statistics/index.html).
