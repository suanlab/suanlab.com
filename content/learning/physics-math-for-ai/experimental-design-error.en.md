## Identify independent units before counting measurements

Uncertainty depends on covariances as well as measurement count. Repeats from one subject, dataset or training run may not be independent. The lab compares three designs with known unit marginal variance and input correlation. Correlation is a model assumption rather than an estimate from observed data. Paired and independent-group modes target a mean difference; clustered mode targets a single population mean.

$$
\operatorname{Var}\!\left(\frac1N\sum_iY_i\right)=\frac1{N^2}\sum_{i,j}\operatorname{Cov}(Y_i,Y_j).
$$

## Shared noise can cancel in a paired difference

For n independent pairs with unit marginal variances and within-pair correlation ρ, difference variance is 2(1−ρ). At n=10,ρ=0.8 the mean-difference standard error is 0.2. Independent groups with n measurements each give √(2/n), about 0.447. Both use 2n measurements but have different covariance structures. The repeat-count control is inactive in these two modes.

$$
\operatorname{Var}(\overline{Y_B-Y_A})=\frac{2(1-\rho)}n,\qquad\operatorname{Var}(\bar Y_B-\bar Y_A)=\frac2n\ \text{for independent groups}.
$$

## Clustered repeats do not add independent units

With n independent clusters, k measurements each and common within-cluster correlation ρ≥0, mean variance is [1+(k−1)ρ]/(nk). It exceeds the naive independence variance by the design effect. At n=10,k=4,ρ=0.8, DE=3.4 and standard errors are approximately 0.292 versus 0.158. Effective sample size is a variance-equivalent count for this specific mean model, not a universal count for every estimator.

$$
\mathrm{DE}=1+(k-1)\rho,\qquad n_{\mathrm{eff}}=\frac{nk}{\mathrm{DE}}.
$$

## Randomization, blocking and evaluation splits

At cluster correlation one, repeated noise is identical and effective sample size equals n. Zero paired-difference error at perfect correlation likewise reflects a strong model assumption rather than a real-world guarantee. Distinguish random assignment for reducing confounding from blocking or pairing for variance reduction, and distinguish assignment from random population sampling. AI evaluations should record reused data, seeds, selected outcomes, held-out splits and planned comparisons.

## Derive the cluster formula from shared and individual noise

For a balanced design with 0≤ρ≤1 and C independent clusters and m measurements per cluster, write Y_ji=μ+U_j+ε_ji. Let the centered U_j have variance ρ and the centered ε_ji variance 1−ρ, with all cluster effects and individual errors mutually independent. Each measurement has variance one and distinct measurements within a cluster have covariance ρ. The grand mean averages the C shared effects once each and the Cm individual errors separately, giving

$$
\operatorname{Var}(\bar Y)=\frac{\rho}{C}+\frac{1-\rho}{Cm}
=\frac{\rho+(1-\rho)/m}{C}.
$$

Gaussian noise is unnecessary for this variance calculation. The formula alone supplies no confidence-coverage guarantee. Its target is the common population mean under the specified independent-cluster model.

## Add a cost model before choosing repeats

Suppose starting each cluster costs a>0, each measurement costs b>0 and total budget is T. Restrict attention to balanced designs with integers C≥2,m≥1 and C(a+bm)≤T. For a continuous relaxation with all budget spent, C=T/(a+bm). At 0<ρ<1 the variance becomes

$$
V(m)=\frac{(a+bm)(\rho+(1-\rho)/m)}{T},\qquad
m_* = \sqrt{\frac{a(1-\rho)}{b\rho}}.
$$

To derive m*, expand TV(m)=aρ+b(1−ρ)+bρm+a(1−ρ)/m. The growing term bρm and decreasing term a(1−ρ)/m balance when the derivative is zero. The second derivative is positive, so this is the unique unconstrained continuous minimum. Bounds such as m≥1 and C≥2 must still be applied. Larger startup cost favors more repeats; stronger shared noise favors spending on more independent clusters.

## Integer budgets can change the best allocation

Take a=9,b=1,T=120 and ρ=1/4. The relaxed repeat optimum is √27≈5.196, but rounding it to five does not give the best feasible balanced integer design. For a fixed m, increasing C lowers variance, so use C=floor(120/(9+m)). The constraint C≥2 makes the complete feasible repeat range 1 through 51.

| m | C | Cost | Mean variance |
|---|---|---|---|
| 1 | 12 | 120 | $1/12$ |
| 3 | 10 | 120 | $1/20$ |
| 5 | 8 | 112 | $1/20$ |
| 6 | 8 | 120 | $3/64$ |

Enumerating that full feasible range gives the unique optimum m=6,C=8, with variance 3/64. At m=5, starting a ninth cluster would exceed budget, leaving eight units unused. The integer cost constraint explains why simple rounding fails. This is an optimum within the stated balanced class; unequal cluster sizes, different costs or informative cluster sampling define other design problems.

## Check the allocation and covariance calculation exactly

The code enumerates every feasible balanced design with fractions and independently sums the full covariance matrix for the displayed choices. It also shows sensitivity to the planning correlation. If ρ is estimated, its uncertainty belongs in planning sensitivity analysis. This cost model is a separate executable extension: the existing controls compare fixed-count paired, independent and clustered designs and do not optimize a budget.

```python
from fractions import Fraction as Q
from math import sqrt

budget, start_cost, repeat_cost = 120, 9, 1
rho = Q(1, 4)

def designs(rho):
    out = []
    for m in range(1, 52):  # Exactly the feasible repeat range when C >= 2.
        C = budget//(start_cost+repeat_cost*m)
        assert C >= 2
        cost = C*(start_cost+repeat_cost*m)
        variance = (rho+(1-rho)/m)/C
        out.append((m, C, cost, variance))
    return out

rows = designs(rho)
best = min(row[3] for row in rows)
assert [row for row in rows if row[3] == best] == [(6, 8, 120, Q(3, 64))]
continuous = sqrt(start_cost*(1-float(rho))/(repeat_cost*float(rho)))
assert round(continuous) == 5
for m, C, cost, variance in rows:
    if m not in [1, 3, 5, 6]:
        continue
    N = C*m
    covariance_sum = sum(Q(1) if i == j else rho if i//m == j//m else Q(0)
                         for i in range(N) for j in range(N))
    assert covariance_sum/(N*N) == variance
    print("repeats, clusters, cost, mean variance:", m, C, cost, variance)
for alternative in [Q(1, 10), Q(1, 4), Q(1, 2)]:
    candidate = designs(alternative)
    value = min(row[3] for row in candidate)
    print("rho:", alternative, "optimal balanced designs:",
          [row for row in candidate if row[3] == value])
print("Continuous repeat optimum:", continuous)
```

## Further questions and research use

1. Can more measurements be less useful than more clusters? **Yes: repeats retain the shared U_j contribution.**
2. Does the continuous optimum remove integer constraints? **No: the table's best integer allocation differs from rounding m*.**
3. What should a study's budget comparison report? **The estimand, independent units, covariance assumptions, cost model and feasible design class.**

The allocation formulas are derived here from the stated random-effects and linear-cost model. Compare [the expectation/variance identity](../expectation-variance/) and [the paired research example](../research-methods/) before interpreting repeated measurements as independent evidence.

## Exercises with explanations

1. What is the cluster design effect at ρ=0? **One.**
2. Does repeat correlation affect variance when k=1? **No: there is only one measurement per cluster.**
3. Do changed seeds on one test set capture every population uncertainty? **No: identify which sources were independently replicated.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
