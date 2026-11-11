## Estimating many small groups

Consider means for hospitals, cities or datasets. Small-group means fluctuate considerably. Complete pooling erases differences; separate estimation discards common information. A hierarchical model assumes group parameters come from a shared distribution and creates partial pooling between these extremes.

$$
\theta_j\sim N(0,\tau^2),\qquad \bar Y_j\mid\theta_j\sim N(\theta_j,1/n).
$$

## Add precisions to obtain the posterior

Individual observation variance is known to be one and each group contains n observations. Prior precision is 1/τ²; sample-mean precision is n. Completing the square gives posterior variance and mean.

$$
v=(n+\tau^{-2})^{-1},\qquad m_j=v n\bar y_j=w\bar y_j,\qquad w=\frac{n\tau^2}{1+n\tau^2}.
$$

With n=4, τ=1 and observed group means two and minus one, w=0.8. Posterior means are 1.6 and −0.8; both posterior variances are 0.2. Blue marks show observed means, orange marks posterior means. Their heights encode means; uncertainty is reported separately in the readout and table.

## Change the pooling strength

A smaller τ asserts that group differences are smaller, pulling estimates more strongly toward zero. At τ=0.25, w=0.2 and the observed mean two becomes posterior mean 0.4. Increasing n strengthens the group data and moves w toward one. The weight follows from a specified probabilistic model rather than an arbitrary averaging rule.

## Understand what is conditioned on

The population center zero and τ are fixed inputs. This lab neither estimates nor integrates a hyperparameter posterior. Conditional posterior covariance between the groups is therefore zero. Making the shared center unknown and inferring it jointly can introduce posterior dependence. Also distinguish empirical Bayes, which estimates hyperparameters and plugs them in, from full Bayes, which integrates their posterior uncertainty.

For a new group with no observations, parameter variance remains τ². Do not substitute the smaller conditional posterior variance v of an observed group. This distinction matters in AI applications involving task-specific performance or regional effects with both shared structure and individual differences.

## An unknown shared center couples the posterior groups

The controls above fix the center at zero. In a separate model, give that center μ a prior. For groups j=1,…,J take n_j≥1, known individual observation variance one, and known s²>0 and τ²>0. Group parameters are independent given μ; observed group means are independent given their parameters.

$$
\mu\sim N(0,s^2),\qquad \theta_j\mid\mu\sim N(\mu,\tau^2),\qquad
\bar Y_j\mid\theta_j\sim N(\theta_j,v_j),\quad v_j=1/n_j.
$$

Integrating θ_j gives a normal observation around μ with variance h_j=τ²+v_j. Collecting precisions and completing the square gives the shared-center posterior.

$$
\mu\mid y\sim N(m_\mu,S),\qquad
S=\left(s^{-2}+\sum_jh_j^{-1}\right)^{-1},\qquad
m_\mu=S\sum_j\bar y_j/h_j.
$$

A noisy group has smaller precision 1/h_j and moves the common center less. Changing one group changes m_μ and thereby the estimates of other groups. This identifies the route through which information is shared in this model.

## Integrating conditional independence need not preserve it

Given μ and the data, each group posterior mean is w_j times its observed mean plus (1−w_j)μ, with variance r_j. Apply total expectation and total covariance.

$$
w_j=\frac{\tau^2}{h_j},\quad r_j=\frac{\tau^2v_j}{h_j},\qquad
E[\theta_j\mid y]=w_j\bar y_j+(1-w_j)m_\mu,
$$

$$
\operatorname{Cov}(\theta_j,\theta_k\mid y)
=\mathbf1_{\{j=k\}}r_j+(1-w_j)(1-w_k)S.
$$

The first term remains even if the center is known. The second reflects the shared movement of that uncertain center. For j≠k, conditional covariance is zero but covariance between the conditional means is positive. Substituting μ=m_μ discards the second term. This calculation does not assert that every plug-in procedure has the same error.

## Two groups and an unobserved new group

With n_1=n_2=4, τ²=s²=1 and observed means (2,−1), obtain S=5/13 and m_μ=4/13. In this table, “Fixed” uses the original screen's μ=0 assumption. “Integrated” uses the new model and integrates its shared-center posterior.

| Quantity | Fixed | Integrated |
|---|---|---|
| First mean | $8/5$ | $108/65$ |
| Each variance | $1/5$ | $14/65$ |
| Group covariance | 0 | $1/65$ |
| New parameter variance | 1 | $18/13$ |

An unobserved new group's parameter θ_new shares μ, so its posterior predictive mean is m_μ and its variance is τ²+S. Do not replace this with the smaller variance of an observed group. Predicting the mean of n_new new observations also adds observation noise 1/n_new. Only the shared center is inferred here; uncertainty in τ is still excluded.

## Verify through the joint precision matrix

The code solves the joint log posterior of (μ,θ_1,θ_2) using a precision matrix constructed directly from its squared terms. Fraction-based Gauss–Jordan elimination checks the decomposition without Monte Carlo error. This executable extension is separate from the fixed-center controls.

```python
from fractions import Fraction as Q

def inverse(matrix):
    n = len(matrix)
    a = [[Q(v) for v in row]+[Q(i == j) for j in range(n)]
         for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = a[j][j]
        assert pivot != 0
        a[j] = [v/pivot for v in a[j]]
        for i in range(n):
            if i != j:
                factor = a[i][j]
                a[i] = [v-factor*w for v, w in zip(a[i], a[j])]
    return [row[n:] for row in a]

# Prior exponent: mu^2 + (theta1-mu)^2 + (theta2-mu)^2.
# Likelihood exponent: 4*(2-theta1)^2 + 4*(-1-theta2)^2.
precision = [[3, -1, -1], [-1, 5, 0], [-1, 0, 5]]
cov = inverse(precision)
linear = [0, 8, -4]
means = [sum(cov[i][j]*linear[j] for j in range(3)) for i in range(3)]
S, center, w, r = Q(5, 13), Q(4, 13), Q(4, 5), Q(1, 5)
assert cov[0][0] == S and means[0] == center
assert means[1:] == [Q(108, 65), Q(-48, 65)]
for j in [1, 2]:
    for k in [1, 2]:
        assert cov[j][k] == (r if j == k else 0)+(1-w)**2*S
assert means[1:] == [w*y+(1-w)*center for y in [2, -1]]
assert 1+S == Q(18, 13)
print("Posterior means (center, group 1, group 2):", means)
print("Group covariance matrix:", [row[1:] for row in cov[1:]])
print("New-group parameter variance:", 1+S)
```

## Questions for research use

1. Do independent conditional posteriors remain independent after integrating μ? **No: shared-center uncertainty induces dependence.**
2. What does replacing μ by m_μ omit here? **The term (1−w_j)(1−w_k)S.**
3. What distinctions belong in a new-region error report? **Observed versus new groups, parameters versus new observations, and fixed versus inferred hyperparameters.**

For hierarchical structure, see the [Stan User’s Guide, Hierarchical regression](https://mc-stan.org/docs/stan-users-guide/regression.html#hierarchical-regression). The Gaussian covariance and fraction example above are derived for the stated model.

## Exercises with explanations

1. What is data weight at n=4,τ=1? **0.8.**
2. Does reducing τ increase reliance on group data? **No: it increases shrinkage toward the common center.**
3. Does the plot incorporate hyperparameter uncertainty? **No: it is a conditional calculation with fixed hyperparameters.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
