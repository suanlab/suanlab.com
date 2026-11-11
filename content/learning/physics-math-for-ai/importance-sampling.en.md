## Estimate a target mean using another distribution

Target masses p=(0.2,0.3,0.5) live on states i=0,1,2, with f(i)=i and exact expectation 1.3. If direct target sampling is difficult, draw from proposal q and correct by p/q. The default proposal is uniform. Changing q(0) divides the remaining mass equally between the other two states.

$$
E_p[f]=E_q\!\left[\frac{p(X)}{q(X)}f(X)\right].
$$

## Check support first

For the screen's normalizer and self-normalized estimates, every positive-target state must be reachable by the proposal. At q(0)=0, target mass 0.2 is permanently missed. Even a displayed sample containing only the other states cannot justify a self-normalized estimate of the full target mean. The lab withholds estimates in this case and separately reports whether the displayed draw itself is possible under q.

## Fixed samples and two estimates

The screen does not generate fresh random draws. It displays three fixed count patterns for six samples imagined to come from the proposal. The ordinary importance estimate of the normalizing constant is the mean weight. The self-normalized mean divides a weighted sum by the total weight.

$$
\widehat Z=\frac1N\sum_s w_s,\qquad
\widehat\mu_{\rm SN}=\frac{\sum_s w_sf(X_s)}{\sum_sw_s}.
$$

A uniform proposal and two samples per state give weights 0.6,0.9,1.5 and estimate exactly 1.3. This is a property of these balanced counts, not zero error for every possible draw. Self-normalized ratios are generally biased at finite sample size. By contrast, an ordinary importance mean for normalized p is unbiased under the support and integrability conditions.

## What ESS can miss

Weight effective sample size is (Σw)²/Σw². It summarizes weight concentration and lies between one and six for six positive-weight samples. The third pattern repeats state two six times. All sample weights are equal, giving ESS six, yet the estimated mean is two. A high weight ESS cannot detect unvisited regions or model error.

$$
\operatorname{ESS}=\frac{(\sum_s w_s)^2}{\sum_s w_s^2}.
$$

An extremely small positive proposal probability differs from missing support. The implementation calculates ratios and ESS using relative log weights. Raw weights or normalizer estimates outside numeric range are withheld with a numeric-range warning.

## Ask which estimator the proposal should serve

This comparison concerns iid draws X_1,…,X_N~q, rather than the fixed counts in the screen. Keep normalized p=(1/5,3/10,1/2), f(i)=i and μ=13/10. Initially assume every q_i>0, making all weights w_i=p_i/q_i and moments finite. If V(q) is the ordinary estimator's single-draw variance, its exact N-draw variance is V(q)/N.

$$
V(q)=\sum_i\frac{p_i^2f_i^2}{q_i}-\mu^2,\qquad
A(q)=\sum_i\frac{p_i^2(f_i-\mu)^2}{q_i}.
$$

A(q) is the self-normalized estimator's CLT coefficient. Apply the delta method to its numerator and denominator sample means: the linearized error averages w(X)(f(X)−μ), which has mean zero and variance A(q). Thus √N(μ̂_SN−μ) converges in law to N(0,A(q)). Do not treat A(q)/N as an exact variance for every finite N. The finite-state, fixed positive-q assumptions here provide the required CLT and law-of-large-numbers conditions.

## Solve the two objectives with Cauchy–Schwarz

For positive q summing to one, (Σ|a_i|)²≤Σa_i²/q_i. Taking a_i=p_i f_i gives an ordinary-mean optimum proportional to p_i|f_i|. Taking a_i=p_i(f_i−μ) gives a self-normalized asymptotic optimum proportional to p_i|f_i−μ|. The normalizing sum must be positive; zero terms require separate support checks.

Here the second proposal is q_SN=(13/35,9/70,1/2), attaining minimum A=(7/10)²=49/100. Constructing it exactly requires the unknown μ, so this is not an algorithm that reveals an unknown mean for free. Pilot or adaptive proposals require their own sampling and dependence analysis.

In the table, V is the ordinary single-draw variance and A the self-normalized CLT coefficient. “Target” means q=p and “SN” means q_SN above.

| Proposal | V | A |
|---|---|---|
| Uniform | 1.58 | 0.5946 |
| Target | 0.61 | 0.61 |
| SN | 1.01 | 0.49 |

A proposal best for self-normalization need not minimize ordinary-mean variance. Also notice that these objectives depend on f, while weight ESS does not.

## Support depends on the quantity being estimated

The formal ordinary-mean optimum is q_*=(0,3/13,10/13). At state zero, p_0 f_0=0, so omitting this state loses no contribution to this particular ordinary integral. On sampled states w_i f_i=13/10 is constant, giving zero variance. However, the same proposal cannot recover p's normalizer or a general self-normalized mean. Its E_q[w]=4/5, making the self-normalized ratio converge to (13/10)/(4/5)=13/8, the wrong target.

To preserve full support, take q_ε=(ε,(1−ε)3/13,(1−ε)10/13), with 0<ε<1. Ordinary single-draw variance is μ²ε/(1−ε)>0 and tends to zero as ε decreases to zero. Hence zero is an unattained infimum within the strictly positive proposal family. The screen jointly reports normalization, self-normalization and ESS, so it continues to require full target support.

## Enumerate every two-draw outcome

For N=2 there are only nine ordered pairs. Weight each pair by q_i q_j to compute expectations and variances with exact fractions. Under the uniform proposal the self-normalized expectation is 481/420, differing from 13/10. Under q=p all weights are one, so self-normalization reduces to the ordinary sample mean and is unbiased. “Generally biased” does not mean biased under every proposal.

```python
from fractions import Fraction as Q
from itertools import product

p = [Q(1, 5), Q(3, 10), Q(1, 2)]
mu = sum(p[i]*i for i in range(3))
plans = {"uniform": [Q(1, 3)]*3, "target": p,
         "SN": [Q(13, 35), Q(9, 70), Q(1, 2)]}
for name, q in plans.items():
    w = [p[i]/q[i] for i in range(3)]
    V = sum(q[i]*(w[i]*i-mu)**2 for i in range(3))
    A = sum(q[i]*w[i]**2*(i-mu)**2 for i in range(3))
    rows = []
    for i, j in product(range(3), repeat=2):
        ordinary = (w[i]*i+w[j]*j)/2
        ratio = (w[i]*i+w[j]*j)/(w[i]+w[j])
        rows.append((q[i]*q[j], ordinary, ratio))
    e_is = sum(prob*x for prob, x, _ in rows)
    e_sn = sum(prob*x for prob, _, x in rows)
    var_is = sum(prob*(x-e_is)**2 for prob, x, _ in rows)
    var_sn = sum(prob*(x-e_sn)**2 for prob, _, x in rows)
    assert e_is == mu and var_is == V/2
    if name == "uniform":
        assert e_sn == Q(481, 420) and A == Q(2973, 5000)
    if name == "target":
        assert e_sn == mu and V == A == Q(61, 100)
    if name == "SN":
        assert V == Q(101, 100) and A == Q(49, 100)
    print(name, "V, A, exact N=2 SN mean and variance:", V, A, e_sn, var_sn)
for eps in [Q(1, 2), Q(1, 10), Q(1, 100)]:
    q = [eps, (1-eps)*Q(3, 13), (1-eps)*Q(10, 13)]
    V = sum(p[i]**2*i*i/q[i] for i in range(3))-mu*mu
    assert V == mu*mu*eps/(1-eps)
    print("epsilon, ordinary variance:", eps, V)
q_star = [Q(0), Q(3, 13), Q(10, 13)]
assert all(p[i]*i/q_star[i] == mu for i in [1, 2])
assert sum(p[i] for i in [1, 2]) == Q(4, 5)
print("Missing-support SN limit:", mu/Q(4, 5))
```

## Questions for research use

1. Can an ordinary-mean optimum be reused for self-normalization? **Check the different objective and support requirements first.**
2. Is A(q)/N the exact N=2 variance? **No: distinguish the exact enumeration from the CLT coefficient.**
3. Is ESS alone enough to choose q? **Include the integrand, error, support and computational cost.**

For the general background, see [Art Owen, Monte Carlo theory, methods and examples, §9.1–9.3](https://artowen.su.domains/mc/Ch-var-is.pdf). The three-state fractions, enumeration and full-support q_ε calculation are computed directly for this example.

## Exercises with explanations

1. Can more samples repair missing proposal support? **No: repair the proposal itself.**
2. Is the self-normalized estimate always unbiased? **No: expectation of a ratio needs care.**
3. Does ESS six imply accuracy? **No: inspect the repeated-state counterexample.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
