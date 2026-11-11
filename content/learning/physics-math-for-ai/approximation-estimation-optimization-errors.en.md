## Different errors compare different objects

Let X be equally likely −1 or +1 and Y=X. Truth has zero risk, but restricting hypotheses to constants gives

$$
R(c)=E(c-X)^2=1+c^2,\qquad\inf_cR(c)=1.
$$

Approximation excess is one. Among sixteen observations with K positives, μ=2K/16−1. Empirical risk Rₙ(c)=1+c²−2μc is minimized at μ, whose realized population estimation excess is μ². Under iid fair sampling, its expectation is 1/16; a single dataset need not equal that.

## Optimize the empirical objective

For loss Rₙ/2, gradient is c−μ. After t steps,

$$
c_t=\mu+(c_0-\mu)(1-\eta)^t.
$$

Empirical optimization excess is (cₜ−μ)², always nonnegative. Population difference to ERM is cₜ²−μ², which can be negative. Calling them the same quantity creates a false decomposition.

## An exact counterexample

At K=12, μ=.5. Start at c₀=0 with zero steps. Approximation is one, ERM estimation .25 and **signed population difference −.25**, totaling risk one. Empirical optimization excess is instead .25.

$$
R(c_t)=1+\mu^2+(c_t^2-\mu^2).
$$

With η=.5 and eight steps, empirical optimization excess falls to about 3.81×10⁻⁶ while population risk rises to about 1.248. Better optimization of this dataset can move away from the population optimum. This does not prove early stopping universally beneficial.

## A nonnegative bound is not an equality

A standard argument uses empirical optimization error and uniform deviation:

$$
R(c_t)-\inf_HR\le2\sup_{c\in H}|R(c)-R_n(c)|+\epsilon_{\rm opt}.
$$

Here H restricts c to [−1,1]. Adding approximation excess gives a nonnegative upper bound, not an exact sum of three nonnegative errors. Blue shows population risk, orange empirical risk and the dot the current population value. The table separates the signed identity from empirical optimization excess.

## Prove the comparison bound by inserting empirical risks

Let Δ=sup_H|R−R_n| and suppose R_n(h_t)≤inf_H R_n+ε_opt. For a population class minimizer h_H, add and subtract R_n(h_t) and R_n(h_H). The empirical-risk difference is at most ε_opt and the two deviations are at most Δ each. Therefore

$$
R(h_t)-R(h_H)\le2\Delta+\varepsilon_{\rm opt}.
$$

If a minimum is not attained, use an ε-minimizer and take ε to zero. This proof does not say the population optimization difference is nonnegative. In the live constant class, Δ=2|μ| because R(c)−R_n(c)=2μc on [−1,1]. The bound is valid but can be loose. Adding the approximation excess one gives a total-risk upper bound, not an exact three-positive-term decomposition.

## Average the optimization path over all training datasets

For a deterministic initial c₀ and fixed η, put γ=(1−η)ᵗ. Then c_t=γc₀+(1−γ)μ. Since Eμ=0 and Eμ²=1/16 under sixteen iid fair signs,

$$
E R(c_t)=1+\gamma^2c_0^2+\frac{(1-\gamma)^2}{16},
\qquad E\varepsilon_{\rm opt}=\gamma^2\left(c_0^2+\frac1{16}\right).
$$

The cross term vanishes in this expectation, not in every realized dataset. For c₀=1 and η=1/2, the table averages the exact binomial probabilities. The expected ERM risk is 17/16=1.0625.

| Steps t | Expected population risk | Expected empirical optimization excess |
|---|---|---|
| 0 | 2 | 1.0625 |
| 1 | 1.265625 | .265625 |
| 4 | 1.058838 | .0041503906 |
| 16 | 1.062498 | 2.4738256×10⁻¹⁰ |

![Expected risk decreases and then approaches the ERM risk from below while empirical optimization excess keeps falling, for the stated fixed binary-data model and initial constant one.](/assets/learning/approximation-expected-risk-en.svg)

The best continuous γ is 1/(16c₀²+1), here 1/17, close to t=4. This illustrates an early-stopping tradeoff for this exact distribution and initialization; it is not an algorithm for knowing the best stopping time in an unknown population. If c₀=0, the initialization already equals the population-optimal constant, changing the tradeoff completely. The code iterates GD for every count K and averages with C(16,K)/2¹⁶ rather than assuming one chosen K is typical.

**Exercise:** Can expected optimization excess be zero at t=16 here? **Explanation:** No. It is small and positive; rounding the display to six decimals would hide it.

The live experiment also retains the analytic displacement instead of reconstructing it from the rounded current constant.

$$
d_t=(c_0-\mu)(1-\eta)^t,\qquad
\varepsilon_{\rm opt}=d_t^2,\qquad
R(c_t)-R(\mu)=d_t(2\mu+d_t).
$$

At K=12,c₀=.5+2⁻⁵³,η=.5,t=16, the binary64 current constant rounds to .5, while displacement is 2⁻⁶⁹ and empirical excess is the positive 2⁻¹³⁸. Limited digits in the current constant or risk are different from exact optimization. An empirical excess or nonzero population difference lost to numeric range is displayed as — with a note. A true zero from starting at ERM, and equal population risks at opposite constants, remain zero.

```python
from fractions import Fraction as F
from math import comb

eta=F(1,2)
c0=F(1)
for steps in [0,1,4,16]:
    expected_risk=F(0)
    expected_empirical_excess=F(0)
    for k in range(17):
        mu=F(2*k,16)-1
        c=c0
        for _ in range(steps): c-=eta*(c-mu)
        probability=F(comb(16,k),2**16)
        expected_risk+=probability*(1+c*c)
        expected_empirical_excess+=probability*(c-mu)**2
    gamma=(1-eta)**steps
    assert expected_risk==1+gamma**2*c0*c0+(1-gamma)**2/16
    assert expected_empirical_excess==gamma**2*(c0*c0+F(1,16))
    print(steps,round(float(expected_risk),6),format(float(expected_empirical_excess),'.8g'))

mu=F(1,2)
initial=F.from_float(.5+2**-53)
displacement=(initial-mu)*F(1,2)**16
assert displacement==F(1,2**69) and displacement**2==F(1,2**138)
print('Analytic displacement and empirical excess:',float(displacement),float(displacement**2))
```

## Exercises with explanations

1. Estimation excess when K=8? **Zero**, since ERM equals the population-optimal constant.
2. Does zero empirical optimization excess imply zero total risk? **No.** Approximation and sampling effects remain.
3. Is 1/16 the value for the selected K? **No.** It averages all possible iid fair training realizations.

Further reading: [MIT 9.520, Statistical Learning Theory](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/).
