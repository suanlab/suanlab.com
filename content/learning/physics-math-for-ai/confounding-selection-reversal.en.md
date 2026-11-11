## A difference between groups is not yet a treatment effect

Imagine comparing failure rates after two system configurations. A more capable configuration might be deployed mainly on difficult tasks. Its observed failure rate can be higher even if switching to it helps every difficulty group. The treated and untreated groups mix different populations. Confounding is a structural explanation for that comparison, not a synonym for any correlation among inputs.

The lab separates two worlds. World A has a common cause Z of treatment X and outcome Y, plus an X→Y effect. World B has independent X and Y, with both influencing selection S. Combining their readouts would mix different data-generating processes. The diagrams and table labels explicitly keep them separate.

## World A: an exact sign reversal

Let Z be Bernoulli(pi), with treatment probabilities e_0 and e_1 in its two strata. The outcome mechanism is

$$
q(z,x)=P(Y=1\mid Z=z,X=x)=b_z+\tau x,
\qquad b_0=0.2,\quad b_1=0.7.
$$

The control restricts tau to [-0.2,0.2], keeping every outcome probability valid. Independent treatment and outcome noises conditional on Z are part of this SCM. Thus changing X within either stratum changes mean Y by exactly tau.

Set pi=0.5, e_0=0.1, e_1=0.9 and tau=-0.1. Among X=1 records, 90% have Z=1; among X=0 records, only 10% do. Therefore

$$
E[Y\mid X=1]=0.9(0.6)+0.1(0.1)=0.55,
$$

$$
E[Y\mid X=0]=0.1(0.7)+0.9(0.2)=0.25.
$$

The crude difference is +0.30. Intervening retains P(Z), giving means 0.35 and 0.45, hence effect -0.10. If Y denotes failure, the helpful treatment appears harmful in the unadjusted comparison.

## Standardization and missing cells

Under consistency, conditional exchangeability given Z, and treatment positivity on target strata, the backdoor adjustment is

$$
E[Y\mid do(X=x)]=\sum_z E[Y\mid X=x,Z=z]P(Z=z).
$$

The implementation obtains the conditional means from its enumerated observational cells before standardizing. It does not silently read the oracle outcome mechanism for missing cells. If e_0=0 and e_1=1 with both strata present, each group lacks one treatment. Adjustment becomes unavailable, displayed as an em dash. The declared SCM still has a causal effect because its equations supply those unobserved outcomes as assumptions. Model-defined and observationally identified are different claims. A stratum with zero target population mass needs no outcome contribution.

## World B: selection creates an association

Now use independent fair X and Y, with no causal arrow between them. Selection has probability one except for (X,Y)=(0,0), where it has probability 1-rho. Conditioning on S=1 gives joint probabilities proportional to

$$
(1-\rho,\ 1,\ 1,\ 1).
$$

At rho=1, the selected sample omits 00. Given X=0, Y must be one; given X=1, Y is one half. The selected difference is -0.5 although the population causal effect is zero. The path X→S←Y is a collider path: selecting its common consequence can induce dependence between otherwise independent causes. This example does not assert that every selection process creates bias or that every collider must be measured.

## Double robustness has an algebraic remainder, not two chances to ignore confounding

For a fixed action x, let q_x(z)=E[Y|X=x,Z=z] and e_x(z)=P(X=x|Z=z). Define an augmented population functional using candidate functions m and g, with g positive on target strata:

$$
\Psi_x(m,g)=E\left[m(Z)+\frac{1\{X=x\}}{g(Z)}(Y-m(Z))\right],
\quad\Psi_x-\mu_x=E\left[\left(1-\frac{e_x(Z)}{g(Z)}\right)(m(Z)-q_x(Z))\right].
$$

Conditioning on Z proves the identity: the residual's expectation is e_x*(q_x−m). The remainder vanishes if either nuisance function is correct. It does not vanish merely because both models fit training data. If g≥eta>0, Cauchy–Schwarz bounds its absolute size by ||g−e_x||₂*||m−q_x||₂/eta. This product motivates rate conditions, but asymptotic inference additionally needs sampling assumptions, variance control and appropriate sample splitting or cross-fitting.

The rational example retains World A's default joint law. Correct outcome means alone or correct propensities alone recover ATE −1/10. Two wrong functions reproduce the biased +3/10 contrast.

| Candidate nuisance functions | Augmented ATE |
|---|---:|
| Correct outcome, wrong propensity | −1/10 |
| Wrong outcome, correct propensity | −1/10 |
| Both wrong | 3/10 |

![Population augmented-mean bias as outcome error varies, with three propensity choices illustrating the product remainder.](/assets/learning/causal-double-robust-en.svg)

The figure isolates one stratum with true propensity 1/5 and outcome mean 7/10; the code below uses the full two-stratum World A.

```python
from fractions import Fraction as F
P=[F(1,2)]*2;e=[F(1,10),F(9,10)];tau=F(-1,10)
q=[[F(1,5),F(1,5)+tau],[F(7,10),F(7,10)+tau]]
wrong=[[F(2,5)]*2 for _ in range(2)]
def augmented(m,g,x):
    true_e=[e[z] if x else 1-e[z] for z in range(2)]
    fitted_e=[g[z] if x else 1-g[z] for z in range(2)]
    value=sum(P[z]*(m[z][x]+true_e[z]/fitted_e[z]*(q[z][x]-m[z][x])) for z in range(2))
    truth=sum(P[z]*q[z][x] for z in range(2))
    remainder=sum(P[z]*(1-true_e[z]/fitted_e[z])*(m[z][x]-q[z][x]) for z in range(2))
    assert value-truth==remainder
    return value
for m,g in ((q,[F(1,2)]*2),(wrong,e),(wrong,[F(1,2)]*2)):
    print(augmented(m,g,1)-augmented(m,g,0))
```

This is a population identity for fixed functions, not a finite-sample unbiasedness claim for nuisances trained on the same records. It also presumes the backdoor identification assumptions: unmeasured common causes, missing support or conditioning on a collider are not repaired by the algebra. Conditional cell ratios now use scaled log masses, retaining supported rare cells whose raw products underflow. Truly absent treatment strata remain unavailable, and World B's selection process remains a separate world.

## Exercise with an explained answer

Keep World A's default values, but randomize treatment by setting e_0=e_1=0.5. Keep rho=1 in World B. Which discrepancy disappears?

**Answer.** In A, both observed treatment groups now have the original 50/50 stratum mixture, so their mean difference becomes -0.1 and agrees with the intervention effect. In B, selection still removes 00; its selected difference remains -0.5. Randomization in one world cannot repair a separate selection mechanism. Run both presets and inspect each world's labeled probabilities.

## Research implications

Before adding adjustment variables to an AI evaluation, draw how difficulty, policy assignment, outcomes and inclusion in the benchmark are generated. A variable associated with both treatment and outcome may be a common cause, a consequence, or a collider. Its role requires substantive assumptions. More conditioning is not a universal improvement. These exact population examples isolate structural issues; finite-sample uncertainty, measurement error and model misspecification require additional analysis.

## Primary references

- Judea Pearl, [A Causal Calculus for Statistical Research](https://proceedings.mlr.press/r0/pearl95a.html): graphical conditions for replacing intervention expressions by observations.
- Miguel Hernán and James Robins, [Causal Inference: What If](https://miguelhernan.org/whatifbook): confounding, selection bias and identification conditions.

The two numerical populations and their separate probability tables are original teaching constructions.
