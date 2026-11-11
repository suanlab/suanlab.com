## Ask what an evaluation claim quantifies over

A classifier can pass every recorded test and still fail on an unobserved population. The first task is to name the population and the claim. Average error under a specified distribution, correctness on every possible input, and coverage of a statistical procedure are three different propositions. This unit connects those distinctions to a two-group evaluation experiment that you can calculate by hand.

The controls describe queried counts rather than generating random samples. Set the number of evaluated examples in each group and the number of errors to inspect. A requested count greater than the sample size clamps to that sample size; both requested and effective values remain visible. Changing an illustrative truth world changes the hypothetical data-generating process, not a fitted estimate.

## Separate a universal claim from average risk

Let the true label be minus one for every real input. Use score s(x) equal to minus the absolute value of x, and predict plus one at a zero score. This score can be represented with two ReLU terms. The classifier is incorrect exactly at zero:

$$
s(x)=-\operatorname{ReLU}(x)-\operatorname{ReLU}(-x),\qquad
\Pr_{X\sim\mathrm{Unif}[-1,1]}(\widehat y(X)\ne-1)=0.
$$

Every finite independent sample from that continuous distribution misses the erroneous point with probability one. Nevertheless, the statement that the classifier is correct for every input is false. Even knowing the exact average error would not repair the quantifier mismatch. A logical certificate over an input set must establish a pointwise property, with its own assumptions about the implemented function and the set.

## Specify the sampling model before constructing an interval

For group g, assume a fixed sample size n_g and identically distributed independent Bernoulli error indicators with unknown probability p_g. The observed count has a binomial law. Independence within each group is needed for this model. Duplicate examples, temporal dependence, or training on the evaluation records can invalidate it.

$$
K_g\sim\mathrm{Bin}(n_g,p_g),\qquad
R_q=q_0p_0+(1-q_0)p_1.
$$

The target mixture weight is a deployment quantity. The evaluation allocation n_0 divided by the total sample size need not equal that weight. Pooling counts answers an allocation-weighted question, whereas target risk uses deployment weights. Keep these quantities separate even when both happen to produce small numbers.

## Invert the binomial distribution for an upper bound

Write F_p(k) for the binomial probability of at most k errors. A one-sided Clopper–Pearson upper endpoint U solves the following equation when the count is below the sample size:

$$
F_U(k)=\beta,\qquad
U(n,n;\beta)=1,\qquad U(0,0;\beta)=1.
$$

For zero errors and positive sample size, the closed form is U equal to one minus beta raised to the power one over n. Zero errors do not imply zero risk. The endpoint shrinks as more zero-error evidence is collected, but remains positive for any finite positive sample size and positive confidence budget.

The numerical implementation uses a log-domain binomial sum and bisection, with explicit boundary cases. These displayed decimal endpoints are numerical approximations; the implementation is not an interval-arithmetic proof of every final machine bit. Coverage means that repeated samples yield an interval containing the fixed parameter at least the advertised fraction of the time. It is not a posterior probability that one already displayed interval contains the parameter.

## Obtain a simultaneous two-group statement

Allocate beta equal to alpha over two to each group. The probability that at least one upper endpoint misses its true parameter is at most alpha, by the union bound:

$$
\Pr\{p_0\le U_0,\ p_1\le U_1\}\ge1-\alpha.
$$

This argument does not require independence between groups. It does require each marginal binomial interval to be valid under its own sampling design. On the simultaneous event, every nonnegative mixture of the two conditional risks is bounded by the corresponding mixture of endpoints. No extra confidence penalty is needed merely to inspect many mixture weights after obtaining this simultaneous rectangle.

## Let an unobserved group remain unobserved

Load the first example: n_0 is 40, k_0 is zero, and group one has no samples. With alpha 0.05, the upper endpoints are approximately 0.0880973 and 1. At equal target weights, the upper risk bound is approximately 0.544049. The pooled observed error is zero, but the target-weighted empirical estimate is unavailable because a positive-weight group has no empirical rate.

Now switch between the two truth worlds. They have conditional risks (0.01, 0.01) and (0.01, 0.4). Since group one is never sampled, the entire observable law is identical in both worlds. The equal-weight target risks are 0.01 and 0.205. No estimator based only on those observed records can identify which of these target risks is correct. The unchanged upper bound honestly reflects this missing information; the orange truth marker is an explanatory oracle, not data-derived evidence.

## Optimize over an uncertain mixture

For two groups, total variation between mixture weights is the absolute difference of their group-zero weights. A radius t around q_0 gives a clipped interval for the alternative weight w:

$$
w\in[\max(0,q_0-t),\min(1,q_0+t)],\qquad
\sup_w\{wU_0+(1-w)U_1\}.
$$

The objective is affine, so checking the two interval endpoints suffices. For q_0 equal to 0.5 and radius 0.3, the interval is [0.2, 0.8]. The worst upper bound in the first example is approximately 0.817619. The green segment is valid over this displayed interval, not over every weight outside it.

Transporting this statement to deployment also assumes unchanged conditional error probabilities. If group labels hide a new task, a new labeling process, or changing within-group composition, mixture uncertainty alone does not describe the shift. Choosing a wide radius is not evidence that the real distribution belongs to the assumed family.

## Research extension: Protect every sample size before adaptive stopping

Fixed-count coverage does not justify repeatedly inspecting an interval and stopping when it looks favorable. One transparent, conservative remedy protects all counts in advance. Keep each predictor fixed and imagine a pre-generated iid Bernoulli error stream for each group. Sampling may adaptively choose which group to inspect next, but cannot change these underlying within-group laws.

For total error budget $\alpha$, assign the endpoint for group $g\in\{0,1\}$ at count $n\ge1$ the budget $\beta_{g,n}=\alpha/[2n(n+1)]$. Since $1/[n(n+1)]=1/n-1/(n+1)$, the budgets sum to $\alpha$ over both groups and all counts. The fixed-count binomial guarantee and a countable union bound therefore prove

$$
\Pr\{\forall g\in\{0,1\},\ \forall n\ge1:\ p_g\le U_{g,n}\}\ge1-\alpha.
$$

No independence between the two groups is needed for the union step. The binomial marginal law for every fixed within-group count is still required. Set $U_{g,0}=1$. On this simultaneous event, random observed counts, a stopping decision and any subsequently chosen mixture weights preserve the corresponding risk upper bound. Revising the predictor based on evaluation outcomes would change the experiment and is outside this construction.

![Pointwise and time-uniform upper endpoints for hypothetical zero-error prefixes. The time-uniform construction spends a summable budget over all counts and is wider at a given count.](/assets/learning/privacy-sequential-upper-en.svg)

The cost is conservatism. With $\alpha=0.05$ and true error $p=0.1$ in one group, exact path accounting gives the probability of missing at least once by count $T$:

| Horizon $T$ | Summable-budget procedure | Reusing fixed budget $\alpha/2$ |
|---|---:|---:|
| 100 | 0.000000000 | 0.059944554 |
| 200 | 0.000002488 | 0.088020245 |
| 400 | 0.000003769 | 0.119503932 |

The right column is not a valid 97.5% guarantee over time. A zero in the left column at a finite horizon means no count path can cross yet in this discrete setting, not an unlimited certainty claim. The code propagates exact probability mass over error counts, absorbs first crossings and uses the strict binomial-CDF condition for noncoverage. The unbounded-time theorem comes from the summable union bound; this 400-count computation is an independent finite illustration. More efficient confidence sequences are studied by [Howard and colleagues](https://arxiv.org/abs/1810.08240); this elementary construction is not their optimized boundary.

```python
from fractions import Fraction as F
from math import comb
p=F(1,10); budget=F(1,40)
states=[[F(1)],[F(1)]]; failed=[F(0),F(0)]
for n in range(1,401):
    next_states=[]
    for state in states:
        nxt=[F(0)]*(n+1)
        for k,mass in enumerate(state):
            nxt[k]+=mass*(1-p); nxt[k+1]+=mass*p
        next_states.append(nxt)
    cdf=F(0)
    for k in range(n+1):
        cdf+=comb(n,k)*p**k*(1-p)**(n-k)
        for mode,beta in enumerate([budget/F(n*(n+1)),budget]):
            if cdf<beta:
                failed[mode]+=next_states[mode][k]
                next_states[mode][k]=F(0)
    states=next_states
    for mode in range(2):assert sum(states[mode])+failed[mode]==1
    assert failed[0]<=budget*F(n,n+1)
    if n in [100,200,400]:
        assert failed[1]>budget
        print(n,*(f'{float(v):.9f}' for v in failed))
```

## Worked exercises and solutions

1. **Exercise:** Why is the missing-group empirical error not zero? **Solution:** It would require dividing zero observed errors by zero observations. There is no empirical rate. A positive deployment weight therefore makes the plug-in target estimate unavailable.
2. **Exercise:** Set the missing group's target weight to zero. Must the plug-in estimate remain unavailable? **Solution:** No. A zero-weight term contributes zero without evaluating its undefined rate. The remaining sampled group supplies the target estimate.
3. **Exercise:** Add 40 examples and 16 errors in group one. **Solution:** The equal-weight empirical estimate is 0.2; the simultaneous numerical upper bound is about 0.327415, and the radius-0.3 bound is about 0.471006. More information removes the unidentified empirical term, although sampling uncertainty remains.
4. **Exercise:** Does doubling a zero-error sample prove universal correctness? **Solution:** No. It reduces an average-risk upper endpoint under a sampling model; the isolated-point counterexample still defeats the universal proposition.
5. **Exercise:** Why can the interval maximum be found without a fine grid? **Solution:** An affine function on a closed interval attains its maximum at an endpoint, including the constant-function case where every point is optimal.
6. **Exercise:** Does a changed orange marker invalidate the confidence method? **Solution:** Not by itself. The marker specifies a different hypothetical truth. A valid procedure can miss on individual samples; assess coverage across the full sampling law rather than judging validity from one queried outcome.

## Sources and research extensions

The [NIST description of exact binomial intervals](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm) explains inversion of the binomial distribution. The two-group union-bound construction and mixture calculation above are explicit derivations for this lesson. A research extension could replace fixed counts with a justified sequential design, investigate dependent errors, or construct a deployment uncertainty set from separate data. Each extension changes the assumptions that must be proved, rather than merely adding a larger confidence percentage to the same formula.
