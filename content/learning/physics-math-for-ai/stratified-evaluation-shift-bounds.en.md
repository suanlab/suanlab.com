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

## Worked exercises and solutions

1. **Exercise:** Why is the missing-group empirical error not zero? **Solution:** It would require dividing zero observed errors by zero observations. There is no empirical rate. A positive deployment weight therefore makes the plug-in target estimate unavailable.
2. **Exercise:** Set the missing group's target weight to zero. Must the plug-in estimate remain unavailable? **Solution:** No. A zero-weight term contributes zero without evaluating its undefined rate. The remaining sampled group supplies the target estimate.
3. **Exercise:** Add 40 examples and 16 errors in group one. **Solution:** The equal-weight empirical estimate is 0.2; the simultaneous numerical upper bound is about 0.327415, and the radius-0.3 bound is about 0.471006. More information removes the unidentified empirical term, although sampling uncertainty remains.
4. **Exercise:** Does doubling a zero-error sample prove universal correctness? **Solution:** No. It reduces an average-risk upper endpoint under a sampling model; the isolated-point counterexample still defeats the universal proposition.
5. **Exercise:** Why can the interval maximum be found without a fine grid? **Solution:** An affine function on a closed interval attains its maximum at an endpoint, including the constant-function case where every point is optimal.
6. **Exercise:** Does a changed orange marker invalidate the confidence method? **Solution:** Not by itself. The marker specifies a different hypothetical truth. A valid procedure can miss on individual samples; assess coverage across the full sampling law rather than judging validity from one queried outcome.

## Sources and research extensions

The [NIST description of exact binomial intervals](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm) explains inversion of the binomial distribution. The two-group union-bound construction and mixture calculation above are explicit derivations for this lesson. A research extension could replace fixed counts with a justified sequential design, investigate dependent errors, or construct a deployment uncertainty set from separate data. Each extension changes the assumptions that must be proved, rather than merely adding a larger confidence percentage to the same formula.
