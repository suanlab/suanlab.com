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

The observed null statistic is two. Under a standard normal, the two-sided tail probability is approximately 0.0455. It exceeds the 5% two-sided critical magnitude, so the test rejects and the corresponding interval excludes zero. A p-value measures data extremeness under the null, not the probability that the null is true. Failure to reject is not proof of the null either.

$$
p_{\mathrm{two}}=2\Phi(-|Z_{\mathrm{obs}}|),\qquad\text{reject if }|Z_{\mathrm{obs}}|>z_{1-\alpha/2}.
$$

## The plotted density and scope

Orange tails mark the rejection region, whose area is the nominal significance level. The observed point is drawn only inside the visible window [−4,4]; its numerical readout is never clipped. A 90% interval is narrower because its critical value is smaller. Unknown variance, nonnormality, selected samples, repeated testing or data-dependent hypothesis selection require reviewing the procedure’s guarantees.

## Exercises with explanations

1. What is the p-value at observed mean zero? **One.**
2. What happens to width when n quadruples? **It halves at fixed σ.**
3. Is this a posterior interval? **No: its construction and interpretation differ.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[NIST, Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) — For repeated-coverage interpretation and the unknown-σ t-interval counterpart.
