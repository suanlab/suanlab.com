## The exponential-family structure

For 0<p<1, a Bernoulli model has finite natural parameter η=log(p/(1−p)) and exponential-family form. The likelihood of four independent trials connects data to the parameter only through their sum T. The table enumerates all sixteen ordered samples, separating changing model probabilities from sample order.

$$
f_\eta(x)=\exp\{\eta T(x)-4A(\eta)\},\quad T(x)=\sum_i x_i,\quad A(\eta)=\log(1+e^\eta).
$$

## What sufficiency means

A statistic is sufficient when, given it, the conditional law of the full sample no longer depends on the parameter within the specified model. This does not preserve every data feature. Order adds no information about p in an iid Bernoulli model, but can matter in a temporal dependence model. Sufficiency is relative to the statistical model.

$$
P_p(X=x\mid T=t)=\frac1{{4\choose t}}\quad\text{when }\sum_i x_i=t\text{ and }P_p(T=t)>0.
$$

## Calculate conditional sample probabilities

For p=0.3 and T=2, each matching ordered sample has probability 0.0441. There are six, giving P(T=2)=0.2646. Dividing one sample’s probability by that total gives 1/6, cancelling p. Samples with a different sum receive conditional mass zero. Change p while holding the condition fixed to verify that the matching conditional probabilities remain unchanged.

$$
P_p(T=t)={4\choose t}p^t(1-p)^{4-t}.
$$

## Boundary parameters and information

The boundary values zero and one define valid Bernoulli laws but lie outside the finite-logit natural-parameter domain. Conditioning on T=2 at p=0 is an undefined ratio. The regular Fisher information formula 4/[p(1−p)] is displayed only in the interior. A zero denominator is not a reason to invent finite information or automatically apply the same asymptotic-normality arguments.

$$
I_4(p)=E_p[(\partial_p\log f_p(X))^2]=\frac4{p(1-p)}\quad(0<p<1).
$$

## Exercises with explanations

1. How many ordered samples have T=1? **Four.**
2. Does sufficiency make order irrelevant in every model? **No: it concerns the specified model.**
3. Is a null-event conditional set to zero? **No: it remains undefined.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
