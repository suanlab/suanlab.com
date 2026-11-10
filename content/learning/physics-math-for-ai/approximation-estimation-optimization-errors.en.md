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

## Exercises with explanations

1. Estimation excess when K=8? **Zero**, since ERM equals the population-optimal constant.
2. Does zero empirical optimization excess imply zero total risk? **No.** Approximation and sampling effects remain.
3. Is 1/16 the value for the selected K? **No.** It averages all possible iid fair training realizations.

Further reading: [MIT 9.520, Statistical Learning Theory](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/).
