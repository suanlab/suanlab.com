## A best parameter versus an average over parameters

MLE chooses a parameter maximizing likelihood. Evidence, or marginal likelihood, averages likelihood under the prior. Even an excellent best fit can receive a modest average if little prior mass lies near it. These operations answer different questions.

$$
Z=p(x)=\int p(x\mid\theta)p(\theta)\,d\theta.
$$

## Calculate a Beta-model example

With n=8 observations and k successes, a Beta(α,β) prior gives the evidence of one ordered sample as a Beta-function ratio. At k=6 and α=β=1, Z=1/252≈0.003968 and log Z≈−5.529, using natural logarithms.

$$
Z=\frac{B(\alpha+k,\beta+n-k)}{B(\alpha,\beta)}.
$$

For the event consisting of every ordering with the same count, multiply by C(8,k). In the default example this count-event probability is 1/9. The plot and readouts consistently use one ordered sample.

## Recover evidence from sequential predictions

Read observations one at a time. Multiply the probability of each observation under the posterior available just before it. After s successes and f failures, next-success probability is (α+s)/(α+β+s+f); the failure probability uses the other numerator.

$$
p(x)=\prod_{i=1}^{n}p(x_i\mid x_1,\ldots,x_{i-1}),\qquad \log p(x)=\sum_i\log p(x_i\mid x_{<i}).
$$

The table places successes first. Exchangeability means that reordering can change intermediate predictions while preserving the final product. Blue shows cumulative log predictive probabilities. Orange evaluates the same observations with the full-data MLE fixed from the beginning. That reference uses the whole dataset in advance; it is not an online prediction procedure.

## Prior sensitivity and scope

Change the prior shapes and observe the evidence. Because evidence averages over a prior, it can be sensitive to that choice. An improper unnormalized prior can leave an arbitrary multiplicative constant, preventing an ordinary evidence comparison. This lab uses only proper normalized Beta priors.

Products of tiny probabilities motivate computing sums of logarithms. Here a small integer example allows verification by both a function ratio and a sequential factorization.

## Exercises with explanations

1. Can evidence exceed maximum likelihood? **A likelihood average under a proper prior cannot exceed its maximum.**
2. Does reordering change final evidence? **Not in this exchangeable model.**
3. Is evidence a parameter probability? **No: it is a data-event probability after integrating the parameter.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
