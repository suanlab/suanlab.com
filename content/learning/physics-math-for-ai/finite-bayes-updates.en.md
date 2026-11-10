## Updating the weight of a hypothesis

Suppose we want to infer a coin's success probability. This lab allows only three hypotheses: θ=0.2, 0.5 and 0.8. That is a finite model, not a continuous prior over [0,1]. Prior mass is the weight assigned before observation. Likelihood scores the fixed observed data under each hypothesis. Multiply these quantities, then divide by their total.

$$
p(\theta_j\mid x)=\frac{p(\theta_j)L_j}{Z},\qquad Z=\sum_jp(\theta_j)L_j.
$$

## Work through one ordered observation

Take one particular ordered sample with three successes in four conditionally iid Bernoulli trials. Its likelihood is θ³(1−θ). The three likelihood values are 0.0064, 0.0625 and 0.1024. Under equal prior masses, evidence is their average: Z=0.0571.

| θ | Prior | Likelihood | Approximate posterior |
|---|---|---|---|
| 0.2 | 1/3 | 0.0064 | 0.0374 |
| 0.5 | 1/3 | 0.0625 | 0.3649 |
| 0.8 | 1/3 | 0.1024 | 0.5978 |

If only the success count is observed, multiply every likelihood by the binomial coefficient four. The common factor cancels in the posterior, but evidence becomes four times as large. Always record which data event is being assigned a probability.

## A posterior summary is not a predictive distribution

The highest posterior mass is at θ=0.8. Nevertheless, the probability of the next success is about 0.6681, not 0.8. Prediction averages over remaining parameter uncertainty rather than replacing it with one selected hypothesis.

$$
P(X_{\mathrm{next}}=1\mid x)=\sum_j\theta_jp(\theta_j\mid x).
$$

Blue marks show prior mass and orange marks show posterior mass. Keep the success count fixed and change only the prior mode. The observations remain unchanged while posterior weights change. Distinguish assumptions, observations and inference results.

## Zero prior mass is a strong exclusion

The third example assigns zero prior mass to θ=0.8. Even four successes leave its posterior mass zero. A hypothesis declared impossible cannot regain weight by multiplication with a finite likelihood. This differs from assigning a small positive probability. In research, check whether the model excludes phenomena the data-generating process can actually produce.

## Exercises with explanations

1. What is evidence under the uniform prior? **0.0571.**
2. Is next-success probability the largest allowed θ? **No: the posterior average is about 0.6681.**
3. What is the posterior mass sum? **One, by normalization.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
