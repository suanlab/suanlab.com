## Conjugacy preserves a distribution family

Put a Beta(α,β) prior on a Bernoulli success probability θ. After k successes in n trials, adding the likelihood exponents to the prior exponents produces another Beta distribution. This preservation is conjugacy. Shapes must be positive; the lab uses integers at least one.

$$
\theta\mid x\sim\operatorname{Beta}(A,B),\qquad A=\alpha+k,\quad B=\beta+n-k.
$$

## Predict a block of four future trials

The observed sample size is fixed at n=8. Predict the success count S in m=4 future trials by integrating the conditional binomial mass against the posterior. The result is Beta-binomial; B denotes the Beta function.

$$
P(S=j\mid x)=\binom mj\frac{B(A+j,B+m-j)}{B(A,B)}.
$$

Four observed successes and a uniform Beta(1,1) prior give Beta(5,5). The future count has mean two and variance 14/11≈1.273. Plugging posterior mean 0.5 into a binomial gives the same mean but variance one. Matching means does not make the full distributions equal.

## Conditional independence does not imply marginal independence

Future trials are independent given θ. After integrating θ, they have positive covariance because they share the same unknown probability. If θ happens to be high, several future trials are more likely to succeed together. Write p=A/(A+B) and s=A+B.

$$
\operatorname{Cov}(Y_1,Y_2\mid x)=\frac{p(1-p)}{s+1},\qquad
\operatorname{Var}(S\mid x)=mp(1-p)\frac{s+m}{s+1}.
$$

## Read the comparison

Blue is the integrated predictive count law. Orange is the posterior-mean plug-in binomial law. With all-success observations, future failure is still possible. Increasing the symmetric prior shapes holds the posterior mean at one half while concentrating the posterior, reducing the difference between the two predictive laws.

All future trials here share one θ. Drawing a fresh independent θ for every trial defines a different generative model. A posterior predictive distribution is also conditional on the chosen likelihood and prior; calculating it does not establish adequacy for real observations.

## Exercises with explanations

1. What is the Beta(5,5) mean? **One half.**
2. Do the two methods differ for a single future Bernoulli trial? **No: only the mean is needed for that single-trial distribution.**
3. Which law can put more mass on extreme multi-trial counts? **The integrated law, reflecting the additional covariance.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
