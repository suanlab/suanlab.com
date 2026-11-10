## What concentration around a mean means

Let independent signs Xᵢ take values −1 and one with equal probability. A single draw lies at an endpoint, but averages often lie near zero. Concentration does not say every realization equals the expectation; it describes the probability of exceeding a specified deviation.

$$
E[\bar X_n]=0,\qquad\operatorname{Var}(\bar X_n)=\frac1n,\qquad \bar X_n=\frac1n\sum_{i=1}^nX_i.
$$

## Compute an exact binomial tail

The positive-sign count K is Binomial(n,.5), and the average is 2K/n−1. Enumerate possible averages and their masses. The plot is an exact sampling law, not a histogram of random draws.

At n=16,t=.5, the event includes K≤4 or K≥12, giving 5034/65536≈.076812. Equality matters: counts four and twelve are included. At t=0, every possible average belongs to the event and probability is one.

$$
P(|\bar X_n|\geq t)=\sum_{k:|2k/n-1|\geq t}\binom nk2^{-n}.
$$

## A variance-only bound

Chebyshev gives P(|X̄|≥t)≤1/(nt²) for t>0. The default value is .25, above the actual probability .0768. It is loose but needs only mean and variance. Displayed probability bounds are capped at one.

## Add independence and boundedness

For independent Xᵢ in [−1,1], Hoeffding gives 2exp(−nt²/2). At n=16,t=.5 this is about .27067, even larger than Chebyshev here. Which bound wins at a particular finite setting differs from whether a bound has exponential decay as n increases.

$$
P(|\bar X_n|\geq t)\leq\min\!\left(1,2e^{-nt^2/2}\right).
$$

## Read the experiment and its scope

Blue is the full average law; orange selects the tail. Increase sample size at fixed threshold and compare exact probabilities with bounds. Similar ideas apply to average losses, measurement noise and minibatch estimates, but dependence or repeated reuse of the same observations can invalidate the stated assumptions. Define randomness, independent units and the deviation event first.

## Exercises with explanations

1. What happens to variance when n doubles? **It halves.**
2. Must a bound equal the true probability? **No: it can be conservative.**
3. May equality be excluded at t=0? **No: the event uses ≥.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
