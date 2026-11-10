## Three statements that should not be collapsed

A stationary law is invariant under a transition. Mixing describes convergence of marginal distributions toward an invariant law. An ergodic time-average statement concerns averages along trajectories. These ideas interact, but using one word for all three can hide exactly the boundary case a research model depends on.

The experiment uses the two-state chain with transition probabilities a and b. Blue is the actual marginal probability of state one; orange is its unique stationary value when that exists. The sampled time average, its exact expected value and its exact finite-window variance appear separately in the readout.

## The second eigenvalue controls marginal decay

When a+b>0, pi1=a/(a+b). The marginal recurrence gives

$$
p_n-\pi_1=(1-a-b)^n(p_0-\pi_1).
$$

Thus absolute second eigenvalue below one gives convergence in this model. A reducible chain can still have a unique attracting closed class; irreducibility is not identical to this particular convergence flag. When both a and b vanish, every initial distribution is stationary and uniqueness disappears.

## A periodic counterexample has good time averages

At a=b=1 the chain alternates deterministically. From a fixed initial state its marginal distribution keeps oscillating, so it does not mix. Yet every even-length sample window has state-one average exactly one half. Irreducible periodic chains can therefore have time-average laws even though ordinary marginal convergence fails. Specify the convention when using the term ergodicity.

With a=b=0, one path instead stays forever at its randomly selected initial state. A time average is zero or one, not automatically the initial probability p0. Across realizations, its variance remains p0(1-p0) regardless of window length.

## Calculate finite-window uncertainty without cancellation

For this chain, conditional expectation is affine with slope lambda=1-a-b. Centered transitions yield martingale innovations with conditional variance determined by the current state. Expanding a time average into the initial centered state and these orthogonal innovations gives a sum of nonnegative weighted variances.

$$
\operatorname{Var}(\overline X)=w_0^2\operatorname{Var}(X_m)
+\sum_j w_j^2 E[\operatorname{Var}(X_{j+1}\mid X_j)].
$$

The weights are finite geometric sums divided by the sample count. This avoids subtracting nearly equal second moments near the periodic case. Burn-in can also leave no retained observations; then the mean and variance are absent rather than invented as zero.

## Exercise: compare two failures of a convergence slogan

Use a=b=1, p0=1, indices zero through nineteen. Verify the sample mean and its variance are one half and zero while the final marginal differs from pi1. Next use a=b=0 and p0=0.3: the average's variance is 0.21 for every nonempty window. Explain why more samples along one frozen trajectory cannot remove uncertainty about its initial class.

Further reading: [MIT graduate stochastic-process notes](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/). The exact window-variance experiment is independently derived.
