## An SDE specifies a noise law as well as a drift

Consider the Ornstein–Uhlenbeck equation dX=-theta X dt+sigma dW from deterministic x0. Its exact transition has an exponentially damped initial state plus a weighted Brownian integral. Euler instead freezes the drift over a step. Comparing them strongly requires putting both on the same underlying noise, not merely matching each marginal variance.

The blue and orange paths use the same Brownian increment over each step. A second independent Gaussian supplies the part of the exact weighted integral that cannot be recovered from that increment. This construction makes the meaning of a coupled path error explicit.

## Derive the exact innovation covariance

For step length h, z=theta h and f(z)=(1-exp(-z))/z, with f(0)=1, let I be the integral of exp(-theta(h-s)) against dW_s. Itô isometry and covariance give

$$
\operatorname{Var}(I)=h f(2z),\qquad
\operatorname{Cov}(\Delta W,I)=h f(z).
$$

Hence, using independent standard Gaussians Z,G,

$$
\Delta W=\sqrt h Z,\qquad
I=\sqrt h\left(f(z)Z+\sqrt{f(2z)-f(z)^2}\,G\right).
$$

Giving both methods the identical scalar Gaussian multiplied by their marginal standard deviations would generally produce a different covariance. The extra bridge variance is nonnegative and vanishes as theta tends to zero. A small-argument series evaluates it without cancellation in nearly equal terms.

## Separate a sample error from its expectation

The experiment runs actual exact and Euler recurrences. Separately, it propagates the coefficients of every independent Gaussian in the terminal error. The theoretical mean-square error is the squared mean error plus the sum of squared noise coefficients. It is not a square taken from the one visible realization.

Exact terminal mean is x0 exp(-theta T); variance is sigma² T f(2theta T). At theta zero both methods reduce to x0+sigma W and agree under the coupling. At sigma zero they become a deterministic ODE comparison.

## Grid changes need their own coupling statement

Within one run, exact and Euler share the stated increments. Changing the grid control here regenerates a different sequence of step innovations from the seed, so cross-grid path differences are not presented as a refinement proof. The analytical coupled MSE supplies the grid-dependent error comparison. This differs from the preceding Brownian chapter's shared finest-grid construction.

Strong error concerns a joint coupling, while weak error compares expectations of test functions. For Euler, mean is x0(1-theta h)^N and variance is sigma²h times the sum of (1-theta h)^(2j). They supply, for example, a weak second-moment comparison through E[X²]=mean²+variance. Matching one weak statistic is not a substitute for a correct strong coupling.

## Exercise: test covariance before comparing errors

At theta=sigma=x0=T=1, compute mean exp(-1) and variance (1-exp(-2))/2. For a single step, integrate the exponential kernel and its square to obtain the covariance and variance above. Explain why their ratio is not generally that of perfectly correlated Gaussian innovations. Finally set theta zero and verify the bridge variance, theoretical MSE and observed path difference all vanish, while terminal variance remains sigma² T.

Further reading: [Lawler's diffusion notes](https://www.math.uchicago.edu/~lawler/finbook.pdf). The covariance-preserving coupling and coefficient error calculation are independently derived.
