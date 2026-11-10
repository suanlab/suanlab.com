## A consistent distribution over finite function evaluations

A Gaussian process makes function values jointly Gaussian at every finite input set. It needs a mean function and a PSD covariance kernel. This lab uses a zero-mean, unit-amplitude RBF prior with independent Gaussian observation noise.

$$
f(X)\sim N(0,K),\qquad y=f(X)+\epsilon,\quad\epsilon\sim N(0,\sigma^2I).
$$

The control is noise **variance**, not standard deviation. Adding a standard deviation directly to the Gram diagonal specifies a different model.

## Condition a joint Gaussian

For a query x*, define k*=k(X,x*) and k**=k(x*,x*). Conditional Gaussian formulas give

$$
\begin{aligned}
m_*&=k_*^T(K+\sigma^2I)^{-1}y,\\
v_*&=k_{**}-k_*^T(K+\sigma^2I)^{-1}k_*.
\end{aligned}
$$

Under unit amplitude and the previous sum-loss convention, the mean equals kernel ridge with λ=σ². The GP additionally supplies function uncertainty and a prior interpretation.

## Latent function versus new observation

Latent variance excludes new measurement noise. A fresh independent noisy observation has variance

$$
\operatorname{Var}(y_*\mid y)=v_*+\sigma^2.
$$

At noise variance .1, the two readouts always differ by .1. Noiseless observed locations have zero latent variance. Away from observations variance often returns toward prior uncertainty, but need not increase monotonically throughout a finite interval.

The implementation uses linear solves. Only rounding-sized negative variances are floored to zero; meaningful negativity raises an error. This numerical correction is not extra observation noise.

## Interpret the interval honestly

Blue is the posterior mean and orange mean ±2√v*. Under the specified Gaussian model, a single fixed query has about 95.45% conditional mass inside this interval. It is not a simultaneous 95% band or a distribution-free guarantee.

With kernel and noise fixed, changing labels changes means but not posterior variances. Heteroscedastic data and uncertain learned hyperparameters require a richer model.

## Exercises with explanations

1. Do matching GP and ridge means imply identical predictive information? **No.** GP uncertainty and assumptions add information.
2. Is variance .1 the same as standard deviation .1? **No:** the latter has variance .01.
3. Must changed labels alter the variance? **Not here.** The fixed-kernel conditional variance does not depend on y.

Further reading: [Rasmussen and Williams, Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
