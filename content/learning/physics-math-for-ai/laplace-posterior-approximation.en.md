## Approximate the neighborhood of a mode

A difficult posterior may still permit calculation of its mode and local curvature. A quadratic expansion of log posterior at the mode produces a Gaussian shape: the core of Laplace approximation. This lab uses an exactly known Beta posterior so the approximation error can be inspected directly.

$$
\log p(\theta\mid x)\approx\log p(\theta_*\mid x)-\tfrac12H(\theta-\theta_*)^2,
\qquad q(\theta)=N(\theta_*,H^{-1}).
$$

## Calculate Beta mode and curvature

There are n=8 observations and a Beta(α,β) prior. Both prior shapes are at least two, ensuring an interior mode for posterior Beta(A,B). Differentiate log density twice to obtain:

$$
\theta_*=\frac{A-1}{A+B-2},\qquad
H=\frac{A-1}{\theta_*^2}+\frac{B-1}{(1-\theta_*)^2}.
$$

At k=4 and α=β=2, the posterior is Beta(6,6), mode 0.5 and H=40. Laplace variance is 0.025; exact variance is 1/52≈0.01923. Matching local curvature does not force all global moments to match.

## Examine skewness and boundaries

At k=0, the mode is 0.1 and the posterior is skewed. Gaussian mean equals that mode, whereas the exact Beta mean is 1/6. Blue plots normalized exact density; orange is an untruncated Gaussian. The horizontal window extends beyond [0,1], revealing Gaussian mass assigned to impossible parameter values.

The table separately reports outside-support mass using a numerical normal CDF. The Gaussian is not silently truncated and renormalized. Doing so would define a different approximation with different moments.

## Coordinate choice and validity

The expansion is performed in θ coordinates. Transforming to a logit coordinate changes density through a Jacobian, so modes and approximations generally change. Do not interpret MAP or Laplace approximation as coordinate-invariant results.

In higher dimensions, inspect whether H is positive definite at a regular interior mode, whether several modes exist, and whether tails are sufficiently Gaussian-like. This example has one interior mode and small integer shapes. Concentration can improve an approximation, but does not automatically solve multimodality or boundary problems.

## Exercises with explanations

1. Is approximate variance H or its inverse? **Its inverse.**
2. Must exact posterior mean equal the mode? **No: inspect the skewed example.**
3. Is the orange density confined to Beta support? **No: it is an untruncated Gaussian.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
