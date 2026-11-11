## 1. Carry the covariance of the inputs

Two sensors may share a calibration, an environment or a reference clock. Their errors can move together. If the inputs have covariance matrix $\Sigma$, a linear combination $z=w^Tx$ has variance $w^T\Sigma w$. The diagonal entries of $\Sigma$ describe each input's variance. The off-diagonal entries describe their joint variation. [JCGM 100, Guide to the Expression of Uncertainty in Measurement](https://doi.org/10.59161/JCGM100-2008E)

## 2. Compare a sum and a difference

For standard deviations $\sigma_1,\sigma_2$ and correlation $\rho$, the variance of $z_\pm=x_1\pm x_2$ is

$$
\operatorname{Var}(z_\pm)=\sigma_1^2+\sigma_2^2\pm2\rho\sigma_1\sigma_2.
$$

With both standard deviations equal to one and $\rho=0.75$, the sum has variance 3.5 and the difference has variance 0.5. At $\rho=1$ with equal scales the difference variance is zero, because the centered deviations agree. The means can still differ.

## 3. Connect the equation to geometry

A cloud of positively correlated errors stretches along a diagonal. Projecting it onto the sum direction keeps that variation. Projecting onto the difference direction suppresses it. In the lab, the ellipse follows the image of a unit circle under a covariance factor. For joint Gaussian inputs this is a contour of fixed Mahalanobis radius, with a degenerate line at perfect correlation.

## 4. Check where first-order propagation breaks down

For differentiable $f$, linearization gives $\operatorname{Var}(f(X))\approx\nabla f^T\Sigma\nabla f$ near the input mean. For $X\sim\mathcal N(0,\sigma^2)$ and $f(X)=X^2$, the gradient at the mean is zero, yet the true variance is $2\sigma^4$. The curvature creates spread that the tangent approximation misses. Distribution propagation, exact moments, or a suitable Monte Carlo calculation can capture it. [JCGM 101, Propagation of Distributions Using a Monte Carlo Method](https://doi.org/10.59161/JCGM101-2008)

## 5. Verify by propagating samples

```python
import numpy as np

rng = np.random.default_rng(2026)
u, v = rng.standard_normal((2, 100_000))
rho = .75
x = u
y = rho*u + np.sqrt(1-rho**2)*v
assert np.isclose(np.var(x+y), 3.5, rtol=.03)
assert np.isclose(np.var(x-y), .5, rtol=.03)
assert np.isclose(np.var(x**2), 2., rtol=.04)
print(np.var(x+y), np.var(x-y), np.var(x**2))
```

## 6. Match the propagation to the decision

Estimate covariance from paired observations or a documented error model. Carry the same input samples through all outputs that share an error source. Check units and the size of nonlinear effects. Distinguish a standard deviation from a prediction interval and state any distributional assumptions used to construct the latter.
