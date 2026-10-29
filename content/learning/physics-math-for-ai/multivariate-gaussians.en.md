## Separate scales and correlation

A centered bivariate Gaussian covariance is determined by two positive standard deviations and a correlation between minus one and one. These restrictions ensure positive semidefiniteness. The defaults use unit standard deviations and correlation 0.5, so the off-diagonal covariance is 0.5. Remember that diagonal covariance entries are variances, not standard deviations.

$$
\Sigma=\begin{pmatrix}\sigma_x^2&\rho\sigma_x\sigma_y\\\rho\sigma_x\sigma_y&\sigma_y^2\end{pmatrix},\qquad\det\Sigma=\sigma_x^2\sigma_y^2(1-\rho^2).
$$

## The ellipse is not a sample cloud

Transforming an independent standard normal vector U by L gives covariance LLᵀ. The drawing transforms the unit-circle boundary; it is not random samples. Equal axis scales preserve the geometric tilt. In the nonsingular case this boundary has Mahalanobis radius one, enclosing about 39.3% probability in two dimensions. It is not the roughly 68% interval from the one-dimensional standard normal.

$$
L=\begin{pmatrix}\sigma_x&0\\\rho\sigma_y&\sigma_y\sqrt{1-\rho^2}\end{pmatrix},\quad Z=LU,\quad P(\|U\|\le1)=1-e^{-1/2}.
$$

## Conditional mean and remaining uncertainty

For a jointly Gaussian pair, conditioning on X=x gives a mean linear in x and a variance independent of x. At the default x=1 these are 0.5 and 0.75. Zero correlation leaves the original variance and zero mean, and the pair is independent. For non-Gaussian laws, zero correlation alone does not imply independence.

$$
E[Y\mid X=x]=\rho\frac{\sigma_y}{\sigma_x}x,\qquad\operatorname{Var}(Y\mid X=x)=\sigma_y^2(1-\rho^2).
$$

## Singular laws and the meaning of conditioning

At correlation plus or minus one, the curve collapses onto a line. The Gaussian law remains valid but has no density with respect to planar area. The lab does not fabricate an inverse covariance or two-dimensional density. Y becomes a linear function of X, giving zero conditional variance. Since a continuous X assigns zero point probability, the conditional formula denotes a regular conditional Gaussian version rather than an elementary event ratio. Distinguish support dimension from the reference measure of a density.

## Exercises with explanations

1. What happens for scales one and two, correlation one and x=1? **Conditional mean two, variance zero.**
2. Does zero determinant invalidate the law? **No: the planar density formula is unavailable.**
3. Are plotted points samples? **No: they trace a transformed unit-circle boundary.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
