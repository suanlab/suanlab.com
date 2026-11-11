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

## Derive conditioning by isolating independent noise

For positive scales and |ρ|<1, write X=σx U1 and Y=ρσy U1+σy√(1−ρ²)U2 with independent standard normal U1,U2. Define β=ρσy/σx and η=Y−βX. Then η depends only on U2 and is independent of X. Knowing X changes the first term of Y=βX+η but leaves the noise law unchanged. This proves the conditional mean and variance rather than merely substituting into a formula. The residual variance is the scalar Schur complement:

$$
\beta=\frac{\Sigma_{YX}}{\Sigma_{XX}},\quad
\eta=Y-\beta X,\quad
\operatorname{Var}(\eta)=\Sigma_{YY}-\frac{\Sigma_{YX}^2}{\Sigma_{XX}}.
$$

For σx=2, σy=3, ρ=1/2, the covariance rows are (4,3),(3,9). Thus β=3/4 and residual variance is 27/4. At X=2 the conditional mean is 3/2, while its conditional variance remains 27/4. This does not follow from treating density at one observed point as an event probability. It specifies a continuous version of the conditional Gaussian law. At |ρ|=1, η is identically zero instead, agreeing with the singular-law discussion above.

## Connect conditional means to optimal prediction

Because η is independent of X and centered, its cross term with any square-integrable function of X vanishes. For every square-integrable predictor h(X), expanding the squared error gives

$$
E[(Y-h(X))^2]=\operatorname{Var}(\eta)+E[(\beta X-h(X))^2].
$$

Therefore βX minimizes expected squared error among all such predictors, not just linear ones. Restricting h(X)=aX gives the sharper calculation Var(η)+σx²(a−β)². The claim uses joint Gaussian structure to get independent residual noise. For a general distribution with finite second moments and Var(X)>0, covariance still determines the best centered linear predictor, but need not determine the conditional mean or the conditional variance.

## Match the covariance and change the conditional law

Let X take −1,0,1 with equal probability and set Y=X²−2/3. Both means are zero. Symmetry makes covariance zero, while Var(X)=2/3 and Var(Y)=2/9.

| X | Y | Joint point mass |
|---|---|---|
| −1 | 1/3 | 1/3 |
| 0 | −2/3 | 1/3 |
| 1 | 1/3 | 1/3 |

A centered jointly Gaussian pair with this same diagonal covariance would be independent. Its conditional mean would be zero and conditional variance 2/9. In the table's actual law, Y is determined by X: its conditional mean is X²−2/3 and its conditional variance is zero. The best linear prediction has error 2/9; the stated nonlinear prediction has error zero. Matching covariance therefore cannot justify importing a Gaussian conditional law into arbitrary data.

## Check the Schur complement and counterexample exactly

The code applies (X,Y)↦(X,Y−βX) to the first covariance matrix and computes the second example's moments and losses using fractions. Diagonal covariance in the code is an algebraic check; independence in the Gaussian model was justified by its independent-noise construction. The finite counterexample shows why that extra assumption matters. These are separate derivations from the existing ellipse controls, which still display the centered Gaussian model.

```python
from fractions import Fraction as Q

Sigma = ((Q(4), Q(3)), (Q(3), Q(9)))
beta = Sigma[1][0] / Sigma[0][0]
residual = Sigma[1][1] - Sigma[1][0]**2 / Sigma[0][0]
A = ((Q(1), Q(0)), (-beta, Q(1)))
transformed = tuple(tuple(sum(A[i][k]*Sigma[k][l]*A[j][l]
                              for k in range(2) for l in range(2))
                          for j in range(2)) for i in range(2))
assert beta == Q(3, 4) and residual == Q(27, 4)
assert transformed == ((Q(4), Q(0)), (Q(0), residual))
assert beta*2 == Q(3, 2)
print("Gaussian slope, residual variance, mean at x=2:", beta, residual, beta*2)

points = [(Q(x), Q(x*x)-Q(2, 3)) for x in [-1, 0, 1]]
expect = lambda f: sum(f(x, y) for x, y in points) / 3
ex, ey = expect(lambda x, y: x), expect(lambda x, y: y)
vx = expect(lambda x, y: (x-ex)**2)
vy = expect(lambda x, y: (y-ey)**2)
cov = expect(lambda x, y: (x-ex)*(y-ey))
assert (ex, ey, vx, vy, cov) == (0, 0, Q(2, 3), Q(2, 9), 0)
linear_error = expect(lambda x, y: (y-0*x)**2)
nonlinear_error = expect(lambda x, y: (y-(x*x-Q(2, 3)))**2)
assert linear_error == Q(2, 9) and nonlinear_error == 0
for point in points:
    print("non-Gaussian x, y, mass:", *point, Q(1, 3))
print("same covariance, different conditional errors:", linear_error, nonlinear_error)
```

## Further questions and research use

1. Does a conditional variance of zero mean that Y itself has variance zero? **No: Y may vary as a deterministic function of X.**
2. Is the optimal predictor still linear after changing to absolute loss? **The squared-error proof no longer applies; conditional medians are the relevant optimum.**
3. What changes if the observed covariance block is singular? **The division used here is unavailable; support and generalized-inverse conditions need their own treatment.**

For the independent-normal construction, see [Pishro-Nik, §5.3.2](https://www.probabilitycourse.com/chapter5/5_3_2_bivariate_normal_dist.php). The matrix, finite counterexample and loss comparison above are direct calculations. Compare the [joint-table unit](../joint-marginal-conditional/) before assuming independence from covariance.

## Exercises with explanations

1. What happens for scales one and two, correlation one and x=1? **Conditional mean two, variance zero.**
2. Does zero determinant invalidate the law? **No: the planar density formula is unavailable.**
3. Are plotted points samples? **No: they trace a transformed unit-circle boundary.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
