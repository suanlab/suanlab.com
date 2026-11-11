## Compare penalties under one loss convention

Regularization trades data fit against a preference for certain parameters. Comparisons are meaningful only after fixing the loss scaling and the design matrix. This lab uses two observations and the identity design, so the loss is half the squared Euclidean distance from w to y:

$$
\min_w\frac12\|w-y\|_2^2+
\lambda\left[\alpha\|w\|_1+\frac{1-\alpha}{2}\|w\|_2^2\right].
$$

Ridge fixes α=0, lasso fixes α=1, and elastic net uses the mixture control. The mixture control is unused in the first two modes. All three share the same data-fit normalization, avoiding an accidental factor-of-two difference in the interpretation of λ.

## Derive the coordinate solution

Because the design is identity, the objective separates by coordinate. At a nonzero coordinate, stationarity determines the sign and shrinkage. At zero, use the interval subgradient of absolute value rather than setting its derivative to zero. The result is

$$
w_i=\frac{\operatorname{sign}(y_i)\max(|y_i|-\lambda\alpha,0)}{1+\lambda(1-\alpha)}.
$$

The numerator can become exactly zero. The denominator continuously shrinks surviving coordinates. Ridge has no threshold for positive finite λ, whereas lasso can remove a nonzero observation entirely. An exactly zero observation can of course remain zero under ridge as well.

## Three worked fits

With y=(2,0.25), ridge at λ=1 gives (1,0.125). Lasso at λ=0.5 gives (1.5,0). Elastic net at α=0.5 and λ=0.5 gives (1.4,0). Compare the fit loss, weighted penalty and their sum, rather than only the number of nonzero coordinates.

For a zero coordinate, the required lasso subgradient can lie anywhere in [−1,1]. The stationarity residual in the lab uses a permissible choice from this interval. Assigning sign(0)=0 mechanically would produce a false residual whenever a small nonzero observation has been thresholded away.

## What the contour means

The blue contour is a level set of the unweighted penalty passing through the current optimum. It is circular for ridge and diamond-shaped for lasso; elastic net rounds the diamond. The blue vector indicates the target and the orange vector indicates the solution.

The contour radius is not λ. It depends on the penalty value at the optimizer, which itself changes with y and λ. Under suitable convex conditions a penalized optimum can also solve a constrained problem with radius equal to that achieved penalty value, but there is no universal one-to-one conversion between λ and radius.

## A correlated-design lasso needs a dual certificate

For P(w)=‖Xw−y‖²/2+λ‖w‖₁ with λ>0, completing the residual dual gives

$$
D(\theta)=y^T\theta-\tfrac12\|\theta\|^2,\qquad
\|X^T\theta\|_\infty\le\lambda.
$$

For any trial w, scale r=y−Xw by max(1,‖Xᵀr‖∞/λ) to obtain a dual-feasible θ. Since D is 1-strongly concave, the optimal dual point lies within R=√(2(P(w)−D(θ))) of θ. KKT then gives a useful sufficient zero-coefficient certificate:

$$
|x_j^T\theta|+\|x_j\|_2R<\lambda\quad\Longrightarrow\quad w_j^*=0.
$$

The strict margin matters, and floating-point implementations need trustworthy feasibility and gap bounds. This is an optimization screening rule, not a causal or statistical variable-selection guarantee.

The example uses correlated columns X=[[1,1],[0,1]] and y=(2,−1/2),λ=1/2. Near w=(3/2,0), the second column is safely screened. At a poor iterate the bound is too loose to conclude anything. The identity-design controls remain unchanged; their coordinatewise closed form must not be applied blindly to this matrix.

| Trial first coefficient, second fixed at zero | Second coefficient certified zero? |
|---:|---|
| 0 | Not yet |
| 1 | Not yet |
| 1.49 | Yes |
| 1.5 | Yes |

For the research development of gap-based screening, see [Fercoq, Gramfort and Salmon, Mind the Duality Gap](https://proceedings.mlr.press/v37/fercoq15.html). The rational two-column example here is independently constructed.

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
from math import sqrt
lam=F(1,2); y=[F(2),-F(1,2)]
for w1 in [F(0),F(1),F(149,100),F(3,2)]:
    r=[y[0]-w1,y[1]]
    scale=max(F(1),abs(r[0])/lam,abs(r[0]+r[1])/lam)
    theta=[a/scale for a in r]
    assert max(abs(theta[0]),abs(sum(theta)))<=lam
    primal=sum(a*a for a in r)/2+lam*abs(w1)
    dual=sum(y[i]*theta[i] for i in range(2))-sum(a*a for a in theta)/2
    gap=primal-dual
    assert gap>=0
    radius=sqrt(float(2*gap))
    safe=abs(float(sum(theta)))+sqrt(2)*radius<float(lam)
    print(w1,'gap:',float(gap),'radius:',radius,'screen second:',safe)
    assert safe==(w1>=F(149,100))
```

## Exercise and research limits

Set λ=0 and verify that every mode returns y. In elastic net move α between zero and one and recover the ridge and lasso endpoints with the same λ. Next vary the second observation through the threshold λα: explain why the coordinate reaches zero continuously while its formula changes.

For a correlated design matrix X, the loss becomes ‖Xw−y‖²/2 and coordinates are coupled. Applying this componentwise formula directly is generally wrong. Proximal gradient or coordinate descent can solve appropriate larger problems, with step sizes and stopping tests still to be specified.

Sparse coefficients alone do not prove causal relevance, stable feature selection or better test accuracy. Select λ using training/validation information or cross-validation without using the final test labels. This lab demonstrates the optimization geometry, not an empirical generalization guarantee.

Further reading: [Parikh and Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).
