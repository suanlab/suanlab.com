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

## Exercise and research limits

Set λ=0 and verify that every mode returns y. In elastic net move α between zero and one and recover the ridge and lasso endpoints with the same λ. Next vary the second observation through the threshold λα: explain why the coordinate reaches zero continuously while its formula changes.

For a correlated design matrix X, the loss becomes ‖Xw−y‖²/2 and coordinates are coupled. Applying this componentwise formula directly is generally wrong. Proximal gradient or coordinate descent can solve appropriate larger problems, with step sizes and stopping tests still to be specified.

Sparse coefficients alone do not prove causal relevance, stable feature selection or better test accuracy. Select λ using training/validation information or cross-validation without using the final test labels. This lab demonstrates the optimization geometry, not an empirical generalization guarantee.

Further reading: [Parikh and Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).
