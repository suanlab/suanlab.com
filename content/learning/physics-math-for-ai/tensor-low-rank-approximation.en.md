## Specify the approximation problem

A low-rank approximation asks how much of an array a small number of components can describe. State both the rank notion and the error norm before comparing answers. Here the components are orthogonal coordinate outer products and the error uses the Frobenius norm, the square root of the sum of squared entries.

$$
X=a\,e_0\otimes e_0\otimes e_0+b\,e_1\otimes e_1\otimes e_1.
$$

The defaults are a=3 and b=1. The plot and table show all eight original entries and the reconstruction. Discrete coordinates are displayed as discrete values, rather than interpolated into a continuous field.

## Compute the discarded energy

Orthogonality makes the squared residual equal the sum of the squared discarded weights. Retaining the first default component therefore gives error one.

$$
\|X\|_F^2=a^2+b^2,\qquad \|X-\widehat X\|_F^2=\sum_{s\text{ discarded}}\lambda_s^2.
$$

Components are selected by absolute magnitude, so a large negative weight is not treated as unimportant. Ties are resolved by coordinate order in the implementation. That convention does not establish a unique optimum: equal magnitudes allow different equally accurate selections.

## Retained energy and the zero case

$$
E_{\mathrm{retained}}=\frac{\sum_{s\text{ retained}}\lambda_s^2}{a^2+b^2}.
$$

Keeping one default component retains 9/10 of the energy. Keeping both retains all of it. Keeping neither gives an error equal to the original norm. If both original weights are zero, the fraction is undefined because its denominator is zero. The interface leaves it undefined rather than inventing a perfect score. The absolute reconstruction error remains zero.

## Where matrix intuition applies

For this special example, an unfolding has singular values |a| and |b|. The matrix rank-one approximation lower bound matches the error of retaining the larger coordinate component. That proves optimality here, rather than for arbitrary CP approximations.

General higher-order CP approximation differs from matrix SVD. A set of tensors with bounded CP rank need not be closed, and a best approximation can fail to exist. Tucker and TT truncations require their own error analyses. Do not transfer the matrix Eckart–Young conclusion to all tensor formats without checking assumptions.

For model compression, reconstruction error is only one measurement. Evaluate prediction quality and execution cost separately; a small tensor error does not automatically preserve a task metric.

## Exercises with answers

1. For a=−3 and b=1, which component survives? **The one with weight −3.**
2. What is the retained fraction for a zero input? **Undefined: it is 0/0.**
3. What happens when both components are retained? **The reconstruction error is zero.**

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
