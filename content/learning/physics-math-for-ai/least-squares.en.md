## What if the equations are inconsistent?

Suppose inputs x=1,2,3 have observed responses y=1,2,2. A line through the origin, y=ax, would require a=1, a=1 and a=2/3 simultaneously. No exact solution exists. Approximation is still possible: **define the error to minimize, then find the closest prediction.**

Here we learn just one slope, with no intercept. Data are synthetic teaching examples and all values are dimensionless. Do not confuse this problem with regression including an intercept or observation weights.

## Symbols and two kinds of space

| Symbol | Meaning | Reference value |
| --- | --- | --- |
| xᵢ, yᵢ | Input and response for observation i | `(1,1)`, `(2,2)`, `(3,2)` |
| x | Vector of the three inputs | `(1,2,3)` |
| y | Vector of the three responses | `(1,2,2)` |
| a | One adjustable slope | Initially 0.5 |
| ax | Vector of three predictions | `(0.5,1,1.5)` |
| r=y−ax | Observed minus predicted responses | `(0.5,1,0.5)` |

In the **observation plane**, a point `(xᵢ,yᵢ)` pairs one input with one response. In **response space**, each observation supplies an axis and the entire vector y becomes one point. The 3D labels obs₁, obs₂ and obs₃ refer to this second space, not physical positions.

## Squared residuals, not their signed sum

Opposite signs can cancel when residuals are simply added. Residuals `(1,−1)` sum to zero without representing a perfect prediction. Squaring measures each observation's error magnitude:

$$
L(a)=\tfrac12\sum_{i=1}^{3}(y_i-ax_i)^2=\tfrac12\|\mathbf y-a\mathbf x\|^2.
$$

The positive factor 1/2 simplifies differentiation without changing the minimizing location. Whether loss is summed or averaged changes its numerical scale and learning-rate behavior, so record that convention.

## Check the starting slope 0.5

| Input xᵢ | Observed yᵢ | Prediction 0.5xᵢ | Residual yᵢ−0.5xᵢ | Squared residual |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.5 | 0.5 | 0.25 |
| 2 | 2 | 1 | 1 | 1 |
| 3 | 2 | 1.5 | 0.5 | 0.25 |

Squared error is 1.5 and loss is 0.75. In the observation chart, inspect the vertical segments between points and line. This objective squares **vertical response errors at fixed inputs**, not the shortest geometric distances to the line.

## Solve for one slope

Expand the loss:

$$
L(a)=\tfrac12\left(\mathbf y^T\mathbf y-2a\mathbf x^T\mathbf y+a^2\mathbf x^T\mathbf x\right).
$$

Here `x·x=1+4+9=14`, `x·y=1+4+6=11`, and `y·y=9`, giving

$$
L(a)=\tfrac12(9-22a+14a^2).
$$

The derivative is `14a−11`, so its zero occurs at

$$
a^*=\frac{\mathbf x^T\mathbf y}{\mathbf x^T\mathbf x}=\frac{11}{14}.
$$

A zero derivative does not establish a minimum for an arbitrary function. Here, the quadratic coefficient is positive, so this is the unique global minimum. Completing the square gives `L(a)=7(a−11/14)²+5/28` as a derivative-free check.

## Error remains at the optimum

The optimal prediction and residual are

$$
a^*\mathbf x=(11/14,22/14,33/14)^T,
\qquad \mathbf r=(3/14,6/14,-5/14)^T.
$$

Squared error is `(9+36+25)/196=70/196=5/14`. Loss is half of that, `5/28≈0.178571`. It is smaller than the initial 0.75 but remains nonzero because the inconsistent observations cannot all fit one slope.

## Obtain the same answer by projection

All possible predictions ax lie on the line spanned by x in response space. Projecting y onto that line gives

$$
\operatorname{proj}_{\mathbf x}\mathbf y
=\frac{\mathbf x^T\mathbf y}{\mathbf x^T\mathbf x}\mathbf x=a^*\mathbf x.
$$

Check `x·r=3/14+12/14−15/14=0`. Orthogonal residual and zero loss derivative are two readings of the same condition. Orthogonality does not mean the residual vector itself is zero.

## Extend to several coefficients

With multiple input features, put one observation in each row of design matrix A. A parameter vector w predicts Aw, leading to

$$
\min_{\mathbf w}\tfrac12\|\mathbf y-A\mathbf w\|^2,\qquad
A^T(A\mathbf w-\mathbf y)=0.
$$

The second expression is the normal equation. It makes the residual orthogonal to every column of A. Independent columns give a unique coefficient vector. Dependent columns can give multiple coefficients with the same prediction, although the optimal prediction itself remains the unique orthogonal projection onto the column space.

Writing the normal equation does not make explicit matrix inversion the best numerical implementation. QR, SVD and numerical stability will distinguish existence, uniqueness and computational error.

## Connect regression to AI evaluation

A neural network's final linear layer and simple regression models combine features to predict responses. One feature column lets us follow every calculation and its geometry. To include an intercept, add a column of ones to A; the answer is then not the same slope 11/14 derived here.

Loss is a chosen objective. A large response change can affect the fitted coefficient strongly because errors are squared. The large-observation preset illustrates this calculation; it does not claim which loss performs best on a particular real dataset.

Even perfectly fitting three training points does not establish accuracy on new observations. Model assumptions, data splits, units, preprocessing and evaluation metrics require separate checks.

## Challenge and research checks

1. Responses `(1,2,3)` give optimal slope 1 and zero loss. Why does this differ from the reference example?
2. Calculate the optimal slope for `(1,2,5)`: the numerator is 20 and denominator 14.
3. How do vertical residuals in the observation plane and the orthogonal residual in response space represent the same three numbers?
4. Can you verify intercept, weighting and sum-versus-mean conventions in a research implementation?

Further study: [MIT — Projection Matrices and Least Squares](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projection-matrices-and-least-squares/).
