## Directions preserved by a transformation

A matrix usually changes both a vector's length and its direction. Some vectors, however, remain on the same line after multiplication. Finding these special directions lets us interpret a transformation as directional scaling. **An eigenvector specifies a direction; its eigenvalue specifies the scale factor.**

This unit studies real symmetric 2×2 matrices. Symmetry permits real eigenvalues and an orthonormal eigenbasis; do not assume these properties for arbitrary matrices. Drawings occupy the plane z=0, rather than an entire high-dimensional model representation.

## Symbols and two interpretations

| Symbol | Meaning | Reference example |
| --- | --- | --- |
| A | Symmetric matrix being analyzed | `[[2,1],[1,2]]` |
| q | Nonzero eigenvector | `(1,1)/√2` |
| λ | Scaling factor in Aq=λq | 3 or 1 |
| C | Covariance of centered data | Set equal to A in the example |
| PC₁ | Unit direction with greatest variance | `(1,1)/√2` |

A matrix acting as a linear map and the same numbers serving as a covariance are different interpretations. Covariance must be positive semidefinite, not merely symmetric. The experiment separates these readings into two views.

## Check the eigenvalue equation

An eigenvector q and eigenvalue λ satisfy

$$
A\mathbf q=\lambda\mathbf q,\qquad \mathbf q\ne\mathbf0.
$$

Zero is excluded because A0=λ0 holds for every λ and reveals no direction or scale factor.

The reference matrix sends `(1,1)` to `(3,3)`, so `(1,1)` is an eigenvector with eigenvalue 3. It sends `(-1,1)` to `(-1,1)`, giving eigenvalue 1. Divide each by √2 to obtain orthogonal unit eigenvectors:

$$
\mathbf q_1=\frac1{\sqrt2}(1,1)^T,\quad
\mathbf q_2=\frac1{\sqrt2}(-1,1)^T,
\quad \mathbf q_1^T\mathbf q_2=0.
$$

Replacing q by −q gives the same eigendirection. Opposite signs in two implementations' eigenvectors do not by themselves establish different results.

## Find the eigenvalues

Rearrange the equation as `(A−λI)q=0`. A nonzero solution requires A−λI to be singular. For this square matrix,

$$
\det(A-\lambda I)=(2-\lambda)^2-1
=\lambda^2-4\lambda+3=(\lambda-3)(\lambda-1)=0.
$$

Thus the eigenvalues are 3 and 1. I is the identity, and λI has λ on each diagonal entry. Hand-expanding this polynomial is a small-example derivation, not a general prescription for computing large matrix eigensystems.

For a real symmetric 2×2 matrix `[[a,b],[b,d]]`,

$$
\lambda_{1,2}=\frac{a+d\pm\sqrt{(a-d)^2+4b^2}}2.
$$

The experiment sorts the larger eigenvalue first. Repeated eigenvalues make the choice of directions nonunique.

## Decompose the transformation by direction

Collect the orthonormal eigenvectors as columns of Q:

$$
A=Q\begin{pmatrix}\lambda_1&0\\0&\lambda_2\end{pmatrix}Q^T.
$$

Read a vector in eigenbasis coordinates, multiply each coordinate by its eigenvalue, then return to the original coordinates. This explains A's action on every vector, rather than decorating the scene with one special arrow.

In **Matrix action**, A sends the dashed unit circle to the green ellipse. Half-axis lengths are `|λ₁|`, `|λ₂|`. A negative eigenvalue reverses its direction, so inspect the input and output arrows as well as the ellipse outline.

## Center before measuring spread

Subtract mean μ from each data point zᵢ. This unit uses denominator-n covariance:

$$
\boldsymbol\mu=\frac1n\sum_i\mathbf z_i,\qquad
C=\frac1n\sum_i(\mathbf z_i-\boldsymbol\mu)(\mathbf z_i-\boldsymbol\mu)^T.
$$

Diagonal entries are feature variances; off-diagonal entries measure joint variation. An uncentered second-moment matrix is generally different. Also distinguish the n−1 denominator used in sample statistics. Changing only that denominator rescales all eigenvalues together but preserves directions and explained-variance ratios.

Variance after projection onto a unit direction w is `wᵀCw`. For covariance eigenvector qⱼ, this becomes λⱼ. It is in this interpretation that an eigenvalue represents directional variance.

## Why does the first component maximize variance?

Write any unit direction as `w=αq₁+βq₂`, with `α²+β²=1`. Orthogonality and the eigenvalue equation give

$$
\mathbf w^TC\mathbf w=\lambda_1\alpha^2+\lambda_2\beta^2\leq\lambda_1.
$$

Since λ₁≥λ₂, direction q₁ achieves maximum variance λ₁. This derives the first principal component in the example. Repeated eigenvalues allow other directions to attain the same maximum.

Total reference variance is `trace(C)=2+2=3+1=4`. Keeping the first component retains variance 3:

$$
\text{First-component variance ratio}=\frac3{3+1}=0.75.
$$

The centered one-component reconstruction is `q₁q₁ᵀ(z−μ)`. Its mean squared reconstruction error is the discarded directional variance, 1. This is a result about a specified linear reconstruction problem, not 75% accuracy on an AI task.

## Read the covariance ellipse and point model

In **Covariance point model**, the green reference ellipse has eigenvector half-axes `√λ₁`, `√λ₂`. These differ from `|λ|` in the matrix-action view. Variance has squared coordinate units, so a length scale uses its square root. Without additional distributional assumptions, the ellipse is not a confidence region.

The 32 blue synthetic points lie on a regular ring √2 times larger than this reference ellipse. This construction has zero mean and denominator-n covariance equal to C. The points are neither Gaussian samples nor measured observations. Compare the computed synthetic mean and C11, C12, C22 with `(0,0)` and `(2,1,2)`.

These are mathematical properties of the construction. Floating-point calculations can display very small residuals instead of exact zeros.

Checking actual point moments separately from a diagram's scale prevents confusion between probability models that merely look similar.

## Repeated, zero and negative eigenvalues

| Matrix | Eigenvalues | Interpretation |
| --- | --- | --- |
| `[[2,1],[1,2]]` | 3, 1 | The first direction has greater variance |
| `[[2,0],[0,2]]` | 2, 2 | Every unit direction has equal variance; no unique first direction |
| `[[1,2],[2,1]]` | 3, −1 | Symmetric but invalid as covariance |
| Zero matrix | 0, 0 | Zero total variance makes variance ratios undefined |

Covariance requires `wᵀCw≥0` for every w. A negative eigenvalue violates that condition along its eigenvector. The experiment reports no covariance variance ratio or square-root geometry in this case. Its numerical sign check uses a tolerance at double-precision roundoff scale for very small negative results.

## AI representations and research checks

PCA reduces representations according to variance after specifying centering, scale and a linear subspace. High variance does not guarantee relevance for classification or prediction. Task information can lie in a low-variance direction.

Different feature units change covariance magnitudes and principal directions. Document whether and why features are standardized. For evaluation, fit the mean and directions on training data, then apply the same transformation to validation data to avoid leakage.

General nonsymmetric matrices can have complex eigenvalues or nonorthogonal eigenvectors. Singular-value decomposition and higher-dimensional PCA implementations follow in later units.

## Challenge and check

1. For `[[3,0],[0,1]]`, what is the first direction and its retained variance fraction? The x-axis and 75%.
2. In the negative-eigenvalue preset, set input angle to 135°. How does Av relate to v? It lies on the same line in the opposite direction.
3. Explain the difference between ellipse half-axis ratio √3 and retained variance ratio 3/4.
4. Can you separately verify centering, scale, denominator, variance ratio and task evaluation when reading a research PCA plot?

Further study: [MIT — Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/), [Stanford — Principal Components Analysis](https://web.stanford.edu/class/stats202/notes/Unsupervised/PCA.html).
