## The question: which directions matter most?

An embedding matrix, an image and a linear layer can all be large arrays. If we retain only a few directions, what exactly is lost? Singular value decomposition (SVD) makes that question precise. It separates a linear map into a choice of input directions, independent stretching and a choice of output directions. Unlike an eigendecomposition, it applies to rectangular matrices too.

The goal is to derive the decomposition from a symmetric problem, calculate an approximation error and recognize when reversing a transformation amplifies noise. The lab uses real 2×2 matrices so every entry can be checked. The theorem itself is not restricted to two dimensions.

## Notation and shapes

| Object | Meaning and shape |
| --- | --- |
| $A$ | A real $m\times n$ linear map |
| $U$ | Orthogonal output directions, $m\times m$ in the full SVD |
| $V$ | Orthogonal input directions, $n\times n$ |
| $\Sigma$ | Rectangular diagonal matrix of nonnegative singular values |
| $\sigma_i$ | Singular values ordered from largest to smallest |
| $r$ | Rank: number of positive singular values in exact arithmetic |

The full factorization is $A=U\Sigma V^T$. A thin version keeps the needed columns; dimensions must still match. Orthogonality preserves Euclidean length, while the diagonal factor changes it. Neither $U$ nor $V$ is uniquely determined when singular values repeat, and paired sign flips leave the product unchanged.

## Derive the input and output directions

For any real vector $v$, $v^TA^TAv=\lVert Av\rVert^2\ge0$. Thus $A^TA$ is symmetric positive semidefinite and admits an orthonormal eigenbasis. Choose $v_i$ satisfying

$$
A^TAv_i=\sigma_i^2v_i.
$$

When $\sigma_i>0$, define $u_i=Av_i/\sigma_i$. For different positive singular directions, their inner products reduce to those of the $v_i$, so these output directions are orthonormal. Complete the zero directions with orthonormal bases as needed. This constructs the SVD and shows why singular values are nonnegative. Negative eigenvalues of $A$ are not negative singular values.

Computing $A^TA$ is useful for this derivation, but it squares a nonzero condition number. Production numerical libraries use more stable algorithms. The tiny teaching implementation is not a general replacement for a numerical linear-algebra package.

## A worked diagonal example

Use $A=\operatorname{diag}(3,1)$. Its input and output directions can be the coordinate axes and its singular values are 3 and 1. A unit circle becomes an ellipse with those half-axis lengths. Both coordinates describe the output space; the picture is not a probability contour.

The decomposition as a sum is

$$
A=3u_1v_1^T+u_2v_2^T.
$$

Keep only the first term. Then $A_1=\operatorname{diag}(3,0)$: the second input direction is lost, and the ellipse collapses onto a line segment. The Frobenius norm of the original is $\sqrt{10}$; the rank-one residual has norm 1. Retaining 90% of squared Frobenius norm does not automatically preserve 90% of downstream prediction quality.

## Why truncation gives the best low-rank approximation

For a rank at most $k$ approximation, the truncated SVD minimizes both spectral and Frobenius error. The errors are

$$
\lVert A-A_k\rVert_2=\sigma_{k+1},\qquad
\lVert A-A_k\rVert_F^2=\sum_{i>k}\sigma_i^2.
$$

A useful lower-bound argument for the spectral result is that a rank-$k$ map must annihilate some nonzero vector in the span of the first $k+1$ right singular vectors. On that subspace, $A$ stretches by at least $\sigma_{k+1}$. Truncation attains the bound. The Frobenius result follows by orthogonal decomposition into singular components; proving the full minimization statement requires the corresponding projection argument, beyond just checking one diagonal example.

These are norm-optimal statements with a specified rank constraint, not universal guarantees for semantic compression or arbitrary task losses.

## Conditioning and inversion

For an invertible square map, $\kappa_2(A)=\sigma_1/\sigma_n$. Replacing the weak value 1 by 0.1 produces condition number 30. Noise along that output direction is divided by 0.1 when inverted. A small residual can therefore coexist with a large parameter error.

When the second singular value is zero, the lab reports no finite condition number. A pseudoinverse can select a minimum-norm solution, but it cannot recover an unobserved component. The rank-one preset has singular values $\sqrt{10},0$ and zero rank-one error. A zero approximation error in a rank-one model does not make its inverse unique.

## Pseudoinverse and minimum-norm least squares

The SVD defines the Moore–Penrose pseudoinverse even for rectangular or singular matrices. Invert positive singular values and leave zero components at zero:

$$
A=U\Sigma V^T,\qquad A^+=V\Sigma^+U^T,\qquad
\Sigma^+_{ii}=\begin{cases}1/\sigma_i&\sigma_i>0,\\0&\sigma_i=0.\end{cases}
$$

The vector x*=A⁺b has the smallest Euclidean norm among least-squares solutions. In the U basis, components along observed directions can be fitted by division by nonzero singular values. Components orthogonal to the column space remain as residual. Adding a kernel component to x leaves the fit unchanged but increases its norm, so the minimum-norm solution sets that component to zero.

For A=diag(1,0),b=(2,3), the pseudoinverse is diag(1,0), x*=(2,0), and b−Ax*=(0,3). Every (2,t) gives the same residual; t=0 minimizes the norm. This choice does not recover an unobserved component.

The exact pseudoinverse satisfies AA⁺A=A, A⁺AA⁺=A⁺, with AA⁺ and A⁺A symmetric. A numerical pseudoinverse that treats small singular values as zero must specify its cutoff. That decision differs from exact rank and can regularize unstable directions.

## Read and reproduce the experiment

Blue shows the mapped unit circle; orange shows its rank-one approximation on the same output axes. Axis ranges adapt to the entries. Compare the numeric error rather than the visual size of ellipses between presets. The reconstruction readout checks the full SVD separately from truncation.

Set $A=\operatorname{diag}(-3,1)$ and explain why singular values remain 3 and 1. Then set the two columns equal and inspect rank and reconstruction. Finally, choose a nonsymmetric matrix to see why using eigenvectors of $A$ as if they were both singular bases is generally wrong.

## AI connections and research checks

Low-rank layers, matrix compression and local Jacobian analysis all use these ideas. For centered data, the eigenvalues of a denominator-$n$ covariance are squared singular values divided by $n$. Centering, denominator and matrix orientation must be specified before connecting PCA to SVD.

A research comparison should vary rank, measure actual task error and record storage and inference cost. Compare against an uncompressed model and a cost-matched baseline. A theoretical best approximation in Frobenius norm does not prove a new compression method is better on unseen data.

## Exercises and source

1. For diag(4,2), calculate rank-one spectral and Frobenius errors. **Answer:** both are 2.
2. Why does a repeated singular value make individual singular vectors nonunique? Explain using a circle.
3. At rank deficiency, identify what data cannot determine before selecting a pseudoinverse.
4. Test a nonsymmetric matrix and check full reconstruction independently of the rank-one residual.

For the decomposition and its linear-algebra background, see [MIT 18.06, Lecture 29: Singular value decomposition](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/). The calculations above are this book’s worked examples.
