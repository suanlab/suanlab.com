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

Let $0\le k\le\min(m,n)$ and set singular values beyond the last index to zero. For a rank at most $k$ approximation, the truncated SVD minimizes both spectral and Frobenius error. At $k=\min(m,n)$, retaining the full matrix gives zero error. The errors are

$$
\lVert A-A_k\rVert_2=\sigma_{k+1},\qquad
\lVert A-A_k\rVert_F^2=\sum_{i>k}\sigma_i^2.
$$

For $k<\min(m,n)$, a useful lower-bound argument for the spectral result is that a rank-$k$ map must annihilate some nonzero vector in the span of the first $k+1$ right singular vectors. On that subspace, $A$ stretches by at least $\sigma_{k+1}$. Truncation attains the bound. The Frobenius result follows by orthogonal decomposition into singular components; proving the full minimization statement requires the corresponding projection argument, beyond just checking one diagonal example.

For the Frobenius lower bound, let $P$ be the orthogonal projector onto the column space of any candidate $B$ with rank at most $k$. Since $PB=B$, the two terms in

$$
A-B=(I-P)A+(PA-B)
$$

have orthogonal columns term by term. Consequently,

$$
\|A-B\|_F^2\ge\|(I-P)A\|_F^2
=\|A\|_F^2-\operatorname{tr}(PAA^T).
$$

Write $AA^T=\sum_i\sigma_i^2u_iu_i^T$, including zero eigenvalues to complete the output basis. Then $\operatorname{tr}(PAA^T)=\sum_i\sigma_i^2\|Pu_i\|^2$. The weights lie in $[0,1]$ and sum to $\operatorname{rank}(P)\le k$. Their weighted sum is at most the sum of the largest $k$ squared singular values. This follows by assigning the available weight first to the largest coefficients. Thus every candidate has squared error at least $\sum_{i>k}\sigma_i^2$, and $A_k$ attains it. This proves the Frobenius minimization for finite real matrices and an unweighted Euclidean norm.

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

## Derive a loss for the layer's actual inputs

Let $A,B\in\mathbb R^{m\times n}$ be the original and compressed maps. Suppose $x$ has finite second moment $M=\mathbb E[xx^T]$. The squared output error is

$$
\mathcal L(B)=\mathbb E\|(A-B)x\|^2
=\operatorname{tr}((A-B)M(A-B)^T)
=\|(A-B)M^{1/2}\|_F^2.
$$

Expand the squared norm into coordinates and take expectations of $x_ix_j$ to obtain the trace identity. The last equality follows from the symmetric square root of $M$. This is the uncentered second moment. For mean $\mu$ and covariance $C$, $M=C+\mu\mu^T$. Replacing it with covariance requires centering or an explicit treatment of the mean and bias.

When $M$ is positive definite, right multiplication by $M^{1/2}$ preserves rank and is invertible. Put $D=BM^{1/2}$. Minimizing $\mathcal L$ over rank-at-most-$k$ matrices becomes the Frobenius problem for $AM^{1/2}$. If $D_k$ is its truncated SVD, a minimizer is

$$
B_{\mathrm{task}}=D_kM^{-1/2}.
$$

This change of variables proves the global optimum for this particular linear, squared-error task. A singular moment matrix needs a separate argument on its observed support. Classification loss and subsequent nonlinear layers introduce different objectives.

## Calculate a reversal of the preferred direction

Keep $A=\operatorname{diag}(3,1)$. Give each of the four inputs $(\pm0.1,\pm10)$ probability one quarter. Their mean is zero and $M=\operatorname{diag}(0.01,100)$. The second input coordinate is much larger on this population, so discarding it can dominate output error.

| Rank-one map B | Squared matrix error | Expected squared output error |
|---|---|---|
| diag(3,0), ordinary SVD | 1 | 100 |
| diag(0,1), task-weighted SVD | 9 | 0.09 |

For the first map, residual output is $(0,x_2)$ and its expected squared length is 100. For the second it is $(3x_1,0)$, giving $9\times0.01=0.09$. Also $AM^{1/2}=\operatorname{diag}(0.3,10)$, whose leading direction is the second axis. The preceding minimization proof therefore certifies the second row among all rank-one maps for this distribution.

If $M=I$, task error equals squared Frobenius error and ordinary SVD is optimal again. This gives a useful control condition. The conclusion depends on which input distribution and output metric the experiment represents.

## Reproduce the comparison from four inputs

The following NumPy example compares direct predictions against the trace formula. It uses all four population inputs with equal weights. Its second moment is known exactly from this finite model. In a data-driven application, estimate calibration statistics on allowed training or calibration data and evaluate on separate data.

```python
from itertools import product
import numpy as np

matrix = np.diag([3., 1.])
inputs = np.array(list(product([-.1, .1], [-10., 10.])))
second_moment = inputs.T @ inputs / len(inputs)
root = np.diag([.1, 10.])
assert np.allclose(root @ root.T, second_moment)

def truncate_rank_one(value):
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    return singular[0] * np.outer(left[:, 0], right[0])

ordinary = truncate_rank_one(matrix)
weighted = truncate_rank_one(matrix @ root) @ np.linalg.inv(root)
for name, approximation, expected in [
    ("ordinary", ordinary, [1., 100.]),
    ("weighted", weighted, [9., .09]),
]:
    squared_matrix_error = np.linalg.norm(matrix - approximation, "fro")**2
    output_errors = inputs @ (matrix - approximation).T
    task_error = np.mean(np.sum(output_errors**2, axis=1))
    trace_error = np.trace(
        (matrix - approximation) @ second_moment @ (matrix - approximation).T
    )
    assert np.isclose(task_error, trace_error)
    assert np.allclose([squared_matrix_error, task_error], expected)
    assert np.linalg.matrix_rank(approximation) == 1
    print(name, round(squared_matrix_error, 2), round(task_error, 2))
```

The output rows are `ordinary 1.0 100.0` and `weighted 9.0 0.09`. Keep squared errors distinct from their square roots when comparing these numbers with the lab's residual-norm readout. The lab above transforms a unit circle and illustrates the unweighted SVD. The table and code introduce the specified nonuniform input population.

## Connect a rank constraint to a trainable update

LoRA freezes a base weight and trains factors of its update. With $W_0\in\mathbb R^{d\times n}$, write $\Delta W=LR$ for $L\in\mathbb R^{d\times r}$ and $R\in\mathbb R^{r\times n}$. The paper's §4.1 defines this parametrization and forward operation. [Hu et al., LoRA, arXiv v2, 16 October 2021, §4.1](https://arxiv.org/pdf/2106.09685v2)

Every column of $LR$ lies in the span of the $r$ columns of $L$, proving $\operatorname{rank}(\Delta W)\le r$. The sum $W_0+LR$ can have full rank. For example, $I_2+\operatorname{diag}(1,0)=\operatorname{diag}(2,1)$ has rank two although its update has rank one. The factors contain $r(d+n)$ scalar entries, compared with $dn$ in a dense update. Savings require the corresponding dimension inequality.

A follow-up can compare an SVD approximation to a fixed target update with factors optimized for a declared task loss. Record the base weight, calibration distribution, rank, parameter count, optimization budget and held-out evaluation. The four-input calculation here establishes an objective-dependent optimum in a synthetic linear model. Reproducing the paper's language-model training experiments remains a separate research exercise.

## Exercises and source

1. For diag(4,2), calculate rank-one spectral and Frobenius errors. **Answer:** both are 2.
2. Why does a repeated singular value make individual singular vectors nonunique? Explain using a circle.
3. At rank deficiency, identify what data cannot determine before selecting a pseudoinverse.
4. Test a nonsymmetric matrix and check full reconstruction independently of the rank-one residual.

5. If $M=I$, why does task loss equal squared Frobenius error? **The moment weighting is the identity, so the trace is the sum of squared matrix entries.**
6. Can a rank-one update yield a rank-two adapted weight? **Yes**, as $I_2+\operatorname{diag}(1,0)$ shows.

For the decomposition and its linear-algebra background, see [MIT 18.06, Lecture 29: Singular value decomposition](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/). The calculations above are this book’s worked examples.
