## Best approximation in a function space

For a closed linear subspace V of a real Hilbert space H, every f has a unique decomposition f=v+r with v∈V and r orthogonal to V. The projection v=Pᵥf is its closest element. Completeness and closedness matter for guaranteed existence.

$$
\|f-w\|^2=\|f-P_Vf\|^2+\|P_Vf-w\|^2,\quad w\in V.
$$

The nonnegative second term establishes optimality and uniqueness. This generalizes finite least squares after fixing the inner product.

## Compute with Legendre polynomials

On L2[−1,1], P₀=1, P₁=x and P₂=(3x²−1)/2 are orthogonal, with squared norms 2, 2/3 and 2/5. They are not normalized; keep the denominator when computing coefficients.

$$
c_k=\frac{\langle f,P_k\rangle}{\|P_k\|^2},\qquad f=x^2+a x=\frac13P_0+aP_1+\frac23P_2.
$$

Projection through degree one gives 1/3+a x, leaving residual x²−1/3. At a=1, squared target, projection and residual norms are 16/15, 8/9 and 8/45. Keeping only constants also omits the linear component, giving error 38/45.

## Inspect orthogonality rather than one point

Blue is the target and orange its projection. Residual inner products vanish for retained basis functions, but may not vanish for omitted ones. Accidental agreement at one query point does not imply zero L2 error.

Keeping degree two includes the target exactly. Coefficient truncation is a projection here because the basis is orthogonal; a general basis requires its Gram matrix.

## Closedness and research use

All polynomials form a dense but nonclosed subspace of L2. A nonpolynomial target can have distance infimum zero without attaining it among polynomials. Our finite-degree spaces are closed because they are finite dimensional.

Galerkin and reduced-order methods use related subspace ideas. Different energy inner products or nonsymmetric bilinear forms require their own stability analysis.

## Exercises with explanations

1. Constant-projection error at a=0? **8/45**, because no linear component is omitted.
2. Is the degree-one residual orthogonal to P₂? **No:** its inner product is 4/15.
3. Can raw inner products be coefficients in an unnormalized basis? **No.** Include squared basis norms or the Gram matrix.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
