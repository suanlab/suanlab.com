## Structure under an exchange of axes

A real matrix is symmetric when exchanging its two indices preserves every entry. It is skew-symmetric when the exchange reverses the sign. This is a relation between coordinates, not a condition on magnitudes. A skew-symmetric real matrix has a zero diagonal because each diagonal entry must equal its own negative.

$$
S^T=S,\qquad K^T=-K.
$$

We begin with real 2×2 matrices. For complex inner-product geometry, Hermitian symmetry uses conjugate transpose instead. Do not silently substitute ordinary transpose for that operation.

## Split any real matrix

Adding the transpose keeps the exchange-invariant part. Subtracting it keeps the sign-reversing part. Dividing by two gives a decomposition that exactly reconstructs the input.

$$
S=\frac{A+A^T}{2},\qquad K=\frac{A-A^T}{2},\qquad A=S+K.
$$

The default A has rows (1,2) and (0,3). Its symmetric part has rows (1,1) and (1,3); its skew part has rows (0,1) and (−1,0). Follow each coordinate in the table and check that S+K equals A. A symmetric input gives K=0, while a skew-symmetric input gives S=0.

## Orthogonality and quadratic forms

The Frobenius inner product sums coordinatewise products. Opposite off-diagonal terms cancel between S and K, giving orthogonality and a Pythagorean norm identity.

$$
\langle S,K\rangle_F=0,\qquad \|A\|_F^2=\|S\|_F^2+\|K\|_F^2.
$$

For a real vector v, the scalar vᵀKv equals its transpose, which is its negative. It must therefore vanish. The quadratic form vᵀAv depends only on S. The lab checks this with v=(1,2). This does not make K irrelevant to every operation: skew-symmetric matrices describe infinitesimal rotations, for example.

## Extend the idea carefully

For an order-d tensor, averaging over all axis permutations gives full symmetrization. Including each permutation’s sign gives full alternation. Over real numbers division by d! is valid. Unlike the order-two decomposition, adding only these two projections does not generally recover every higher-order tensor. Other permutation symmetry types can occur.

The controls and evaluated decomposition here remain order two. They are not an implementation of general high-order symmetrization. In AI, this foundation helps with Hessian symmetry under suitable differentiability assumptions, geometric models and constructions that preserve invariants.

## Exercises with answers

1. What is K for a symmetric input? **The zero matrix.**
2. Does K contribute to a real quadratic form? **No: vᵀKv is always zero.**
3. Do full symmetry and full alternation exhaust higher-order tensors? **Not in general.**

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
