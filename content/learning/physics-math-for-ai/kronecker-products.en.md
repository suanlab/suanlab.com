## Build a larger map from blocks

If A is m×n and B is p×q, their Kronecker product has shape mp×nq. Replace each entry of A by that scalar times the entire matrix B:

$$
A\otimes B=\begin{pmatrix}a_{11}B&\cdots&a_{1n}B\\\vdots&\ddots&\vdots\\a_{m1}B&\cdots&a_{mn}B\end{pmatrix}.
$$

With zero-based indices, (A⊗B)₍ᵢₚ₊ₖ,ⱼq₊ₗ₎=AᵢⱼBₖₗ. Separate block indices from within-block indices when checking an implementation. This construction differs from ordinary matrix multiplication AB.

## Fix column-major vectorization

Here vec stacks columns:

$$
X=\begin{pmatrix}1&2\\3&4\end{pmatrix},\qquad
\operatorname{vec}(X)=(1,3,2,4)^T.
$$

Stacking rows gives a different convention. For a NumPy reshape, explicitly use order='F' to implement this column-major convention. Mathematical index order and framework memory layout are related but distinct; inspect actual arrays rather than assuming a default.

## Two-sided matrix action becomes a vector action

For Y=AXBᵀ, component expansion gives Yᵢⱼ=ΣₖₗAᵢₖXₖₗBⱼₗ. Placing this at column-major position i+mj yields

$$
\operatorname{vec}(AXB^T)=(B\otimes A)\operatorname{vec}(X).
$$

The right-side factor B comes first because column index is the outer vectorization index. Derive the rule rather than guessing. A⊗B and B⊗A generally differ.

## Work the reference example

The lab uses A=[[1,s],[0,1]], B=diag(t,1), and the fixed X above. At s=1,t=2,

$$
AXB^T=\begin{pmatrix}8&6\\6&4\end{pmatrix},\qquad
\operatorname{vec}(AXB^T)=(8,6,6,4)^T.
$$

The reversed factor order A⊗B instead produces (6,7,4,4). The table deliberately displays A⊗B, while the identity check uses B⊗A. Keep these labels explicit.

Direct two-by-two multiplication and the four-by-four Kronecker action are computed separately, then compared by a residual norm. A zero residual checks the specified example; copying one result to both sides would not be independent validation.

## Useful identities and assumptions

For compatible dimensions,

$$
(A\otimes B)(C\otimes D)=(AC)\otimes(BD).
$$

Invertible factors give inverse A⁻¹⊗B⁻¹. In general rank(A⊗B)=rank(A)rank(B). For square factors of sizes m and p, det(A⊗B)=det(A)ᵖdet(B)ᵐ. In this lab A is always invertible; when t=0, B has rank one and the product has rank two.

These identities do not establish commutativity. Swapping factors can be related by appropriate permutation matrices without producing the same entry array in a fixed ordering.

## Read the plot and connect to AI

The horizontal axis contains component indices zero through three, not physical space or time. Lines connect components for visual comparison; intermediate indices have no physical value. Check ordering in the numeric table.

Separable filters, matrix-valued linear operations and structured covariance or curvature approximations use Kronecker structure. Applying AXBᵀ can avoid explicitly materializing a large Kronecker matrix. Compute actual memory and operation costs from dimensions and layout. A structured curvature approximation is not automatically the exact curvature.

## Exercises and answers

1. A 2×3 factor and a 4×5 factor give **an 8×15 product**.
2. At s=0,t=1, the action is **the four-dimensional identity**.
3. At t=0 the product is **singular with rank two**.

Further reading: [MIT Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/). The vectorization identity is independently derived by component expansion above.
