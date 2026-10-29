## Area magnitude and orientation

For a two-by-two matrix, the determinant is the oriented area of the parallelogram formed by its columns. Relative to a fixed ordered basis, a positive sign preserves orientation and a negative sign reverses it. Unsigned area uses |det A|.

$$
A=\begin{pmatrix}a&b\\c&d\end{pmatrix},\qquad\det A=ad-bc,\qquad\operatorname{tr}A=a+d.
$$

The blue boundary joins zero, the first column, the sum of columns, and the second column. Orange is the unit square. Equal axis units preserve area geometry; adding drawn edge lengths does not compute the determinant.

## Rotation, reflection and collapse

A quarter turn [[0,−1],[1,0]] has determinant one and trace zero. It preserves oriented area without being the identity. Reflection diag(1,−1) has determinant minus one and area magnitude one.

The matrix [[1,2],[2,4]] has determinant zero and collapses area onto a line. It does not send every vector to zero. Singular means noninvertible, not necessarily rank zero.

$$
\det(AB)=\det A\det B,\qquad\det(A^{-1})=(\det A)^{-1}.
$$

The inverse formula requires invertibility. In n dimensions, the absolute determinant scales volume, and composition multiplies scaling factors.

## Trace and basis invariance

Whenever the rectangular products AB and BA are defined, component expansion gives

$$
\operatorname{tr}(AB)=\sum_{i,j}A_{ij}B_{ji}=\operatorname{tr}(BA).
$$

Consequently, for invertible P, tr(P⁻¹AP)=tr A. Likewise det(P⁻¹AP)=det A. These are similarity invariants of a linear map, independent of the chosen basis.

They do not determine the whole matrix. Both I and [[1,1],[0,1]] have trace two and determinant one, but the shear is not I. Equal invariants need not recover eigenvectors or all structural information.

## Characteristic polynomial and complex roots

For two-by-two matrices,

$$
\det(\lambda I-A)=\lambda^2-(\operatorname{tr}A)\lambda+\det A.
$$

Eigenvalues sum to the trace and multiply to the determinant. A real matrix need not have real eigenvalues: the quarter turn gives λ²+1=0, with roots ±i. Its real-eigenvalue readout is —; the separate pair (0,1) records common real part and imaginary magnitude. This does not mean eigenvalues fail to exist over the complex numbers.

## First-order change near identity

The following two-dimensional identity is exact:

$$
\det(I+tA)=1+t\operatorname{tr}A+t^2\det A.
$$

The derivative of area scaling at t=0 is therefore the trace. Trace is not itself the finite-t volume factor. This gives intuition for divergence in continuous dynamics, while the derivation here is only a static matrix identity.

## AI connections and exercises

Density changes under invertible maps involve log|det J|. A negative determinant changes orientation, not probability sign. Large computations often need stable decompositions and log-determinants rather than direct products that underflow or overflow. Singular matrices do not have a finite log absolute determinant.

1. Is a zero-determinant map invertible? **No.**
2. Does trace zero mean the zero matrix? **No; a quarter turn is a counterexample.**
3. For diag(2,3), determinant and trace are **six and five**.

Further reading: [MIT Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/). The small-matrix examples and identities are independently expanded.
