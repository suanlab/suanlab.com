## Eigenvalues do not guarantee an eigenbasis

Even finite matrices can lack enough eigenvectors. With the real Euclidean inner product, adjoint means transpose. Real symmetric matrices have real eigenvalues and an orthogonal eigenbasis. Over complex scalars, normality A*A=AA* characterizes unitary diagonalization.

$$
A=Q\Lambda Q^T,\qquad Q^TQ=I\quad\text{for real symmetric }A.
$$

## Compare three explicit matrices

The symmetric example is [[a,c],[c,b]], with eigenvalues (a+b)/2 plus or minus √(((a−b)/2)²+c²). At a=2,b=1,c=.5, they are about 2.207 and .793; orthogonal reconstruction error is at rounding scale.

$$
J=\begin{pmatrix}a&c\\0&a\end{pmatrix},\qquad R=\begin{pmatrix}0&-c\\c&0\end{pmatrix}.
$$

For c≠0, J has repeated real eigenvalue a but a one-dimensional eigenspace, so it is defective. R is normal yet has eigenvalues ±ic. It is unitarily diagonalizable over complex scalars, not orthogonally diagonalizable over real scalars. At c=0 both counterexamples degenerate.

Blue shows the image of a unit circle; orange is the reference. Equal pixel unit lengths preserve shape. Dots mark Ae1 and e1; the zero matrix therefore has a visible image point at the origin. Use the commutator norm, not shape alone, to assess normality. Undefined real orthogonal reconstruction appears as —. The b control is unused outside symmetric mode; rotation uses only c.

## Infinite-dimensional hypotheses matter

A compact self-adjoint Hilbert-space operator has an orthonormal eigenbasis including its kernel. Nonzero eigenvalues have finite multiplicity and can accumulate only at zero. General bounded self-adjoint operators need not have an eigenbasis.

Multiplication Mf(x)=x f(x) on L2[0,1] is bounded self-adjoint with spectrum [0,1]. Its eigenvalue equation forces f to vanish almost everywhere, so there are no nonzero eigenfunctions. The general theorem uses a projection-valued measure. For unbounded operators, formal symmetry does not establish self-adjointness without comparing operator and adjoint domains.

## Exercises with explanations

1. Do real eigenvalues imply orthogonal diagonalization? **No:** the Jordan example is a counterexample.
2. Does lack of real diagonalization imply nonnormality? **No:** the rotation generator is normal.
3. Do finite discretization eigenvalues create genuine eigenfunctions of a continuous-spectrum operator? **No.** Distinguish approximation from the original spectrum.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
