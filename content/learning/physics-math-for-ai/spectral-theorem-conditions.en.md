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

## Read the plot and numerical results

Blue shows the image of a unit circle; orange is the reference. Equal pixel unit lengths preserve shape. Dots mark Ae1 and e1; the zero matrix therefore has a visible image point at the origin. Use the commutator norm, not shape alone, to assess normality. Undefined real orthogonal reconstruction appears as —. The b control is unused outside symmetric mode; rotation uses only c.

## Infinite-dimensional hypotheses matter

A compact self-adjoint Hilbert-space operator has an orthonormal eigenbasis including its kernel. Nonzero eigenvalues have finite multiplicity and can accumulate only at zero. General bounded self-adjoint operators need not have an eigenbasis.

Multiplication Mf(x)=x f(x) on L2[0,1] is bounded self-adjoint with spectrum [0,1]. Its eigenvalue equation forces f to vanish almost everywhere, so there are no nonzero eigenfunctions. The general theorem uses a projection-valued measure. For unbounded operators, formal symmetry does not establish self-adjointness without comparing operator and adjoint domains.

## Verify a multiplication operator's spectrum directly

Consider Mf(x)=x f(x) on complex L²[0,1]. For z outside [0,1], (M−zI)⁻¹f=f/(x−z) is bounded, with norm 1/dist(z,[0,1]). Such z belong to the resolvent set.

For λ∈[0,1], take a short interval I near λ of length ℓ and normalize u=1_I/√ℓ. Then ‖u‖₂=1 while ‖(M−λI)u‖₂≤supx∈I|x−λ| tends to zero as I shrinks. A bounded inverse would imply 1≤‖(M−λI)⁻¹‖‖(M−λI)u‖, a contradiction. The spectrum is therefore exactly [0,1].

Yet an eigenfunction must vanish outside {λ}. A singleton has measure zero, so the only L² eigenfunction is zero. Approximate eigenvectors do not establish a genuine nonzero eigenvector.

## Unit-norm functions concentrating near one point

For λ=1/2 and 0<h<1/2, let u_h=1_[1/2−h,1/2+h]/√(2h). Then

$$
\|u_h\|_2=1,\qquad
\|(M-\tfrac12 I)u_h\|_2=\frac h{\sqrt3},\qquad
\int_0^1u_h=\sqrt{2h}.
$$

For any fixed v∈L², |⟨u_h,v⟩|≤(∫Ih|v|²)¹ᐟ²→0, so u_h converges weakly to zero. Its norm remains one, preventing strong convergence. This normalization does not retain the unit mass of a Dirac approximation: it preserves L² norm while integral mass tends to zero.

![Three normalized interval functions become narrower and taller while retaining L2 norm one; their multiplication-operator eigen-equation residuals shrink and their integral masses tend to zero.](/assets/learning/functional-continuous-spectrum-en.svg)

| h | Residual norm | Integral mass | Height |
|---|---|---|---|
| 1/4 | .144338 | .707107 | 1.414214 |
| 1/16 | .036084 | .353553 | 2.828427 |
| 1/64 | .009021 | .176777 | 5.656854 |

The code uses exact-fraction interval polynomial integrals. This is an infinite-dimensional example separate from the screen's 2×2 matrices. It explains why the compact self-adjoint eigenbasis theorem differs from the projection-valued-measure theorem for general self-adjoint operators. Formally inserting a Dirac into an eigen-equation would not make it an ordinary L² eigenfunction.

**Exercise:** What if some u_h subsequence converged strongly? **Explanation:** Its limit would have norm one and solve (M−1/2)u=0, contradicting the absence of a nonzero L² eigenfunction.

```python
from fractions import Fraction as F
from math import sqrt

lam=F(1,2)
for h in [F(1,4),F(1,16),F(1,64)]:
    lo,hi=lam-h,lam+h
    squared_height=F(1,2*h)
    norm_squared=(hi-lo)*squared_height
    residual_squared=squared_height*((hi-lam)**3-(lo-lam)**3)/3
    assert norm_squared==1
    assert residual_squared==h*h/3
    pairing_with_one=sqrt(float(2*h))
    print('h, residual norm, integral u_h, peak height:',
          str(h),round(sqrt(float(residual_squared)),6),
          round(pairing_with_one,6),round(sqrt(float(squared_height)),6))
```


## Exercises with explanations

1. Do real eigenvalues imply orthogonal diagonalization? **No:** the Jordan example is a counterexample.
2. Does lack of real diagonalization imply nonnormality? **No:** the rotation generator is normal.
3. Do finite discretization eigenvalues create genuine eigenfunctions of a continuous-spectrum operator? **No.** Distinguish approximation from the original spectrum.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
