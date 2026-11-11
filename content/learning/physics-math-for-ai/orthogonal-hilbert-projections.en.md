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

## Why a minimizing sequence yields an actual projection

Let V be a closed linear subspace and d=infv∈V‖f−v‖. Choose v_n with ‖f−v_n‖→d. Its midpoint remains in V, so the parallelogram law gives

$$
\|v_n-v_m\|^2
=2\|f-v_n\|^2+2\|f-v_m\|^2
-4\left\|f-\frac{v_n+v_m}{2}\right\|^2
\le2\|f-v_n\|^2+2\|f-v_m\|^2-4d^2\to0.
$$

Completeness of H and closedness of V put its limit v inside V, attaining distance d. For any w∈V, ‖f−v−tw‖² is minimized at t=0. Its linear coefficient vanishes, giving ⟨f−v,w⟩=0. Orthogonality then gives Pythagoras and uniqueness. Agreement at a selected point was not an existence or optimality argument.

## Changing basis preserves the optimal function

Represent the same V=span{1,x} using the nonorthogonal basis φ₀=1,φ₁=1+x. Coefficients solve Gc=b, where Gᵢⱼ=⟨φⱼ,φᵢ⟩ and bᵢ=⟨f,φᵢ⟩. For f=x²+a x,

$$
G=\begin{pmatrix}2&2\\2&8/3\end{pmatrix},\quad
b=\begin{pmatrix}2/3\\2/3+2a/3\end{pmatrix},\quad
c=\begin{pmatrix}1/3-a\\a\end{pmatrix}.
$$

Thus c₀+c₁(1+x)=1/3+a x, the same Legendre projection. Using b directly as coefficients instead ignores the Gram matrix's correlations and scale.

| Method, a=1 | Coefficients | Squared L² residual |
|---|---|---|
| Orthogonal {1,x} | (1/3,1) | 8/45 |
| Nonorthogonal {1,1+x}, Gc=b | (−2/3,1) | 8/45 |
| Nonorthogonal, use b directly | (2/3,4/3) | 784/135 |

Exact-fraction polynomial integration checks the Gram matrix, residual orthogonality and errors. An invertible basis change preserves the best function but can alter numerical conditioning. Inaccurate linear-system solves are another error source in high-order approximation.

**Exercise:** Must a minimizing sequence's limit remain in a dense but nonclosed V? **Explanation:** No. Completeness gives a limit in H; without closedness it may lie outside V.

```python
from fractions import Fraction as F

def inner(p,q):
    return sum(a*b*F(1-(-1)**(i+j+1),i+j+1)
               for i,a in enumerate(p) for j,b in enumerate(q))

basis=[[F(1),F(0),F(0)],[F(1),F(1),F(0)]]
f=[F(0),F(1),F(1)]  # x^2+x
G=[[inner(p,q) for q in basis] for p in basis]
rhs=[inner(f,p) for p in basis]
det=G[0][0]*G[1][1]-G[0][1]*G[1][0]
coeff=[(rhs[0]*G[1][1]-rhs[1]*G[0][1])/det,
       (G[0][0]*rhs[1]-G[1][0]*rhs[0])/det]
assert coeff==[F(-2,3),F(1)]
def residual(c):
    return [f[j]-sum(c[i]*basis[i][j] for i in range(2)) for j in range(3)]
r=residual(coeff)
assert all(inner(r,p)==0 for p in basis)
assert inner(r,r)==F(8,45)
assert inner(residual(rhs),residual(rhs))==F(784,135)
print('Gram matrix:',G)
print('correct coefficients, error squared:',coeff,inner(r,r))
print('raw-inner-product coefficients, error squared:',rhs,inner(residual(rhs),residual(rhs)))
```


## Exercises with explanations

1. Constant-projection error at a=0? **8/45**, because no linear component is omitted.
2. Is the degree-one residual orthogonal to P₂? **No:** its inner product is 4/15.
3. Can raw inner products be coefficients in an unnormalized basis? **No.** Include squared basis norms or the Gram matrix.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
