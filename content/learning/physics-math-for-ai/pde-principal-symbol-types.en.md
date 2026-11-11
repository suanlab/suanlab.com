## Read the principal part before naming the equation type

For a constant-coefficient second-order equation in two independent variables, write its principal part as A u_xx+2B u_xy+C u_yy. Lower-order terms affect solutions but do not change the principal quadratic form. The factor two is essential: if the written mixed coefficient is b rather than 2B, the matrix entry is b/2.

The lab studies this two-variable second-order classification directly. The heat equation is parabolic in an evolutionary sense involving a first time derivative and second spatial derivatives; classifying only its spatial Laplacian is a different question. Naming independent variables and differential orders prevents these related conventions from being mixed.

## The symbol is a directional quadratic form

For a unit normal n=(cos(phi),sin(phi)), define

$$
Q=\begin{pmatrix}A&B\\B&C\end{pmatrix},\qquad
p(n)=n^TQn,
\qquad D=B^2-AC.
$$

If D is negative, Q is definite and the principal part is elliptic; both positive and negative definite matrices qualify. Positive D gives an indefinite, hyperbolic form. Zero D with nonzero Q gives a rank-one parabolic principal part. If all three coefficients vanish, there is no second-order principal part, rather than a rank-one operator.

Blue plots p(n) over normal angle. An elliptic form has one nonzero sign for every unit normal. A hyperbolic form changes sign and has characteristic normal directions where the symbol vanishes. A rank-one form touches zero in its missing direction. Near degeneracy, coefficient accuracy and floating-point arithmetic deserve attention in addition to the displayed class code.

## Rotate the coordinates and the normal together

Let R be a rotation matrix and use rotated coordinates. Then

$$
Q'=R^TQR,\qquad n'=R^Tn,\qquad
(n')^TQ'n'=n^TQn.
$$

The orange curve evaluates the rotated coefficients on the same physical normal, whose rotated angle is phi-theta. Rotating Q while leaving normal coordinates unchanged would compare different physical directions. The traces and determinants are invariant under this orthogonal change, although their floating-point recomputations can differ by rounding.

## Work through three principal parts

For u_xx+u_yy, Q is the identity and every directional value equals one. For u_xx-u_yy, the symbol is cos(2phi), changing sign and vanishing at diagonal normals. For u_xx alone, the symbol is cos²(phi), a nonnegative rank-one form. These are statements about derivative directions rather than trajectories of particles.

## Coordinate invariance and quantitative ellipticity

For an invertible linear coordinate change x=Jy, the principal matrix becomes Q′=J⁻¹QJ⁻ᵀ. The quadratic form at ξ is the original form at J⁻ᵀξ. Because this map is bijective, the maximal dimensions of positive, negative and null subspaces are unchanged: this is the inertia argument behind type invariance. Orthogonal rotation is the special case used by the live experiment. Eigenvalue magnitudes generally change under a nonorthogonal coordinate map, even though inertia does not.

For positive-definite Q, quantitative ellipticity requires a lower bound ξᵀQξ≥α‖ξ‖². If Q≥αI, then

$$
\xi^TQ'\xi\ge\alpha\|J^{-T}\xi\|^2
\ge\frac{\alpha}{\|J\|_2^2}\|\xi\|^2.
$$

Thus the coordinate transformation can worsen constants controlling PDE estimates. Consider Q=diag(1,ε), 0<ε≤1. Every member is elliptic, but the best lower constant ε tends to zero. Calling all of them elliptic hides conditioning relevant to anisotropic diffusion and neural PDE training.

| ε | Type | Best Euclidean lower constant |
|---|---|---:|
| 1 | elliptic | 1 |
| 1/10 | elliptic | 1/10 |
| 1/100 | elliptic | 1/100 |

```python
from fractions import Fraction as F
def mm(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def det(A): return A[0][0]*A[1][1]-A[0][1]*A[1][0]
Jinv = [[F(1), F(-2)], [F(0), F(1)]]
JTinv = list(map(list, zip(*Jinv)))
for eps in (F(1), F(1, 10), F(1, 100)):
    Q = [[F(1), F(0)], [F(0), eps]]
    transformed = mm(mm(Jinv, Q), JTinv)
    assert det(transformed) == det(Jinv)**2*det(Q) > 0
    for v in ((F(1), F(0)), (F(0), F(1)), (F(1), F(-1))):
        form = sum(v[i]*Q[i][j]*v[j] for i in range(2) for j in range(2))
        assert form >= eps*sum(x*x for x in v)
    print(eps, transformed)
```

The rational shear calculation checks congruence and determinant scaling exactly, rather than trusting rounded eigenvalues near degeneracy. For Q=diag(1,−1), the characteristic normals satisfy ξ₁²−ξ₂²=0; these are directions in the symbol, not state-space trajectories. Principal type alone also does not provide boundary conditions or a complete well-posedness theorem.

The live type now uses the exact determinant sign of the stored binary64 coefficients, represented as integer multiples of 2⁻¹⁰⁷⁴. This prevents underflow or rounded cancellation from changing the type. Displayed determinant and eigenvalues remain floating diagnostics; exact classification of stored numbers does not certify uncertain physical coefficients.

## Exercise: check the mixed-term convention

Consider u_xx+u_xy+u_yy. Here B=1/2, so D=1/4-1=-3/4 and the form is elliptic. At phi=pi/4 its directional value is 1.5. If you incorrectly enter B=1, you instead construct a rank-one form. Rotate the correct coefficients by pi/2 and verify that evaluating the transformed normal preserves the same directional value. Finally explain why multiplying the entire equation by minus one does not change elliptic type.

Further reading: [MIT introduction to PDEs](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/). The matrix, angular experiment and convention checks are derived here.
