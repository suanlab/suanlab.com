## A continuous group has a tangent algebra

A Lie group is both a smooth manifold and a group, with smooth multiplication and inversion. Its Lie algebra is the tangent space at identity together with a bracket. In the matrix group SO(3), differentiating a curve R(t) through identity in R(t)ᵀR(t)=I gives Ωᵀ+Ω=0. Thus so(3) consists of skew-symmetric three-by-three matrices.

A vector ω corresponds to the cross-product operator hat(ω), defined by hat(ω)v=ω×v. This identifies the three independent skew entries with three angular-generator coordinates; it does not identify finite rotations with ordinary vector addition globally.

$$
\widehat\omega=\begin{pmatrix}0&-\omega_z&\omega_y\\
\omega_z&0&-\omega_x\\-\omega_y&\omega_x&0\end{pmatrix},\qquad
[\widehat a,\widehat b]=\widehat{a\times b}.
$$

## The bracket measures infinitesimal noncommutativity

For matrices, [A,B]=AB−BA. It is again skew when A and B are skew, so the algebra closes under the bracket. In this example A=a hat(ex), B=b hat(ey), and their bracket is ab hat(ez). The full matrices are calculated, not replaced with a planar rotation.

The group commutator of four small rotations satisfies

$$
K(h)=e^{hA}e^{hB}e^{-hA}e^{-hB}
=I+h^2[A,B]+O(h^3).
$$

Expand each exponential through second order to see the constant terms cancel to I, the linear terms cancel, and the remaining mixed term is AB−BA. For bounded fixed A,B, dividing K−I by h² approaches the bracket with first-order truncation error.

## Inspect the finite calculation

At a=b=1 and h=0.1, the scaled [1,0] entry is about 0.99665, compared with bracket value one. The full scaled-matrix Frobenius discrepancy is about 0.10011. At h=0.05 it is about 0.05001. A single entry can converge faster than the full matrix; do not infer every error rate from that entry alone.

The 3D drawing shows reference vectors and their actual transformation under K. The lower plot shows the chosen matrix entry versus h, while orange is the bracket reference. Apparent angles in the oblique drawing are not exact rotation angles.

## Zero and numerical precision are different cases

At h=0, (K−I)/h² is undefined even though its limit and the bracket exist. Very small positive h introduces subtraction cancellation amplified by h⁻². This implementation hides the scaled quotient below 10⁻⁶ and explicitly labels the precision limit; that threshold is a numerical display policy, not a new mathematical domain restriction. The reference curve uses h≥0.01.

## Separate the bracket, the remainder and floating-point error

Associativity of matrix multiplication implies the Jacobi identity: expanding the three nested commutators cancels every cubic monomial with its negative. Thus the skew matrices form a Lie algebra, not merely a vector space with an arbitrary noncommutative operation.

For fixed bounded matrices, multiplying the exponential series one order further gives

$$
K(h)=I+h^2[A,B]+\frac{h^3}{2}[A+B,[A,B]]+O(h^4).
$$

The quadratic term is even in h and the cubic term odd. Hence averaging K(h) and K(−h) cancels the cubic term. After subtracting identity and dividing by h², this centered quotient has O(h²) truncation error, whereas the one-sided quotient generally has O(h) error. This centered construction is a separate Python comparison; the lab retains its original one-sided quotient and explicit small-h precision policy.

| Quantity | Mathematical limiting value | Truncation order |
|---|---|---|
| (K(h)−I)/h² | [A,B] | O(h) |
| (K(h)+K(−h)−2I)/(2h²) | [A,B] | O(h²) |
| Computed quotient at extremely small h | Not certified by a limit alone | Roundoff may dominate |

The code independently sums matrix exponentials rather than calling the lab's Rodrigues helper. Halve h and inspect the error ratios: roughly two versus four in the asymptotic window. Decreasing h indefinitely does not monotonically improve a floating-point difference quotient.

If either generator is zero, the group commutator is exactly identity for every h. The implementation evaluates this known identity directly, rather than turning roundoff in a rotation times its inverse into a spurious scaled defect. The h=0 and small-h display policies remain separate.

## Reproduce the calculation in Python

```python
from math import sqrt
I=[[float(i==j) for j in range(3)] for i in range(3)]
A=[[0,0,0],[0,0,-1],[0,1,0]]; B=[[0,0,1],[0,0,0],[-1,0,0]]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def exp(M):
    total=[r[:] for r in I]; term=[r[:] for r in I]
    for k in range(1,45):
        term=[[v/k for v in row] for row in mm(term,M)]
        total=[[total[i][j]+term[i][j] for j in range(3)] for i in range(3)]
    return total
def K(h):
    out=I
    for M,s in [(A,h),(B,h),(A,-h),(B,-h)]: out=mm(out,exp([[s*v for v in row] for row in M]))
    return out
AB,BA=mm(A,B),mm(B,A); C=[[AB[i][j]-BA[i][j] for j in range(3)] for i in range(3)]
for h in [.1,.05,.025,.0125]:
    plus,minus=K(h),K(-h)
    forward=sqrt(sum(((plus[i][j]-I[i][j])/h**2-C[i][j])**2 for i in range(3) for j in range(3)))
    centered=sqrt(sum(((plus[i][j]+minus[i][j]-2*I[i][j])/(2*h*h)-C[i][j])**2 for i in range(3) for j in range(3)))
    assert centered < forward
    print(h, round(forward,9), round(centered,9))
```

## Exercise and research connection

Set b=0. Then B=0, K=I and the bracket is zero for all permitted a. Verify the matrix and plot. Next compare h=0.1 and 0.05 for nonzero a,b and explain the second-order group effect versus first-order scaled error.

Lie-algebra updates help keep learned rotations on a group through exponentiation. Noncommutative generators also explain why update order matters. A finite difference experiment illustrates the bracket identity, while the series expansion and algebraic closure supply the reasoning beyond a particular numerical example.

Further reading: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
