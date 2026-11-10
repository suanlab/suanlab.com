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

## Exercise and research connection

Set b=0. Then B=0, K=I and the bracket is zero for all permitted a. Verify the matrix and plot. Next compare h=0.1 and 0.05 for nonzero a,b and explain the second-order group effect versus first-order scaled error.

Lie-algebra updates help keep learned rotations on a group through exponentiation. Noncommutative generators also explain why update order matters. A finite difference experiment illustrates the bracket identity, while the series expansion and algebraic closure supply the reasoning beyond a particular numerical example.

Further reading: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
