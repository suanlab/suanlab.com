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

## Exercise: check the mixed-term convention

Consider u_xx+u_xy+u_yy. Here B=1/2, so D=1/4-1=-3/4 and the form is elliptic. At phi=pi/4 its directional value is 1.5. If you incorrectly enter B=1, you instead construct a rank-one form. Rotate the correct coefficients by pi/2 and verify that evaluating the transformed normal preserves the same directional value. Finally explain why multiplying the entire equation by minus one does not change elliptic type.

Further reading: [MIT introduction to PDEs](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/). The matrix, angular experiment and convention checks are derived here.
