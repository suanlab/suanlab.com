## A vector at every position

A field maps positions to vectors. It may represent velocity, force or an optimization direction, with units specified separately. Here F=(ax+by,cx+dy), displayed with arrows scaled by 0.3.

Trajectories instead solve r′(t)=F(r(t)). A finite straight arrow is not an integrated particle path; this plot performs no time integration.

## Different summaries of the Jacobian

$$
J_F=\begin{pmatrix}a&b\\c&d\end{pmatrix},\qquad
\operatorname{div}F=\partial_xF_x+\partial_yF_y=a+d.
$$

Divergence connects to net outward flux and local volume expansion. It is neither vector magnitude nor determinant. For a small flow step I+hJ, determinant=1+h tr J+O(h²), linking trace to first-order area change.

Embed the planar field in three dimensions with zero z component:

$$
(\nabla\times F)_z=\partial_xF_y-\partial_yF_x=c-b.
$$

Curl measures local circulation density, not the angle of an individual arrow.

## Compare rotation, expansion and shear

The rotational field (−y,x) has divergence zero and curl two. The radial field (x,y) has divergence two and curl zero. Shear (y,0) has divergence zero and curl minus one: shear can contain local rotation.

The decomposition

$$
J=\frac{J+J^T}{2}+\frac{J-J^T}{2}
$$

separates local deformation from rotation. In this planar linear setting, angular-velocity component is curl_z/2, not curl_z itself.

## Potentials need domain assumptions

Curl-free fields relate to potentials under appropriate smoothness and simply connected domain hypotheses. For this global linear family, b=c explicitly gives

$$
F=\nabla\left(\frac a2x^2+bxy+\frac d2y^2\right).
$$

A general field on a punctured domain cannot inherit this conclusion without checking topology. Finite arrows illustrate the field; exact Jacobian entries certify the displayed predicates.

## AI connections and exercises

Continuous normalizing flows involve velocity-field divergence in density evolution. Derivations need Jacobians and continuity-equation assumptions, not just planar visual intuition. A smooth gradient flow has a symmetric Hessian derivative, while an arbitrary learning update field need not be a gradient.

1. For (−y,x), divergence and curl are **zero and two**.
2. A potential for (x,y) is **(x²+y²)/2**.
3. Doubling only display-arrow scale **does not change the mathematical field or its curl**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
