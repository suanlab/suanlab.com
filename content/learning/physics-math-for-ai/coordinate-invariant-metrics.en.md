## Coordinates label vectors; they do not define their length

A physical vector can have many component lists depending on the chosen basis. In orthonormal Cartesian coordinates, squared length is the sum of squared components. In scaled or sheared coordinates that rule needs the metric matrix. Confusing component changes with changes of the vector is a common source of incorrect geometric algorithms.

Use an invertible linear coordinate map x=A u with determinant one:

$$
A=\begin{pmatrix}s&k\\0&1/s\end{pmatrix},\qquad
v^u=A^{-1}v^x,\qquad g_u=A^TA
=\begin{pmatrix}s^2&sk\\sk&k^2+s^{-2}\end{pmatrix}.
$$

Here s>0 and k is a shear. The diagram always uses physical x axes. Blue segments are the two columns of A, representing the u-coordinate basis in physical space. Orange is the physical vector vˣ=(a,b). Changing A changes the basis picture, while fixed a,b keep the actual vector fixed.

## Work a nonorthonormal example

At s=2,k=1 and vˣ=(1,1), inverse coordinates give vᵘ=(−0.5,2), while g=[[4,2],[2,1.25]]. Naively summing the new component squares gives 4.25. The correct calculation is

$$
(v^u)^Tg_uv^u=(v^x)^Tv^x=2.
$$

Compute g vᵘ=(2,1.5), then dot with (−0.5,2): −1+3=2. The cancellation is geometric consistency, not a fitted numerical coincidence. The metric stays positive definite because wᵀg w=||Aw||²>0 for every nonzero w. Its determinant is one in this family, but determinant one does not make the basis orthonormal.

## Differentials and gradients have different component rules

Let f(x)=x₁+2x₂. Its differential has physical covector components αₓ=(1,2). The chain rule gives αᵤ=Aᵀαₓ, while the gradient vector requires metric raising:

$$
\alpha_u=A^T\alpha_x,\qquad
(\operatorname{grad} f)^u=g_u^{-1}\alpha_u
=A^{-1}(1,2)^T.
$$

In the example, αᵤ=(2,2) but gradient components are (−1.5,4). Both represent related objects with different tensor types. Evaluating the covector on the vector gives αᵤ·vᵘ=3, equal to a+2b in physical coordinates. Treating αᵤ directly as a vector would use the wrong metric.

## Exercise and explained answer

Set s=1,k=1 and vˣ=(1,0). Then vᵘ=(1,0), αᵤ=(1,3), and gradient components are (−1,2). Multiply the latter by A to recover physical gradient (1,2). The pairing is one and squared vector norm is one. Repeat after changing only the shear: physical scalar quantities stay fixed even if component lists change.

## Connection to invariant learning

Geometric features, latent-space distances and optimization updates should declare their tensor type and transformation rule. For nonlinear charts replace A by the local Jacobian and allow the metric to vary with position. This finite constant-Jacobian example verifies algebraic consistency; it does not establish that an arbitrary learned representation preserves a metric or topology. Coordinate independence is a property to derive and test, rather than a consequence of using geometric terminology.

Further reading: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
