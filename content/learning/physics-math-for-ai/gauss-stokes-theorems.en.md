## Boundary and interior descriptions

The divergence theorem relates outward flux through a closed boundary to the integral of divergence:

$$
\int_{\partial V}F\cdot n\,dS=\int_V\nabla\cdot F\,dV.
$$

Stokes relates boundary circulation to curl flux through an oriented surface:

$$
\oint_{\partial S}F\cdot dr=\int_S(\nabla\times F)\cdot n\,dS.
$$

Match boundary orientation to the surface normal by the right-hand rule. Smoothness and piecewise regular domain assumptions are part of the statements.

## Check both with one field

Use F=a(x,y,z)+b(−y,x,0), a ball of radius R and its equatorial disk. Divergence is 3a and curl is (0,0,2b), so

$$
\int_V\operatorname{div}F\,dV=4\pi aR^3,\qquad
\int_S\operatorname{curl}F\cdot n\,dS=2\pi bR^2.
$$

Rotation is tangent to the sphere and contributes no outward flux. The radial component is orthogonal to the equatorial tangent and contributes no circulation.

## Compute the boundary independently

Sphere flux reduces to 2πaR³∫₀πsinφ dφ. The lab uses 64-panel Simpson quadrature and compares it with the analytic volume expression. Circle circulation uses actual field–tangent dot products in a midpoint sum. The two sides are not copied from one expression.

Small discrepancies are numerical quadrature or rounding error. A finite verification does not prove the theorem. The picture is only the equatorial cross-section, not the whole sphere.

## Singularities invalidate naive application

The field

$$
F(x)=\frac{x}{\|x\|^3}
$$

has zero divergence away from zero but flux 4π through a sphere enclosing zero. There is no contradiction: the field is singular inside the ball. Apply the theorem instead on a shell excluding the origin, accounting for its inner boundary orientation.

Check interior regularity and holes in the domain, not merely smooth-looking boundary values.

## Orientation, limits and AI

Reversing only one orientation changes a sign. At R=0 there is no regular sphere or disk; the displayed zeros are degenerate limits.

Physics-informed objectives need boundary conditions, measures and orientation made explicit. Small residual at finitely many collocation points is not an exact certificate over an entire domain.

## Exercises and answers

1. At a=b=R=1 the interior results are **4π and 2π**.
2. Setting only a=0 **does not remove circulation**, which depends on b.
3. Applying the ball theorem across the origin singularity **violates its assumptions**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
