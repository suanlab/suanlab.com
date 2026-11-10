## Integrate a rule that acts on velocities

A differential one-form α=P dx+Q dy is a covector field. Along a parameterized curve γ(t)=(x(t),y(t)), it evaluates the velocity and then integrates:

$$
\int_\gamma\alpha=\int\big(P(\gamma(t))\dot x(t)+Q(\gamma(t))\dot y(t)\big)\,dt,
\qquad d\alpha=(\partial_xQ-\partial_yP)dx\wedge dy.
$$

The wedge product records orientation: dx∧dy=−dy∧dx and dx∧dx=0. The exterior derivative is coordinate compatible and satisfies d²=0. Therefore an exact form α=df is closed, dα=0. The converse depends on the domain's topology.

## Compare two smooth forms on a disk

First take α=x dx+y dy=d((x²+y²)/2). Its integral over any closed smooth loop is zero by the fundamental theorem for line integrals. A translated circle gives the same result; exactness is not a symmetry accident of a circle centered at the origin.

Next take α=(−y dx+x dy)/2. Its exterior derivative is dx∧dy. For a counterclockwise radius-R circle bounding an oriented disk D,

$$
\oint_{\partial D}\alpha=\int_Dd\alpha=\pi R^2.
$$

Reversing orientation gives −πR². The exterior derivative coefficient remains one; the oriented integral changes because the disk and its induced boundary orientation reverse. Orange short segments show field directions normalized to a common drawing length, not actual coefficient magnitudes or the path's tangent vectors.

## A missing point changes the conclusion

The angular form α=(−y dx+x dy)/(x²+y²) is smooth only on the punctured plane. Direct differentiation gives dα=0 there. Nevertheless, a counterclockwise circle surrounding the origin has integral 2π. Parameterizing a centered circle cancels R² and leaves integral ∫₀²π1 dt.

This does not contradict Stokes. The form is undefined at the origin, so it is not smooth on the spanning disk. If the disk avoids the origin, both integrals are zero. If the boundary itself passes through the origin, the ordinary line integral is not assigned a value by this lab. No principal-value regularization is silently introduced.

## Exercise and explained answer

For the angular form choose R=0.5 and center (1,0). The disk avoids the origin, so the integral is zero and disk Stokes applies. Move the center to (0,0): boundary integral becomes 2π and disk Stokes no longer applies. At center (0.5,0), the boundary hits the excluded point and its ordinary integral is undefined. Verify all three states.

## Connection to scientific AI

Forms organize fluxes, conservation laws and coordinate-compatible differential operators. When a learned field appears curl free, a global potential still needs domain hypotheses. Holes, excluded singularities and boundary conditions can invalidate that inference even when all local derivatives look correct. A finite directional plot is an aid to the calculation, not a proof that a sampled field satisfies a global theorem.

Further reading: [Lorenzo Sadun, Lecture Notes on Differential Forms](https://arxiv.org/abs/1604.07862).
