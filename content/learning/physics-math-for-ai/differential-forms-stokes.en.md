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

## A constructive potential needs a domain hypothesis

On a star-shaped open domain containing the origin, a continuously differentiable closed one-form has a potential. Define F by integrating along the radial segment. Differentiation under the integral and the closedness identity give

$$
F(x)=\int_0^1\alpha_{tx}(x)\,dt,\qquad
\partial_jF(x)=\int_0^1\frac{d}{dt}\big[t\alpha_j(tx)\big]dt=\alpha_j(x).
$$

The segment must stay in the domain. On a punctured plane it may pass through the missing point, so this proof cannot produce a global angle potential. The integral of the angular form around a loop instead records 2π times its winding number. A local angle exists on a suitable slit domain; a single-valued global angle does not exist on the whole punctured plane.

The numerical code directly integrates the angular form along translated circles, including reversed orientation. It avoids the singular boundary case rather than giving it a principal value. Increasing the quadrature resolution tests convergence of these examples; it cannot replace checking smoothness on a proposed spanning surface.

| Center x | Radius | Orientation | Angular-form integral |
|---:|---:|---|---:|
| 0 | 1 | Counterclockwise | 2π |
| 0 | 1 | Clockwise | -2π |
| 2 | 1 | Counterclockwise | 0 |

## Reproduce the calculation in Python

```python
from math import pi, sin, cos
n = 20000
for center, sign in [(0,1),(0,-1),(2,1)]:
    total = 0.0
    for i in range(n):
        t = sign*2*pi*(i+.5)/n
        x, y = center+cos(t), sin(t)
        dx, dy = -sin(t), cos(t)
        total += (-y*dx+x*dy)/(x*x+y*y)*sign*2*pi/n
    expected = sign*2*pi if center < 1 else 0
    assert abs(total-expected) < 1e-10
    print(center, sign, round(total, 9))
```

## Exercise and explained answer

For the angular form choose R=0.5 and center (1,0). The disk avoids the origin, so the integral is zero and disk Stokes applies. Move the center to (0,0): boundary integral becomes 2π and disk Stokes no longer applies. At center (0.5,0), the boundary hits the excluded point and its ordinary integral is undefined. Verify all three states.

## Connection to scientific AI

Forms organize fluxes, conservation laws and coordinate-compatible differential operators. When a learned field appears curl free, a global potential still needs domain hypotheses. Holes, excluded singularities and boundary conditions can invalidate that inference even when all local derivatives look correct. A finite directional plot is an aid to the calculation, not a proof that a sampled field satisfies a global theorem.

Further reading: [Lorenzo Sadun, Lecture Notes on Differential Forms](https://arxiv.org/abs/1604.07862).
