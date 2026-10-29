## Specify the quantity and measure

Along one curve, accumulate scalar density, tangential work or normal flux. These are different integrals:

$$
\int_C g\,ds=\int_a^b g(r(t))\|r'(t)\|\,dt,\qquad
\int_C F\cdot dr=\int_a^b F(r(t))\cdot r'(t)\,dt.
$$

Arc length ds is orientation-independent; dr changes sign when direction reverses. State which one the problem uses.

## Circle circulation versus outward flux

For the counterclockwise circle r(t)=(R cos t,R sin t), the rotational field (−y,x) gives

$$
\oint_C F\cdot dr=2\pi R^2,\qquad\oint_C F\cdot n\,ds=0.
$$

The radial field (x,y) instead has zero circulation and outward flux 2πR². Same geometry and field magnitude do not make tangential and normal integrals equal.

## Area and surface elements

Polar area measure is r dr dθ. Omitting its Jacobian gives incorrect disk area. For a regular parametrized surface X(u,v),

$$
dS=\|X_u\times X_v\|\,du\,dv,\qquad
\int_S F\cdot n\,dS=\int F(X)\cdot(X_u\times X_v)\,du\,dv.
$$

Swapping parameter order reverses oriented flux but preserves unsigned area. Check regularity and overlap when covering a surface with multiple patches.

## Volume and coordinate changes

Volume integrals accumulate spatial density. A regular one-to-one coordinate change uses the absolute Jacobian determinant. Spherical volume measure is r² sinφ dr dφ dθ, distinct from polar area:

$$
\int_{\|x\|\le R}1\,dV=\frac{4\pi R^3}{3},\qquad
\int_{x^2+y^2\le R^2}1\,dA=\pi R^2.
$$

Work, outward flow and contained mass have different units despite all being integrals.

## Scope and AI interpretation

The visualization is a planar circle with field arrows, not a rendered general three-dimensional surface. R=0 reports a degenerate limit, not a regular curve satisfying unchanged parametrization assumptions.

Path costs, continuous-model densities and sensor fluxes require clear measures. A dataset average approximating a surface integral needs the correct sampling distribution and area weights.

## Exercises and answers

1. Reverse the circle orientation: circulation becomes **−2πR²**.
2. A radius-two disk has area **4π**.
3. Omitting r in polar coordinates uses **the wrong area measure**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
