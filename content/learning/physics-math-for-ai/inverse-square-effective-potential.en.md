## From a force vector to an allowed radial interval

A drawing of an ellipse does not explain why an orbit is possible. We first ask a more basic question: given an initial position and velocity, which distances from the force center can the particle reach? This turns a two-dimensional differential equation into a one-dimensional energy constraint without pretending that the constraint already determines time.

Imagine throwing a small object sideways near a massive source. Attraction pulls inward, but its sideways velocity gives angular momentum. Reaching a smaller radius then requires greater tangential speed. The resulting kinetic-energy cost can prevent approach to the origin even though the physical force is attractive everywhere. The effective potential records this competition.

## Declare the physical model and its units

The source is fixed at the origin of an inertial frame. The orbiting particle has constant positive mass m. We prescribe the acceleration, rather than solving a moving two-body source:

$$
\ddot q=-\frac{\mu q}{r^3},\qquad r=|q|>0,\qquad \mu>0.
$$

Here μ has units m³/s² and is held fixed when the particle mass changes. The force is m times this acceleration. The physical potential energy is −mμ/r with zero at infinite separation. We neglect drag, other sources, relativistic corrections and finite-size contact. At r=0 the equation is singular; the experiment supplies no rule for passing through that point.

The initial position is q₀=(r₀,0). Initial velocity is (vᵣ₀,vₜ₀), where the second component is signed tangential velocity. An inward initial radial speed is allowed. A negative tangential speed reverses angular orientation rather than making the radius negative.

## Derive the two conserved quantities

The force is central, so its torque q×F vanishes. Consequently the specific angular momentum h and total angular momentum L are constant:

$$
h=q_xv_y-q_yv_x=r^2\dot\theta=r_0v_{t0},\qquad L=mh.
$$

The potential is time independent. Dotting the acceleration equation with velocity gives conservation of the specific mechanical energy:

$$
\epsilon=\frac{v^2}{2}-\frac\mu r
=\frac{v_{r0}^2+v_{t0}^2}{2}-\frac\mu{r_0}.
$$

“Specific” means per unit mass. Its units are m²/s², not joules. The corresponding total energy is mε. Changing m at fixed μ and the same initial velocity scales total energy and angular momentum, while leaving this prescribed acceleration and the orbit unchanged.

## Eliminate the tangential speed, not the angular motion

Write v²=ṙ²+r²θ̇² and substitute θ̇=h/r². The energy becomes

$$
\frac{\dot r^2}{2}+U_{\rm eff}(r)=\epsilon,\qquad
U_{\rm eff}(r)=-\frac\mu r+\frac{h^2}{2r^2}.
$$

The first term is the physical potential per mass. The second is the tangential kinetic energy required at radius r by conserved h. It is called the angular barrier. It does not introduce an additional physical outward interaction in the original inertial vector equation.

The reduction preserves the angular motion through θ̇=h/r². Forgetting this relation would turn a useful reduction into an incorrect one-dimensional physical model.

## Read admissibility before taking a square root

A queried positive radius is allowed only when ε−Ueff(r)≥0. At an allowed radius,

$$
\dot r=\pm\sqrt{2(\epsilon-U_{\rm eff}(r))},\qquad v_t=\frac h r.
$$

Both signs must be retained away from a turning point. They describe inward and outward branches of the energy constraint. The display does not silently choose the future branch from a radial query alone. Initial data and a time solution would be needed for that question.

If the difference is negative, the state is forbidden: the radial speed is undefined, not zero. At equality the radial speed is zero, but tangential speed can remain nonzero. An apsis is generally not a point where the particle is at rest.

The control values define exact rational initial data and query radii. The sign of the radial kinetic budget is tested exactly, avoiding a tolerance that could turn a small forbidden interval into an allowed one. Square roots and spatial geometry are numerical evaluations.

## Circular balance and local radial stability

For nonzero h, the effective potential has a stationary point:

$$
r_c=\frac{h^2}{\mu},\qquad
U_{\rm eff}(r_c)=-\frac{\mu^2}{2h^2},\qquad
U_{\rm eff}''(r_c)=\frac\mu{r_c^3}>0.
$$

The positive curvature establishes a local radial minimum at fixed h. Circular motion also needs the energy to equal this minimum; merely querying rc does not make an arbitrary initial state circular. At that energy, the only allowed radius is rc, while the particle continues moving tangentially.

The initial circular condition is vᵣ₀=0 and vₜ₀²=μ/r₀. Its period depends on radius. This differs from the previously studied isotropic harmonic force, whose oscillation frequency is independent of amplitude. A result about one central force law should not be copied to all central forces.

## Recover conic geometry from an invariant vector

For this inverse-square law, the eccentricity vector is constant:

$$
\mathbf e=\frac{v^2q-(q\cdot v)v}{\mu}-\frac q r.
$$

At the chosen initial point it becomes

$$
\mathbf e=\left(\frac{r_0v_{t0}^2}{\mu}-1,
-\frac{r_0v_{r0}v_{t0}}{\mu}\right),\qquad
e^2=1+\frac{2\epsilon h^2}{\mu^2}.
$$

For h≠0, let f be the angle measured from the direction of this vector and p=h²/μ. Then r=p/(1+e cos f), with a strictly positive denominator. Negative, zero and positive energy give an ellipse, parabola and hyperbola respectively; e=0 is the circular limit. In a circle the periapsis direction is undefined, so a chosen drawing phase is only a convention.

## Turning radii and finite drawing windows

For nonradial motion, the periapsis is rp=p/(1+e). This formula avoids subtracting nearly equal terms when angular momentum is small. For negative energy the apoapsis is ra=−μ/ε−rp. Parabolic and hyperbolic orbits have no finite apoapsis.

The spatial panel is orbit geometry, not a record of elapsed motion. Its drawing window ends at radius 12 m. Unbound branches and large ellipses are explicitly clipped there. The implementation never joins across a hyperbolic asymptote, evaluates a negative polar radius, or presents the clipped endpoint as a physical turning point. The initial point is included in the sampled geometry.

The radial energy panel has its own positive-radius window. Its lower bound is three quarters of the smallest initial, queried or nonradial periapsis radius, preserving the energy well without an unrelated near-origin scale dominating the picture. The upper bound is12m; the window is a display choice, not the physical allowed interval. Its x-axis is log10(r/m): an increase of one unit multiplies radius by ten. Specific energy on the vertical axis is not transformed. Values near a tiny periapsis require dense sampling; the calculation includes analytic turning and circular radii where they lie in that window. Axis labels distinguish radius from time and specific energy from total energy.

## Handle the zero-angular-momentum branch separately

When h=0, the barrier vanishes and θ cannot be used as an independent variable. The invariant identity gives e=1 for every such initial state, including negative-energy radial falls. Therefore “e=1 means parabolic motion” requires the nonzero-angular-momentum qualification.

A negative-energy radial state has an outer turning radius −μ/ε and can approach the singular origin. A nonnegative-energy radial state has no finite outer turning radius. Initial radial direction still matters. The experiment reports the radial class and allowed radii, leaves conic geometry undefined, and supplies no invented collision continuation.

## Research extension: certify an orbit family before reporting its distant turning point

A tiny uncertainty near escape can produce a large interval of possible apoapsides and periods. A positive nominal estimate of an elliptic parameter does not certify that every compatible orbit is bound. This extension makes that statement quantitative, while preserving the native exact-sign test for its declared control inputs.

Keep the same fixed-source acceleration and known positive mu. Start at a known periapsis radius rp with zero radial velocity and positive tangential speed vp. We use a nonradial periapsis family; the h=0 collision branch remains separate. Define the dimensionless distance delta from the parabolic boundary:

$$
\delta=1-e=2-\frac{r_pv_p^2}{\mu},\qquad
\epsilon=-\frac{\mu\delta}{2r_p},\qquad h^2=\mu r_p(2-\delta).
$$

For delta<1 the starting point is periapsis and h is nonzero. Then 0<delta<1 gives a noncircular ellipse, delta=0 a parabola, and delta<0 a hyperbola. The identities use this initial-state preparation. They are not a replacement for the general eccentricity-vector calculation when radial speed is nonzero or the initial point is an apoapsis.

### Identify the inverse dependence before drawing a finite window

In the elliptic part of this family:

$$
a=\frac{r_p}{\delta},\qquad r_a=r_p\left(\frac2\delta-1\right),\qquad
T=2\pi\sqrt{\frac{r_p^3}{\mu}}\,\delta^{-3/2}\quad(0<\delta<1).
$$

As delta approaches zero from above, periapsis stays fixed but the distant turning radius grows like its reciprocal and period like delta to the power −3/2. The original twelve-metre geometry window is a rendering choice. Its clipped endpoint cannot be used as that distant turning point.

### Propagate a certified parameter interval

Suppose rp and mu are known exactly, and an estimate d of delta has a certified absolute error eta. If the whole interval stays in 0<delta<1, the monotonicity of a and ra gives exact bounds:

$$
|\delta-d|\le\eta,\quad 0\le\eta<d,\quad d+\eta<1,\qquad
\frac{r_p}{d+\eta}\le a\le\frac{r_p}{d-\eta},\qquad
r_p\left(\frac2{d+\eta}-1\right)\le r_a\le r_p\left(\frac2{d-\eta}-1\right).
$$

Every bound keeps units: d and eta are dimensionless, while a,ra,rp are lengths. Period is increasing with a, so the same endpoints bound T. Its relative interval and the sensitivities are:

$$
\left(\frac d{d+\eta}\right)^{3/2}\le\frac{T}{T(d)}\le
\left(\frac d{d-\eta}\right)^{3/2},\qquad
\frac{dr_a}{d\delta}=-\frac{2r_p}{\delta^2},\qquad
\frac{d\log T}{d\log\delta}=-\frac32.
$$

The large derivative of ra is an absolute sensitivity. When eta/d remains small, relative period uncertainty is controlled by that ratio; do not describe every relative perturbation as infinitely amplified. Near zero, a fixed absolute eta can instead become large relative to d, or cross the classification boundary entirely.

Use rp=1 m,mu=2 m³/s²,d=1/100,eta=1/1000:

| quantity | lower bound | nominal value | upper bound |
|---|---|---|---|
| a (m) | 1000/11 | 100 | 1000/9 |
| ra (m) | 1989/11 | 199 | 1991/9 |
| T² divided by 4pi²rp³/mu | (1000/11)³ | 100³ | (1000/9)³ |

The period bounds are approximately 3851.021 and 5203.566 s, around nominal 4442.883 s. These are deterministic interval bounds under the stated calibration assumptions, not confidence intervals.

![Apoapsis uncertainty grows near the parabolic boundary; a second panel shows that a positive nominal delta does not certify an ellipse when its interval includes zero.](/assets/learning/classical-apsis-uncertainty.en.svg)

The upper panel uses logarithmic axes for d and apoapsis in metres, with eta=0.001 and only intervals contained in the elliptic family. The orange point is d=0.01,ra=199 m. The lower panel plots two dimensionless delta intervals against the zero boundary. It does not share the upper radius axis.

### Refuse a finite apoapsis when the interval crosses escape

For d=1/1000,eta=1/500 the admissible delta interval is [−1/1000,3/1000]. The nominal state is elliptic, but parabolic and hyperbolic states are also permitted. The useful decision is:

$$
0<d-\eta\le\delta\le d+\eta<1\ \Longrightarrow\ \text{noncircular ellipse},\qquad
d-\eta\le0\le d+\eta\ \Longrightarrow\ \text{no finite apoapsis certificate for the whole interval}.
$$

An interval touching zero already includes a state with no finite apoapsis. Reporting a finite upper bound by substituting only the positive center would hide that possibility. This is uncertainty amplification and classification margin, rather than the exact full-history nonidentifiability of the earlier friction example. Additional precision or independent calibration may resolve it.

### Carry velocity uncertainty into the dimensionless margin

For a calibrated speed estimate vhat with an absolute error bound xi, expand the difference of squares and apply the triangle inequality:

$$
|v_p-\widehat v|\le\xi,\qquad
\widehat\delta=2-\frac{r_p\widehat v^2}{\mu},\qquad
|\delta-\widehat\delta|\le\frac{r_p}{\mu}\left(2|\widehat v|\xi+\xi^2\right).
$$

The right side is dimensionless: rp/mu multiplies a squared-speed error. An additional example uses vhat=1.99 m/s,xi=0.001 m/s with the same rp,mu. It gives delta_hat=399/20000 and an error bound 3981/2000000. These are distinct from the d=0.01 interval used in the table.

If rp,mu or the exactly tangential initial preparation is uncertain too, this speed-only expression does not certify the full uncertainty. A rounded native velocity code also does not exactly represent an arbitrarily precise measurement. The native model correctly tests the rational energy of its specified inputs; this analysis does not allege a failure of that exact-sign calculation.

### Reproduce exact intervals and native classification anchors

The code uses fractions for the parameter intervals, speed-error propagation and energy signs. Only the period values use floating square roots and pi. It rejects an interval crossing the boundary rather than attaching a finite ellipse bound to it.

```python
from fractions import Fraction as F
from math import pi, sqrt

rp, mu = F(1), F(2)
def ellipse_bounds(d, eta):
    if not (0 <= eta < d and d+eta < 1):
        raise ValueError("The whole interval must stay in 0 < delta < 1")
    low, high = rp/(d+eta), rp/(d-eta)
    return low, high, 2*low-rp, 2*high-rp

d, eta = F(1, 100), F(1, 1000)
a_lo, a_hi, ra_lo, ra_hi = ellipse_bounds(d, eta)
assert (a_lo, a_hi, ra_lo, ra_hi) == (F(1000, 11), F(1000, 9), F(1989, 11), F(1991, 9))
for delta in [d-eta, d, d+eta]:
    a, ra = rp/delta, rp*(2/delta-1)
    assert a_lo <= a <= a_hi and ra_lo <= ra <= ra_hi
    assert (rp/(d+eta))**3 <= a**3 <= (rp/(d-eta))**3
print("a and apoapsis bounds:", a_lo, a_hi, ra_lo, ra_hi)
print("period bounds (s):", 2*pi*sqrt(float(a_lo**3/mu)), 2*pi*sqrt(float(a_hi**3/mu)))
try:
    ellipse_bounds(F(1, 1000), F(1, 500))
except ValueError:
    pass
else:
    raise AssertionError("A boundary-crossing interval must not certify an ellipse")
vhat, xi = F(199, 100), F(1, 1000)
dhat = 2-rp*vhat*vhat/mu
error = rp*(2*abs(vhat)*xi+xi*xi)/mu
for velocity in [vhat-xi, vhat, vhat+xi]:
    assert abs((2-rp*velocity*velocity/mu)-dhat) <= error
assert error == F(3981, 2000000)
print("delta estimate and error bound:", dhat, error)
for velocity in [F(19, 10), F(2), F(21, 10)]:
    epsilon = velocity*velocity/2-mu/rp
    print("native speed anchor and specific energy:", velocity, epsilon)
```

The original geometry control order is (mu,2r0,10vr,10vt,10rQuery,mass). Codes (2,2,0,19,20,1),(2,2,0,20,20,1),(2,2,0,21,20,1) give an ellipse, parabola and hyperbola with specific energies −39/200,0,41/200 m²/s² and periapsis 1 m. The finer delta family above is a standalone research calculation, not a newly available velocity-input resolution.

### Worked questions for an orbital surrogate

1. **Does a positive d establish a finite apoapsis for all compatible states?** Only when its certified interval stays strictly positive and within the declared family. A nominal label alone is insufficient.
2. **Why can a twelve-metre picture be misleading near escape?** The geometric window clips large radii; it does not measure a physical turning point or close an unbound orbit.
3. **Which information should an estimated period include?** Report the state preparation, units, calibration uncertainties, classification margin and propagated interval. Separate those from the numerical rendering and the exact diagnostics for a particular control code.

The interval and sensitivity formulas are derived from the same inverse-square invariants used above.

## Worked examples and exercises with solutions

1. **Circular equality.** Set μ=1, r₀=1, vᵣ₀=0 and vₜ₀=1. **Solution:** h=1, ε=−1/2 and e=0. The minimum is at rc=1. At queried r=2, Ueff=−3/8, so ε−Ueff=−1/8: that radius is forbidden for these conserved quantities.

2. **An ellipse from an apsis.** Keep μ=r₀=1 and set vₜ₀=1/2, vᵣ₀=0. **Solution:** h=1/2, ε=−7/8 and e=3/4. The starting point is the apoapsis ra=1; periapsis is rp=1/7. Zero radial speed at the initial point does not imply a circular orbit.

3. **Parabolic threshold.** Set μ=2, r₀=1, vᵣ₀=0 and vₜ₀=2. **Solution:** ε=0, h=2, p=2 and e=1. The periapsis is 1 and there is no finite apoapsis. The denominator condition excludes the limiting direction where distance diverges.

4. **Two radial branches.** Set μ=r₀=1 and vᵣ₀=vₜ₀=1, and query r=1. **Solution:** ε=0 and h=1. Ueff=−1/2, so the allowed radial speeds are −1 and +1. The positive initial choice does not eliminate the inward branch from the energy surface.

5. **Radial degeneracy.** Set μ=r₀=1 and both initial velocity components to zero. **Solution:** h=0 and ε=−1. Although the eccentricity vector has norm 1, this is a bound radial fall, not a nondegenerate parabola. Its outer turning radius is 1, and queried r=3 is forbidden.

6. **Mass and orientation.** Multiply m by four, then separately reverse vₜ₀. **Solution:** the mass change multiplies total energy and L by four while preserving h, ε and the orbit. Tangential reversal changes the sign of h and reflects the eccentricity-vector y component; radial admissibility depends on h² and is unchanged.

## Connection to AI research and further reading

This experiment separates a model invariant, an admissibility certificate and a time prediction. A learned model can fit a plausible curve while violating the energy budget, crossing an excluded radius, or confusing radial degeneracy with an unbound orbit. These are distinct evaluation failures. Testing exact initial invariants, singular limits and direction reversals gives more information than checking whether an image looks elliptical.

The next lesson adds actual time on bound ellipses through Kepler's equation. For a primary derivation reference, see the [MIT OpenCourseWare Kepler problem notes](https://ocw.mit.edu/courses/8-282j-introduction-to-astronomy-spring-2006/93b6bc699c653da0efd67170a90c7c81_kepler_prob.pdf). The interactive calculation and examples here use the explicitly declared fixed-source model above.
