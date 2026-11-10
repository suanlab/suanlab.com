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
