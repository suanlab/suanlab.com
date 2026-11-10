## An ellipse needs a clock

An orbit equation gives possible positions, not the time spent at each position. Moving a dot around an ellipse at uniform drawing angle usually produces the wrong velocity, angular momentum and force. This lesson adds a physical clock to the inverse-square ellipse by solving Kepler's equation.

Picture a satellite passing close to its source. It crosses the near side quickly and spends more time far away. Equal time intervals sweep equal areas about the force center, not equal distances along the ellipse and not equal angles about its geometric center. We will calculate position and velocity together so this statement can be checked rather than merely illustrated.

## Choose a collision-free bound family

We keep the fixed-source inertial model q̈=−μq/r³ with positive μ and mass m. Choose a positive semimajor axis a and eccentricity 0≤e≤0.95. The initial point is periapsis, the closest point to the focus. For e=0, every point is equally close; the initial point then means a chosen phase rather than a distinguished periapsis.

The allowed family has

$$
r_p=a(1-e)>0,\qquad r_a=a(1+e),\qquad b=a\sqrt{1-e^2}.
$$

Consequently these trajectories never reach the singular origin. The upper eccentricity limit keeps all control combinations strictly elliptic. It is a teaching range, not an approximation that sets all larger eccentricities equal to 0.95. This lesson does not solve parabolic flight, hyperbolic flight or radial collision time; those require other parametrizations and domain decisions.

A fixed orientation angle rotates the ellipse in the laboratory. A direction selector changes prograde to retrograde motion. Neither operation changes a, e, the period or the specific energy.

## Distinguish the focus from the ellipse center

With the periapsis direction along positive x, the eccentric anomaly E parametrizes the ellipse as

$$
x=a(\cos E-e),\qquad y=b\sin E.
$$

The force center is (0,0), one focus. The geometric ellipse center is (−ae,0), which generally is not a force center. E is an auxiliary parameter measured through a related circle; it is not the polar angle of q about the focus. Adding the offset −ae is essential. An otherwise attractive-looking centered ellipse would implement a different physical geometry.

A fixed rotation matrix transforms both q and v into the chosen laboratory orientation. Rotating position while leaving velocity unrotated would violate the initial tangency and alter the measured angular momentum.

## Derive the time equation from the area law

Define the positive mean motion n and period T by

$$
n=\sqrt{\frac\mu{a^3}},\qquad T=\frac{2\pi}{n}.
$$

Let s=+1 for prograde and −1 for retrograde motion. Starting at periapsis at t=0, the signed mean anomaly is M=snt. Kepler's equation is

$$
E-e\sin E=M.
$$

One way to understand the sine correction is to compute the focus-centered area swept as the auxiliary angle changes. Differentiating q with respect to E gives q×dq/dE=ab(1−e cos E). Integrating from E=0 yields A(E)=ab(E−e sin E)/2. Requiring constant areal velocity then gives the time relation above.

This also explains why uniform E is incorrect when e>0: the factor 1−e cos E is not constant. Mean anomaly grows uniformly in time, while eccentric anomaly generally does not.

## Solve a monotone implicit equation safely

The derivative of the left side is 1−e cos E≥1−e>0. Therefore the equation has exactly one root for every finite M in this elliptic range. Since |e sin E|≤e, a guaranteed mathematical bracket is [M−e,M+e].

The numerical solver starts within that bracket. It attempts a Newton step using derivative 1−e cos E. If the proposed step leaves the bracket, it bisects instead. Each evaluation updates the appropriate bracket endpoint. The output reports the residual E−e sin E−M and iteration count.

A residual is evidence of solving the implicit equation, not by itself proof of the entire physical implementation. The position formula, velocity derivative, force center and time scaling can still be wrong independently. We therefore check them using high-precision roots and a separate Cartesian differential-equation integration.

The circular case e=0 returns E=M directly. The zero-time root is also exact. The solver retains unwrapped anomalies instead of reducing them to a principal angle. This matters for repeated revolutions and accumulated area.

## Differentiate with respect to actual time

Implicit differentiation gives

$$
\dot E=\frac{s n}{1-e\cos E}.
$$

The compatible velocity is

$$
v_x=-a\sin E\,\dot E,\qquad
v_y=b\cos E\,\dot E.
$$

These are derivatives of the displayed position, not an independently chosen tangent arrow. At periapsis, the denominator is 1−e, so the motion is fast. At apoapsis it is 1+e, so the motion is slow. Retrograde motion reverses the signed evolution of E; its positive-time position is the reflection across the periapsis axis of the corresponding prograde point before applying the laboratory rotation.

The radius identity r=a(1−e cos E) stays strictly positive. Acceleration and force are evaluated about the focus using −μq/r³ and −mμq/r³.

## Check energy, angular momentum and speed

For every time in this family,

$$
\epsilon=-\frac\mu{2a},\qquad
h=s\sqrt{\mu a(1-e^2)},\qquad L=mh.
$$

The instantaneous kinetic and potential energies are K=m|v|²/2 and U=−mμ/r. Their sum must equal mε. The speed also satisfies the vis-viva relation

$$
|v|^2=\mu\left(\frac2r-\frac1a\right).
$$

K and U change along an eccentric orbit, even though total energy is constant. A panel that displays only total energy would conceal that exchange. Conversely a changing kinetic energy alone would not imply dissipation. There is no drag or heat term in this model.

## Area is signed and accumulated

The focus-centered areal velocity and swept area are

$$
\dot A=\frac{q\times v}{2}=\frac h2,\qquad A(t)=\frac{ht}{2}.
$$

The second evaluation A(E)=ab(E−e sin E)/2 provides an independent geometric consistency check once E has been solved. After a full prograde revolution the area increment is πab; a retrograde revolution contributes −πab. After many revolutions it continues accumulating. Reducing E modulo 2π before evaluating this formula would incorrectly erase those full-ellipse contributions.

Here area means the oriented integral of q×dq/2 along the traversed path. Repeated traversal can exceed the area of the geometric ellipse. It is not the area of the union of visited pixels on the screen.

## Read the geometry and time plots separately

The full closed ellipse is reference geometry over one revolution. The actual path stops at the selected query time; at t=0 it contains only the initial point. A query dot identifies the computed state. A full reference ellipse must not imply that every point has already been visited.

Time panels span 0 to 20 seconds. Radius, speed, energy and accumulated area have different physical units and need separate axes or panels. The query is explicitly included in the samples, as are periapsides and apoapsides within the interval. Sampling uniformly in eccentric anomaly and converting each sample to its actual Kepler time places more samples near the short, rapid periapsis passage. A coarse uniform-time grid can miss that feature at high eccentricity.

Sampling is a rendering decision; it does not replace the implicit time solution. The queried state is solved directly, so dragging the query does not merely select a nearby precomputed frame.

## Worked examples and exercises with solutions

1. **Circular limit.** Set μ=a=1, e=0, prograde direction and t=1. **Solution:** n=1, E=M=1, q=(cos 1,sin 1), speed=1, radius=1 and T=2π. Area is 1/2. The force center and ellipse center coincide only because e=0.

2. **Initial eccentric state.** Set μ=2, a=2, e=1/2, m=3 and t=0. **Solution:** rp=1, ra=3, b=√3 and the initial speed is √3. Initial acceleration is (−2,0), total energy is −3/2 and angular momentum is 3√3. The ellipse center is (−1,0), while the force acts toward (0,0).

3. **Period scaling.** Double a while holding μ and e fixed. **Solution:** T increases by 2 to the power 3/2. Quadrupling μ at fixed a instead halves T. The invariant relation is T²/a³=4π²/μ. Eccentricity changes the shape and timing within a revolution, but does not enter this period at fixed a and μ.

4. **Equal areas, unequal angles.** Compare a high-eccentricity orbit near periapsis and apoapsis over equal short time intervals. **Solution:** each interval contributes hΔt/2, but the near-side interval traverses a larger angular change and higher speed. Uniform polar-angle animation would violate the area law.

5. **Reverse direction and rotate.** Reverse the direction selector, then rotate the orientation by π/2. **Solution:** direction reversal negates h, L and accumulated area while preserving energy and radius at equal positive time. Rotation maps (x,y) to (−y,x) and transforms velocity in the same way. Neither operation changes the period.

6. **Multiple revolutions.** Use μ=6, a=1/2, e=0.95 and t=20. **Solution:** n=√48, so the orbit completes more than 22 revolutions. E must remain greater than 40π in prograde motion, and accumulated area exceeds many ellipse areas. A principal-angle root would display a plausible current position but the wrong total area and revolution history.

## Research use and verification limits

A learned dynamics model should be checked at the focus, at high eccentricity and across long phase accumulation, not only on a nearly circular picture. Coordinate accuracy, energy conservation, timing accuracy and area accumulation are separate requirements. Small invariant residuals can coexist with a phase error, so independent position and velocity references remain necessary.

The accompanying calculation uses 70-digit bisection roots and independent Cartesian inverse-square integrations as references. Floating-point residuals are reported numerically rather than labeled exact. The physical assumptions remain fixed-source Newtonian attraction and a collision-free elliptic family. For the underlying construction, see the [MIT OpenCourseWare Kepler problem notes](https://ocw.mit.edu/courses/8-282j-introduction-to-astronomy-spring-2006/93b6bc699c653da0efd67170a90c7c81_kepler_prob.pdf).
