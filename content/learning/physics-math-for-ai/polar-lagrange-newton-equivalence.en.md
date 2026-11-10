## One motion, two descriptions

A camera reports Cartesian position, while a range-and-bearing sensor reports distance and angle. Neither sensor changes the physical motion. Nevertheless, a Cartesian acceleration component and the second derivative of a polar coordinate can be very different numbers. Lagrange equations provide a systematic way to obtain the correct coordinate equations from energy.

This lesson moves beyond instantaneous coordinate conversion. We derive the differential equations, solve the corresponding central-force motion in Cartesian coordinates, and then examine the polar description along that actual trajectory. A coordinate singularity will appear in one experiment even though position, velocity and force remain finite. This is a limitation of the description rather than a failure of Newtonian motion.

## Recall what stationary action requires

Choose fixed endpoint times and positions. For a smooth path q and a variation η that vanishes at both endpoints, vary the action S=∫L(q,qdot)dt. The product rule and integration by parts give

$$
\delta S=\left[\frac{\partial L}{\partial\dot q_i}\eta_i\right]_0^T
-\int_0^T\left[\frac{d}{dt}\frac{\partial L}{\partial\dot q_i}-\frac{\partial L}{\partial q_i}\right]\eta_i\,dt.
$$

Sum the displayed expression over coordinates i. The boundary term vanishes for these admissible variations. If the first variation vanishes for every sufficiently regular interior variation, the fundamental lemma of variational calculus sets each bracket to zero. This yields the Euler–Lagrange equations. Stationarity does not by itself establish a minimum; the previous lesson supplied examples of flat directions and saddles.

For the general mechanics formulation, see Russ Tedrake's MIT [Multi-Body Dynamics](https://underactuated.mit.edu/multibody.html). The equations below are derived directly for our polar model, including its domain restriction and units.

## Specify the system before differentiating

Let m>0 and choose the central potential U=kr²/2 with k≥0. The force in Cartesian coordinates is −kx. In a regular polar chart, r>0 and

$$
\boldsymbol x=r\begin{pmatrix}\cos\theta\\\sin\theta\end{pmatrix},\qquad
L=\frac m2(\dot r^2+r^2\dot\theta^2)-\frac k2r^2.
$$

Radius is a length, angle is in radians, and L has units J. The coordinate domain excludes the origin. An angular branch can also require a local restriction; repeated angles describe the same position. Do not interpret a principal-angle jump across a branch cut as an impulsive physical motion.

A partial derivative of L holds the other coordinate and velocity arguments independent. A total time derivative follows their joint evolution. Confusing those two operations is a common source of missing inertial terms.

## Derive the radial equation

The radial momentum is ∂L/∂rdot=m rdot, so its total derivative is m rddot. The coordinate derivative is ∂L/∂r=m r thetadot²−kr. Subtracting gives

$$
m\ddot r-mr\dot\theta^2+kr=0,
\qquad \ddot r=r\dot\theta^2-\frac{k}{m}r.
$$

Every term in the first equation has force units N. The positive r thetadot² term in the coordinate equation does not mean the spring pushes outward. It accounts for how the radial coordinate changes while the direction basis rotates. We will reconstruct the physical acceleration to identify the actual force direction.

## Derive the angular equation

The Lagrangian has no explicit theta dependence, but its angular momentum coefficient depends on radius:

$$
\frac{\partial L}{\partial\dot\theta}=mr^2\dot\theta,\qquad
\frac{d}{dt}(mr^2\dot\theta)
=m(r^2\ddot\theta+2r\dot r\dot\theta).
$$

Thus, for r>0,

$$
\ddot\theta=-\frac{2\dot r\dot\theta}{r}.
$$

The product rule produces the term with factor two. Dropping it would make the angular momentum mr²thetadot change without an applied torque. The angular Euler–Lagrange equation has torque units N m, distinct from the radial equation's N. Here we verify this conservation identity directly; the full symmetry-to-conservation theorem is a later topic.

## Recover Newton acceleration

The basis vectors satisfy erdot=thetadot etheta and ethetadot=−thetadot er. Differentiating position twice gives

$$
\boldsymbol a=(\ddot r-r\dot\theta^2)\boldsymbol e_r
+(r\ddot\theta+2\dot r\dot\theta)\boldsymbol e_\theta.
$$

Substitute both Lagrange equations. The physical radial component is −kr/m and the tangential component is zero. Therefore a=−(k/m)x, precisely the Cartesian Newton equation. Coordinate acceleration rddot is not the projection a·er; the latter includes the basis-rotation correction.

The velocity Hessian of L is diagonal with entries m and mr². It is invertible for positive radius, allowing the coordinate equations to determine accelerations. At r=0 that regular-coordinate argument fails. This does not make the Cartesian mass m vanish.

## Use an actual analytic trajectory

Write the natural frequency as Ω=√(k/m). Given initial Cartesian position x₀ and velocity v₀, the solution is

$$
\boldsymbol x(t)=\boldsymbol x_0\cos(\Omega t)
+\frac{\boldsymbol v_0}{\Omega}\sin(\Omega t),\qquad
\boldsymbol v(t)=-\Omega\boldsymbol x_0\sin(\Omega t)+\boldsymbol v_0\cos(\Omega t).
$$

When k=0 the separate free branch is x=x₀+t v₀ and v=v₀. It avoids division by zero. These formulas are analytic solutions; evaluating sine and cosine on a computer still uses floating-point arithmetic. The model does not claim exact decimal energy conservation.

For each regular position, recover r=|x|, rdot=x·v/r and thetadot=(x×v)/r². The plotted angle, when shown, uses a principal branch. Its rate is recovered from physical vectors and is not computed by subtracting wrapped angles.

## Explain the controls and time axis

The lab sets Ω=πf/4 s⁻¹ with integer f from zero to four and k=mΩ². Changing mass at fixed f also changes stiffness; it retains the natural frequency. Initial radius is the radius code divided by two meters, and initial radial velocity is its code divided by two m/s. Initial directions use multiples of π/2.

The angular-rate control W gives initial thetadot=W Ωref/2, where Ωref=Ω for f>0 and Ωref=1 s⁻¹ in the free branch. This permits an exact circular-orbit relation W=±2 without asking a decimal input to equal π. Query time ranges from zero to two seconds. It selects a point on the same physical trajectory rather than changing a solver step size or perturbing a path with epsilon.

## Worked example: straight motion with changing polar rates

Choose m=1, k=0, r₀=1, initial angle zero, rdot₀=1 and thetadot₀=1. Cartesian position is (1,0) and velocity is (1,1). Newton acceleration is zero. Yet initially rddot=1 and thetaddot=−2. The two rotating-basis corrections exactly cancel those coordinate accelerations in the physical vector.

After one second x=(2,1), r=√5, rdot=3/√5 and thetadot=1/5. Cartesian velocity remains (1,1), kinetic energy remains 1 J and angular momentum remains 1 kg m²/s. A changing angular rate does not imply a torque when radius changes too.

## Worked example: circular motion

Choose m=1, f=2, r₀=1, rdot₀=0 and W=2. Then Ω=π/2, initial angular rate equals Ω, and the trajectory is a circle. Radius stays constant, so radial coordinate acceleration is zero. Physical radial acceleration is nevertheless −π²/4 m/s².

Kinetic and potential energies are each π²/8 J, total energy is π²/4 J, and angular momentum is π/2 kg m²/s. At one second the point is (0,1) and velocity is (−π/2,0). The constant radius does not make the motion force-free.

## Worked example: cross the origin without breaking motion

Keep m=1 and f=2, but set both initial rates to zero and start at (1,0). The path is x=(cos(πt/2),0). At t=1 it passes through the origin with velocity (−π/2,0). Force and acceleration are zero at that instant, and the total energy is π²/8 J.

Radius and inverse polar angle/rates cannot form a regular coordinate description at that point. The implementation returns an undefined polar state and splits polar-rate curves around it. It retains the finite Cartesian trajectory. Analytic radial crossing times are inserted even when they fall between the regular output samples, so a polar curve never bridges an undefined origin. Known cardinal phase identities identify represented query-time origins exactly; other event times are floating-point evaluations of analytic roots, with zero position supplied by the root identity. No artificial small positive radius is inserted. Near an origin the chart remains defined but can be poorly conditioned, which is a different issue.

## Exercises with explanations

1. Why does the angular equation contain a factor two? **Solution:** Differentiate r²thetadot using the product rule. The derivative of r² is 2r rdot, so the full coefficient is r²thetaddot+2r rdot thetadot.
2. In the free example, calculate angular momentum after one second. **Solution:** m(x vy−y vx)=1(2−1)=1 kg m²/s. Equivalently mr²thetadot=1·5·1/5=1. The angular rate alone is not conserved.
3. A circular orbit has rddot=0. Is its physical radial acceleration zero? **Solution:** No. It equals rddot−r thetadot²=−rΩ² and requires the inward central force.
4. At the origin crossing, should the program divide by a small chosen radius? **Solution:** No. That would invent a coordinate state. Keep Cartesian position and velocity and report the inverse polar state as undefined.
5. Double mass while retaining f and the same initial coordinate data. **Solution:** Stiffness doubles, the physical trajectory stays the same, and energy and angular momentum double. This experiment fixes frequency rather than stiffness.
6. Does deriving Euler–Lagrange equations prove that the trajectory minimizes action over every horizon? **Solution:** No. The equations give stationarity under the stated boundary variations. The previous oscillator example showed how curvature can become flat or have both signs.

## Research connection and verification scope

A learned dynamics model should agree across regular coordinate descriptions. Compare physical accelerations, energy and angular momentum rather than equating coordinate second derivatives. Retain coordinate-domain information in an AI state representation so a singular inverse does not become a fabricated finite state.

The implementation is checked against independent 60-digit Cartesian flow and radius derivatives. Its force and energy diagnostics describe an analytic solution evaluated numerically. The next unit uses Lagrange equations for a coupled system whose trajectory needs numerical integration, so it must also expose integration and refinement errors.
