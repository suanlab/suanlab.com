## A central force conserves more than a circular picture suggests

A particle can orbit on an ellipse, move along a radial line or remain stationary under the same central force law. Zero torque about the force center does not select a circle. It constrains angular momentum; initial position and velocity determine the actual trajectory.

This lesson uses the isotropic planar harmonic force F=−kq about a fixed origin, with constant positive mass m and stiffness k. We solve the physical initial-value problem and compare angular quantities about the force center and another fixed point D=d. Changing D changes the reference used for measurement, not the physical force or orbit.

## Solve both spatial components with the same frequency

Newton's law gives

$$
\ddot q=-\omega^2q,\qquad \omega^2=\frac{k}{m}.
$$

With specified initial position q0 and velocity v0,

$$
q(t)=q_0\cos(\omega t)+\frac{v_0}{\omega}\sin(\omega t),
$$

$$
v(t)=-\omega q_0\sin(\omega t)+v_0\cos(\omega t).
$$

Both components use the same frequency because the stiffness is isotropic. The position is a linear image of the unit circle in sine/cosine coordinates, so the orbit is an ellipse or a degenerate line segment or point. The period of the nonstationary solution is 2π/ω. The diagram displays one full period, while the time panels retain a ten-second horizon.

## Derive center torque and angular momentum conservation

About the force center,

$$
\tau_C=(q\times(-kq))_z=0,\qquad
L_C=(q\times mv)_z=m(q_0\times v_0)_z.
$$

The last equality follows by substituting the trigonometric solution: cross terms with identical vectors vanish, and cos²+sin²=1 combines the remaining terms. Equivalently, differentiating LC gives the vanishing center torque.

The conserved value may be positive, negative or zero. It is fixed by initial data rather than by the phrase “central force.” A nonzero value prevents the particle from passing through the center, since that would require q=0 and hence L=0 in this nonsingular model.

## Energy provides a separate invariant

The harmonic potential is V=k|q|²/2, so

$$
E=\frac12m|v|^2+\frac12k|q|^2
=\frac12m|v_0|^2+\frac12k|q_0|^2.
$$

Angular momentum and energy are different constraints. Energy is nonnegative here, while angular momentum is signed. Preserving one does not automatically verify a numerical model's other invariant or identify its phase along an orbit. The interface therefore reports separate numerical residuals for both.

Initial angular momentum and the classification tests use exact rational control values. Trigonometric positions, velocities and query cross products are floating-point evaluations, so their conservation residuals are numerical diagnostics rather than exact-rational equalities.

## Circles require additional initial conditions

For a nonzero initial radius, circular motion requires

$$
q_0\cdot v_0=0,\qquad m|v_0|^2=k|q_0|^2.
$$

The first condition makes velocity tangential. The second makes the central force match the needed centripetal acceleration. Orthogonality alone is insufficient: the wrong tangential speed gives an ellipse. The interface checks both conditions exactly, including the requirement of nonzero radius.

When q0×v0=0, the initial vectors are collinear and motion stays on a line through the center. This radial motion can have substantial speed and energy while angular momentum vanishes. If both initial vectors are zero, the trajectory is the stationary center point. The radial flag includes that degenerate case; the separate stationary flag makes the distinction explicit.

## Define swept area as a signed integral

During a short interval, the radius vector sweeps an oriented triangular area. Taking its time limit gives

$$
\frac{dA_C}{dt}=\frac12(q\times v)_z=\frac{L_C}{2m},\qquad
A_C(t)=\frac{L_C}{2m}t.
$$

This is constant areal velocity about the force center. Clockwise orientation gives negative area. A radial trajectory has zero area even while the particle moves. Multiple turns accumulate signed area rather than resetting after each completed orbit.

The plotted quantity is therefore not the nonnegative area of a filled polygon or the union of all visited sectors. Its definition retains orientation and repeated traversal. That distinction matters when comparing a long-time area integral with the ordinary area enclosed by an ellipse.

## Moving the comparison marker changes angular quantities

The point D is fixed at displacement d from the force center. With laboratory momentum retained,

$$
L_D=L_C-(d\times p)_z,\qquad
\tau_D=(q-d)\times(-kq)=k(d\times q)_z.
$$

The center torque vanishes but the comparison torque generally varies. Consequently LD and areal velocity about D generally vary too. This does not indicate a failure of the central-force law: the point about which that law has zero torque is the actual force center.

Both points are fixed in one inertial frame. This is not the moving-reference-point problem and needs no fictitious coordinate force. The trajectory and physical force readouts remain unchanged when only d is edited.

## Integrate area about the comparison point correctly

Its instantaneous areal velocity is LD/(2m). Integrating the shift term gives

$$
A_D(t)=A_C(t)-\frac12\left[d\times(q(t)-q_0)\right]_z.
$$

The position-dependent correction is essential. Substituting the constant center areal velocity for the comparison one would silently use the wrong reference. After an entire closed orbit q returns to q0, so both accumulated areas agree over that complete traversal even though their instantaneous rates differed.

The independent verification integrates the differential equations together with each area's own rate and the comparison torque. It does not obtain the comparison area by declaring it equal to the center area.

## Read the three panels

The spatial panel shows one-period motion in meters, the force center, the comparison point and radius vectors to the query state. The angular panel compares center and comparison L in kg m²/s. The area panel compares their signed time integrals in m². Spatial arrows represent radii, not a hidden conversion of force into distance.

Try the elliptic example, then the radial example, then the displaced-reference circular example. Reverse the initial transverse velocity to reverse orientation. Set both initial vectors to zero to inspect a stationary degenerate plot. Change mass or stiffness to change frequency; change only the comparison point to leave physical motion fixed.

## Six exercises with worked solutions

1. **Exercise: ellipse from initial data.** Set m=k=1,q0=(2,0),v0=(0,1). **Solution:** q=(2cos t,sin t) and v=(−2sin t,cos t). The orbit is elliptical, LC=2, energy=5/2J and areal velocity=1m²/s.
2. **Exercise: area at a query time.** For exercise1, what is the area at1s? **Solution:** AC=1m². No angle inversion or polygon approximation is needed. A full period gives area2πm², matching the ellipse with semiaxes2 and1.
3. **Exercise: radial motion.** Set q0=(1,0),v0=0,m=k=1. **Solution:** q=(cos t,0),v=(−sin t,0). Angular momentum and area vanish, but the particle moves and energy remains1/2J.
4. **Exercise: the circle condition.** Take m=1,k=4,q0=(1,0). What tangential velocity makes a circle? **Solution:** v0=(0,2) or(0,−2), since m|v0|²=k|q0|². Speed1 gives an ellipse despite being perpendicular to q0.
5. **Exercise: displaced reference.** Set m=k=1,q0=(1,0),v0=(0,1),D=(1,0). **Solution:** LC=1,LD=1−cos t and τD=sin t. Areas are AC=t/2 and AD=(t−sin t)/2. They agree after2πs but generally differ before then.
6. **Exercise: clockwise repeated turns.** Reverse v0 in exercise5 and set D=0. At10s? **Solution:** LC=−1 and A=−5m². Its magnitude exceeds the disk areaπ because the integral counts repeated signed sweeping, rather than only the union of visited locations.

## A central-force statement is not the entire Kepler model

Zero torque and equal areas follow from the force direction. They do not by themselves imply an inverse-square law, a focus-centered gravitational ellipse or Kepler's period–size relation. Our harmonic ellipse is centered on the force origin and has a frequency independent of amplitude. More general central forces can produce different orbit shapes and precession.

For learned simulators, evaluate trajectory phase and force law alongside invariant residuals and reference-point consistency. Matching angular momentum alone is insufficient to identify the dynamics.

## Background and scope

For central-force angular conservation, see [MIT Physics8.01, Central Force Motion](https://web.mit.edu/8.01t/www/materials/Presentations/Presentations_f10/Presentation_W14D1.pdf). The harmonic trajectory and signed-area examples above follow from the stated model. Singular forces, moving reference points and rigid-body spin are separate topics.
