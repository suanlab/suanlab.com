## No work does not mean no force

Imagine a force that always pushes sideways relative to the current velocity. It changes the direction of motion without changing speed. Kinetic energy is constant, yet the vector momentum rotates. This is a concrete reason not to treat energy, linear momentum and angular momentum conservation as interchangeable statements.

We select one positive constant point mass in an inertial plane. A prescribed external velocity-dependent force acts; the mechanism producing it is outside the selected subsystem. The model is deliberately limited to its stated motion law. Zero work does not make this particle an isolated system.

## Specify the quarter-turn operator and sign convention

Define J(vx,vy)=(−vy,vx), a counterclockwise ninety-degree rotation. It obeys J²=−I and v·Jv=0. The declared force is

$$
F=m\Omega Jv,\qquad \dot v=\Omega Jv,
$$

where Ω is a signed constant with units s⁻¹. Positive Ω rotates velocity counterclockwise and negative Ω rotates it clockwise. Its magnitude is an angular frequency, not cycles per second.

This is a velocity-dependent force. The same position with different velocities generally has different forces, so it is not a global position-only potential-gradient model. The interface does not introduce an arbitrary scalar potential to explain away the velocity dependence.

## Derive kinetic-energy conservation directly

Power is force dotted with actual velocity:

$$
\frac{dK}{dt}=F\cdot v=m\Omega(Jv)\cdot v=0,
\qquad K=\frac12m|v|^2.
$$

Thus speed and kinetic energy remain their initial values. The accumulated work is zero. This statement concerns scalar power. The momentum law remains dp/dt=F, so nonzero force changes vector momentum even while its magnitude stays fixed.

The numerical query power may contain floating-point roundoff from cancellation. Work is analytically zero by the perpendicular-force identity; it is not defined as the negative of an unexplained energy residual. Independent numerical integration checks the work rate and external impulse.

## Solve the rotating velocity

Let R(θ)=cosθ I+sinθ J. The initial-value solution is

$$
v(t)=R(\Omega t)v_0
=v_0\cos(\Omega t)+Jv_0\sin(\Omega t).
$$

Rotation preserves vector length. The two momentum-component curves are generally sinusoidal and neither needs to remain constant. The zero-velocity initial state stays stationary, even if Ω is nonzero, because its force remains zero.

The exact conservation classification uses the input values: vector momentum is conserved throughout the motion only in the Ω=0 branch or the stationary state. A momentum component can have zero derivative at an individual instant without satisfying that whole-motion condition.

## Integrate position and handle the free branch

For Ω≠0,

$$
q(t)=q_0+\frac{\sin(\Omega t)}{\Omega}v_0
+\frac{1-\cos(\Omega t)}{\Omega}Jv_0.
$$

At Ω=0 the model explicitly uses q=q0+v0t and v=v0. It does not divide by zero or invent a finite circle center. For numerical stability at short times, the nonzero branch evaluates sine ratios with a sinc function and the cosine difference using a half-angle identity. The exact limit is the same free motion.

Query time ranges over ten seconds. The spatial panel uses one full period when Ω is nonzero and a ten-second line when Ω is zero. Its horizon is explicitly distinguished from the time panels' fixed ten seconds.

## Find the guiding center without adding another body

For Ω≠0 define

$$
C=q+\frac{Jv}{\Omega}.
$$

Differentiating gives Cdot=v+J(ΩJv)/Ω=v−v=0. Therefore the trajectory is a circle about the fixed C for nonzero speed, with radius |v0|/|Ω| and period 2π/|Ω|. C is derived from the initial state; it is not a second physical particle or a new applied force.

The center is undefined in this formula at Ω=0. If v0=0 and Ω≠0, C=q0 and the spatial orbit degenerates to a point. A reference marker does not imply a nonzero orbit radius.

## Mechanical angular momentum generally changes

About the fixed laboratory origin, let L=(q×mv)z. Then

$$
\dot L=(q\times F)_z=m\Omega(q\cdot v).
$$

This torque is generally nonzero despite zero work. Work uses a dot product with velocity; torque uses a cross product with a reference-dependent position. There is no contradiction in having one vanish and the other remain nonzero.

A special orbit whose guiding center is the origin has q perpendicular to v, making mechanical L constant. The interface distinguishes that case exactly from a generic displaced circle. Energy conservation therefore permits several combinations of other conserved quantities rather than imposing one universal pattern.

## Verify a modified angular invariant

Since d|q|²/dt=2q·v,

$$
I=L-\frac{m\Omega}{2}|q|^2,
\qquad \dot I=0.
$$

This is a different quantity from mechanical L. It has the same units kg m²/s, but a different definition. We establish its conservation by differentiation rather than invoking an unstated general theorem.

The torque integral also satisfies ΔL=mΩΔ|q|²/2. The numerical angular-balance residual checks that identity independently of treating L itself as conserved. In the free branch the correction vanishes and I reduces to the ordinary conserved angular momentum.

## State what is external and what is not modeled

The selected particle receives an external force and generally an external torque. The apparatus or field producing that force is not included. Preserving the particle's kinetic energy does not establish momentum conservation for the particle alone.

A uniform perpendicular magnetic field can motivate this type of velocity rotation, with the appropriate charge/field sign absorbed into Ω. We do not model field dynamics, radiation, electric acceleration, spin or relativistic corrections. A full apparatus-plus-field conservation budget is a separate problem.

## Read the four diagrams

The spatial panel shows actual motion in meters, the fixed origin and the guiding center when defined. The momentum panel compares Px and Py in kg m/s. The angular panel compares mechanical L and modified I in kg m²/s. The energy panel shows K in joules.

Initial K,L,I and guiding-center coordinates are exact rational calculations from control codes. Trigonometric motion is evaluated numerically. Reverse Ω to change bending direction, move q0 to change the reference-dependent angular history, and use Ω=0 to inspect the free limit. These controls change specified parts of the physical model rather than decorative animation parameters.

## Six exercises with worked solutions

1. **Exercise: work-free bending.** Set m=1,Ω=1,q0=0,v0=(1,0). **Solution:** v=(cos t,sin t),q=(sin t,1−cos t),K=1/2J. The external force is nonzero and rotates momentum.
2. **Exercise: angular quantities.** For exercise1, find L and I. **Solution:** L=1−cos t and |q|²=2(1−cos t). Hence I=0 although mechanical L varies. External torque is sin t.
3. **Exercise: origin-centered special case.** Set q0=(1,0),v0=(0,1),m=Ω=1. **Solution:** C=0,q=(cos t,sin t),L=1 and I=1/2. Linear momentum still changes direction, demonstrating a different conservation combination.
4. **Exercise: free motion.** Set m=2,Ω=0,q0=(0,1),v0=(1,0). **Solution:** q=(t,1),p=(2,0),K=1J,L=I=−2. Force and work are zero; guiding center is undefined rather than set to the origin.
5. **Exercise: reverse orientation.** Replace Ω by−1 in exercise1. **Solution:** x=sin t,y=−(1−cos t),L=−(1−cos t). K remains1/2J. The modified invariant remains zero with the reversed Ω sign.
6. **Exercise: stationary state.** Take any initial q0, set v0=0 and choose a nonzero Ω. **Solution:** q stays at q0,F=0,K=0 and L=0. The modified invariant is the constant−mΩ|q0|²/2. A nonzero force coefficient does not imply a nonzero force when velocity vanishes.

## Connection to learned dynamical systems

A learned simulator can keep speed constant while predicting the wrong rotation frequency or direction. Energy-only losses cannot detect that error. Check vector momentum, external impulse, angular balance, the modified invariant and actual positions together. Conservation diagnostics must match the selected system and the declared force dependence; they do not replace validation of the force law against observations.

## Background and scope

For a physical perpendicular-force example, see [MIT OCW, Introduction to Magnetic Fields](https://ocw.mit.edu/courses/8-02t-electricity-and-magnetism-spring-2005/90510e513af27a2dde30b890a61bbcd1_ch8magneti_field.pdf). The signed Ω model, guiding center and modified invariant above are derived directly here. General symmetry theorems and complete electromagnetic conservation are developed separately.
