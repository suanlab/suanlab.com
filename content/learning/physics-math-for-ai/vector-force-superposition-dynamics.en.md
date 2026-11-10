## The net force changes velocity, not just position

A particle can keep moving while its forces balance. It can also have zero velocity at one instant while a nonzero force continues to accelerate it. These observations are compatible because Newton's second law relates net force to acceleration, the time derivative of velocity.

This lesson uses a constant positive point-particle mass in a Newtonian inertial laboratory. Two prescribed constant external forces act in a plane. We solve the resulting initial-value problem exactly and compare force addition, spatial motion, momentum balance and work. There is no contact, rigid-body rotation or changing-mass process hidden inside the example.

## Choose the particle and its external interactions

The selected system is one particle of mass m>0. Let the two external force vectors be F1 and F2. Their net force is

$$
\mathbf F_{\mathrm{net}}=\mathbf F_1+\mathbf F_2,
\qquad m\mathbf a=\mathbf F_{\mathrm{net}}.
$$

Every component has its own signed coordinate. Adding magnitudes without directions generally gives the wrong result. For example, a2 N rightward force and a2 N upward force produce the vector(2,2) N, not a4 N rightward force.

Both forces in this sum act on the selected particle. An action-reaction partner acting on a different body is not another term in this one-particle force balance. Equal and opposite forces on the same particle can balance, but that is a different statement from identifying a Newton-third-law pair across two bodies.

## Keep force geometry separate from position geometry

A force-plane drawing has horizontal and vertical axes measured in newtons. Arrows from its origin represent force components. A head-to-tail translated copy is a vector-addition construction; its apparent location is not the physical attachment point of another force.

The spatial trajectory uses meters on both axes. Its shape depends on the initial velocity and acceleration. We should not draw a force arrow on a position axis without declaring an additional scale that converts the arrow into a displacement illustration.

Maintaining equal scales within each two-dimensional plane also matters. Stretching one spatial axis changes apparent angles and curvature. The numerical model retains actual components; the plot should not quietly distort their geometry.

## Solve the constant-force initial-value problem

The illustration fixes initial position to the origin and allows an arbitrary initial velocity v0. Since the forces and mass are constant, acceleration is constant and integration gives

$$
\mathbf a=\frac{\mathbf F_{\mathrm{net}}}{m},\qquad
\mathbf v(t)=\mathbf v_0+\frac{\mathbf F_{\mathrm{net}}}{m}t,
\qquad
\mathbf q(t)=\mathbf v_0t+\frac{\mathbf F_{\mathrm{net}}}{2m}t^2.
$$

Velocity has units m/s and position has units m. Differentiate q to recover v, then differentiate v to recover a. These checks confirm that the displayed solution satisfies both the differential law and the declared initial data.

The law alone does not select a unique trajectory without initial conditions. Two particles under the same mass and forces can follow different paths if their initial velocities differ. A force-space diagram cannot replace the initial-value specification.

## Work through a genuinely two-dimensional example

Choose m=2 kg,F1=(2,0) N,F2=(0,2) N and v0=(1,0) m/s. The net force is(2,2) N, so acceleration is(1,1) m/s squared.

At t=2 s, velocity is(3,2) m/s and position is(4,2) m. The spatial path is curved because the initial velocity is not parallel to acceleration. The velocity evolves along a straight line in velocity space, while position contains a quadratic time term.

Initial kinetic energy is1 J and query kinetic energy is13 J. Force1 does8 J of work and force2 does4 J, totaling12 J. The calculation connects the vector motion to energy without replacing the force law with an unrelated scalar shortcut.

## Balanced forces allow uniform motion

Now take F1=(2,0) N,F2=(-2,0) N with the same mass and initial velocity. Net force and acceleration are zero, but the velocity remains(1,0) m/s. At2 s the particle has moved to(2,0) m.

It is in force balance, but it is not stationary. A stationary motion would additionally require initial velocity zero. A numerical flag for balanced forces should therefore remain separate from a flag for rest throughout the motion.

The forces can still do individual signed work: after that displacement, their works are4 J and-4 J. Total work is zero and kinetic energy stays constant. Zero net work does not require every individual work term to vanish.

## A turning instant is not a static equilibrium

Consider m=1 kg,F1=(-1,0) N,F2=0 and v0=(1,0) m/s. At1 s, velocity is zero and position is(0.5,0) m. Acceleration remains(-1,0) m/s squared.

Immediately afterward the velocity becomes negative in the x direction. The particle was momentarily at rest, not at a stationary equilibrium. Reading one zero velocity value cannot certify zero net force or a constant future position.

This distinction also applies at the initial instant. Starting at rest under a nonzero force does not make the force disappear. A model should report the instantaneous velocity and the entire-motion property separately.

## Verify the vector momentum balance

With constant mass, momentum is P=m*v. Subtracting its initial value gives

$$
\mathbf P(t)-\mathbf P_0=\mathbf F_{\mathrm{net}}t.
$$

Both vector components must satisfy the balance. At t=0 the momentum change is zero even if the force is nonzero; that single equality does not establish momentum conservation for all times.

The constant-mass assumption is explicit. For an open system gaining or losing material, momentum carried across the system boundary must be accounted for. Substituting a changing mass into this point-particle formula without specifying mass transfer is not an authorized extension of this model.

## Derive the kinetic-work identity

For kinetic energy K=m*v dot v/2, substitution of the velocity solution gives

$$
K(t)-K_0
=\mathbf F_{\mathrm{net}}\cdot\mathbf v_0t
+\frac{\|\mathbf F_{\mathrm{net}}\|^2}{2m}t^2
=\mathbf F_{\mathrm{net}}\cdot\mathbf q(t).
$$

Because the individual forces are constant,

$$
W_1+W_2=\mathbf F_1\cdot\mathbf q+\mathbf F_2\cdot\mathbf q
=K(t)-K_0.
$$

Work is signed. A force opposing displacement can contribute negative work. The turning example loses0.5 J of kinetic energy by the query time. Nothing in this calculation identifies every possible thermal or internal-energy transfer; kinetic energy is the quantity being balanced under the declared particle model.

## Understand the controls and exact checks

Mass uses kilograms directly. Force and initial-velocity codes are divided by10, and the time code is divided by10 seconds. The initial position is fixed to zero. All supported query values and balance flags use reduced rational arithmetic.

The sampled spatial path uses decimal coordinates for rendering. Its geometric appearance illustrates the analytic solution rather than proving a balance by sampled agreement. The query time moves a selected state along the declared solution; it does not change the constant-force law.

Changing mass while holding physical forces fixed changes acceleration. Changing initial velocity can change the whole spatial path while preserving the force diagram. These are different experimental changes and should be inspected one at a time.

## Implications for learned dynamics

A learned acceleration model should respond to vector superposition and mass in the stated way if that is its intended physical domain. Checking only one component can hide a missing force in another direction. Checking only velocity at a turning instant can also hide persistent acceleration.

An informative evaluation reports the selected system, force inputs, mass, initial data and observer frame. It distinguishes exact model identities from empirical correctness of the model in a real apparatus. A successful constant-force prediction does not by itself validate contact laws, rotations or changing-mass behavior outside the lesson's assumptions.

## Exercises with worked solutions

1. **Exercise:** Add(2,0) N and(0,2) N for a2 kg particle. **Solution:** Net force is(2,2) N and acceleration is(1,1) m/s squared.
2. **Exercise:** With v0=(1,0) m/s, find the state after2 s. **Solution:** Velocity is(3,2) m/s and position is(4,2) m.
3. **Exercise:** Do opposite forces require the particle to stop? **Solution:** No. Zero net force preserves the initial velocity, which can be nonzero.
4. **Exercise:** Is zero velocity at the turning instant a force-balance certificate? **Solution:** No. The example at1 s has zero velocity and acceleration(-1,0) m/s squared.
5. **Exercise:** What are the two works in the two-dimensional example? **Solution:** F1 dot q=8 J and F2 dot q=4 J, matching the12 J kinetic increase.
6. **Exercise:** Should a reaction force on another body be added to this particle's free-body sum? **Solution:** No. The sum contains forces acting on the selected particle.

## Sources and further study

[NASA Glenn's Newton's Second Law](https://www.grc.nasa.gov/WWW/K-12/BGP/newton2.html) gives background on constant-mass vector components. The initial-value solution, balanced/turning examples and exact momentum/work checks above are direct derivations for this declared two-force particle model.
