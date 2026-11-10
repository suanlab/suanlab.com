## Three conserved totals can coexist with complicated motion

Two interacting particles can accelerate, exchange momentum and trade kinetic energy for potential energy while their total momentum, angular momentum and energy remain constant. A conservation law constrains a total. It does not freeze each component or determine the current phase of the motion.

This lesson selects both particles as the system. Their masses are positive constants, no external force acts, and a declared central harmonic interaction couples them. We solve their actual planar motion and inspect three distinct conservation mechanisms. The particles are mathematical points; crossing paths do not trigger an undeclared contact or collision rule.

## Specify the system and count the interaction once

Let z=q1−q2 and let stiffness k>0. The forces and pair potential are

$$
F_1=-kz,\qquad F_2=+kz,\qquad V=\frac{k}{2}|z|^2.
$$

The potential belongs to the pair and is counted once, not once per particle. Its minimum is at zero separation. This is a declared zero-rest-separation model, not a rod or a finite-rest-length spring between solid bodies.

Selecting only particle1 would put particle2 outside the system boundary and make F1 external to that subsystem. The closed-system statements below apply to the selected two-particle system, while individual momenta generally change.

## Force cancellation conserves total linear momentum

Add the two Newton equations:

$$
P=m_1v_1+m_2v_2,\qquad \dot P=F_1+F_2=0.
$$

The equality-and-opposition of the pair is sufficient for this total force cancellation. It does not require either force to vanish. A nonzero initial P persists, and zero initial P allows equal-and-opposite momentum exchange.

In the momentum panel the total x component is compared with both individual x components. Both y-component balances are also checked numerically in the readout. A constant blue total and varying individual curves are compatible, rather than contradictory.

## Central direction also cancels internal torque

About a fixed laboratory origin,

$$
L=q_1\times m_1v_1+q_2\times m_2v_2,
$$

$$
\dot L=q_1\times F_1+q_2\times F_2
=(q_1-q_2)\times F_1=z\times(-kz)=0.
$$

Planar cross products denote their signed z components. Equal and opposite forces alone would leave the term z×F1; its vanishing here also uses central direction. Internal torque is therefore a separate calculation from total force.

The conserved angular momentum can be nonzero. Internal rotation does not require external torque to sustain its already specified angular momentum. The interaction changes individual velocities while preserving the aggregate.

## A potential-work relation conserves total energy

Define kinetic energy as the sum over both particles. Their total kinetic power is

$$
\dot K=F_1\cdot v_1+F_2\cdot v_2=-kz\cdot(v_1-v_2),
$$

$$
\dot V=kz\cdot\dot z=kz\cdot(v_1-v_2).
$$

Thus E=K+V is constant. The internal forces can do nonzero total work because the two particles have different displacements. Their equal-and-opposite vector sum does not cancel their work automatically.

Central direction by itself does not establish energy conservation for every possible interaction. Here we additionally supplied a time-independent potential with the correct force gradients. A velocity-dependent dissipative central interaction would require another energy budget.

## Separate center motion from relative motion

Let M=m1+m2 and define the reduced mass μ=m1m2/M. The center coordinate and relative coordinate satisfy

$$
R=\frac{m_1q_1+m_2q_2}{M},\qquad
\ddot R=0,\qquad \mu\ddot z=-kz.
$$

The center moves uniformly with velocity P/M. The relative oscillator has frequency ω=√(k/μ). Starting from the actual particle initial data, compute R0,z0 and w0=v10−v20. Then

$$
R(t)=R_0+\frac{P}{M}t,\qquad
z(t)=z_0\cos(\omega t)+\frac{w_0}{\omega}\sin(\omega t).
$$

Reconstruct the two particles using q1=R+(m2/M)z and q2=R−(m1/M)z; differentiate for velocities. Positive stiffness is part of this experiment. No division by a zero oscillator frequency is used.

## Decompose kinetic and angular quantities

Substitution and cancellation of cross terms give

$$
K=\frac{|P|^2}{2M}+\frac{\mu}{2}|\dot z|^2,
$$

$$
L=R\times P+\mu z\times\dot z.
$$

The first terms describe center motion; the second terms describe relative motion. Zero total momentum removes center kinetic energy but leaves internal kinetic energy. A stationary center does not imply stationary particles.

For this closed pair, center angular momentum is constant because R changes parallel to P. Relative angular momentum is constant because the relative force is central. The interface checks the decompositions as well as the totals; these are useful diagnostics against a wrong mass weighting.

## Change a fixed reference point without changing the system

For another fixed point D=d,

$$
L_D=L-d\times P.
$$

Both L and P are constant, so LD is constant too, although its value can differ. If total P=0, every fixed translated point reports the same total L. This is distinct from selecting one particle as a different subsystem, which changes the forces classified as external.

The comparison-point controls affect angular reporting only. Particle paths, forces and stored energy remain unchanged.

## Read the four panels with their own units

The spatial panel shows both actual paths and center motion in meters over ten seconds. Query dots match the particle/center line colors. The momentum panel uses kg m/s, the angular panel kg m²/s and the energy panel joules. Different units are not superimposed on a shared numerical axis.

Initial P,L,E are exact rational calculations from the integer control codes. Trigonometric trajectories and query residuals are floating-point evaluations of the analytic solution. Independent numerical integration checks both particle motion and accumulated internal work; heat or work is not invented from a residual to make a test pass.

## Six exercises with worked solutions

1. **Exercise: internal oscillation.** Set equal1kg masses at(−1,0),(1,0), both initially at rest, with k=1. **Solution:** ω=√2 and q1x=−cos(√2t),q2x=cos(√2t). Total P=L=0 but particles move. K=2sin²(√2t),V=2cos²(√2t),so E=2J.
2. **Exercise: avoid double counting.** What is the initial potential in exercise1? **Solution:** separation has squared length4, so V=4/2=2J. Counting that same pair twice would incorrectly give4J.
3. **Exercise: internal rotation.** Give particle1 velocity(0,−1) and particle2(0,1) in exercise1. **Solution:** each initial angular contribution is1, so L=2. Initial K=1J and V=2J give E=3J. The relative motion is elliptical and does not require an external torque.
4. **Exercise: drifting unequal masses.** Use m1=1,m2=2, the same initial positions, v1=(1,1),v2=(1,−1/2). **Solution:** P=(3,0),R=(1/3+t,0),L=−2 and E=17/4J. Center kinetic energy is3/2J; the remaining11/4J is internal kinetic plus pair potential.
5. **Exercise: translated angular reference.** In exercise4 choose D=(0,1). **Solution:** d×P=−3,so LD=−2−(−3)=1. Both angular quantities are conserved although their values differ. Particle trajectories remain identical.
6. **Exercise: a thought variation beyond the controls.** If k were externally changed with time, would all three totals still be conserved? **Solution:** equal-and-opposite central forces still cancel total force and torque. But dE/dt=(dk/dt)|z|²/2,so energy generally changes through the external modulation. The current widget keeps k constant.

## Connection to learning physical systems

Conservation constraints do not determine the force law or trajectory phase. A learned simulator may preserve total P while getting internal oscillation wrong, or preserve energy while using incorrect mass weighting. Compare trajectories, force directions, decompositions and independent work integrals together. The algebra proves identities for the declared model; measurements are needed to determine whether the model fits a physical apparatus.

## Background and scope

For system-level conservation assumptions, see [MIT OCW, Conservation Laws for Systems of Particles](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/5b68bc0342763cf099422af5a075940a_MIT16_07F09_Lec11.pdf). The harmonic pair and worked values above are derived here. Distributed fields, rigid-body spin, contact and dissipation require additional state or interactions.
