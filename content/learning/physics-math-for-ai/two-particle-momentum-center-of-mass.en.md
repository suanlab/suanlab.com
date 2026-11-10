## Two moving particles can have zero total momentum

Imagine two carts initially at rest, with an internal actuator pushing them in opposite directions. The center of mass can stay fixed while both carts acquire speed. Adding their momenta gives zero; adding their kinetic energies gives a positive number. Cancellation of signed momentum is different from cancellation of motion or energy.

This lesson replaces the actuator with a deliberately simple force law: a prescribed constant pair +f on particle1 and −f on particle2. Independent external forces E1 and E2 may also act. Both masses are constant and positive. We solve one-dimensional Newtonian motion in an inertial frame. The pair is a declared mathematical interaction, rather than an automatic spring, rope or contact model.

## Start with a separate law for each particle

Choose signed positions q1,q2 and velocities v1,v2. The force laws are

$$
m_1a_1=f+E_1,\qquad m_2a_2=-f+E_2.
$$

Only forces acting on the particle belong in its own equation. The partner force belongs on the other particle's equation. Constant accelerations give

$$
v_i(t)=v_{i0}+a_it,\qquad
q_i(t)=q_{i0}+v_{i0}t+\frac12a_it^2.
$$

Initial positions need not coincide, and initial velocities need not vanish. A negative interaction reverses the directions of the internal pair. The two particle paths may cross. Crossing is allowed for this point-particle model and does not trigger an impact, exchange of velocities or change in the prescribed forces.

## Add the equations to obtain total momentum balance

Define total momentum P=m1v1+m2v2. Adding the equations cancels f against −f:

$$
\dot P=E_1+E_2,\qquad
P(t)-P(0)=(E_1+E_2)t.
$$

The cancellation happens because both members of the interaction pair lie inside the selected two-particle system. It does not say that either particle has zero force. When the total external force is zero, total momentum remains its initial value. That value can be zero or nonzero. Conservation constrains the sum; individual momenta can still change by equal and opposite amounts.

Opposing external forces also deserve care. If E1=−E2, total momentum is conserved even though each external force can perform work. Zero net external force is a statement about a sum of forces, not a statement that the surroundings have no effect on relative motion or energy.

## Center of mass is a mass-weighted position

Let M=m1+m2 and define

$$
R=\frac{m_1q_1+m_2q_2}{M},\qquad
\dot R=\frac{P}{M},\qquad
\ddot R=\frac{E_1+E_2}{M}.
$$

The fixed masses allow differentiation without extra mass-transfer terms. The center behaves like a representative point carrying the total mass under the total external force. It is a derived coordinate, not a third particle with its own extra interaction.

For unequal masses, R lies closer to the heavier particle. The arithmetic midpoint is correct only for equal masses. A constant center position requires zero total external force and zero initial total momentum in this constant-force model. Zero center acceleration alone permits uniform center motion.

## Changing the boundary changes which force is external

The interactive selection code0 includes both particles. Code1 selects only particle1; code2 selects only particle2. Changing this code does not change physical motion. It changes the momentum balance being inspected:

$$
\dot P_{\{1\}}=f+E_1,\qquad
\dot P_{\{2\}}=-f+E_2.
$$

For a one-particle system the partner lies outside the boundary, so its force is external to that subsystem. Calling f an internal force once and then discarding it from every possible subsystem balance would be an error. Always name the system before classifying a force.

The exact table reports selected momentum, selected external force and selected impulse. The full-system center and total momentum remain visible as separate reference quantities. In particular, total momentum may be conserved while selected one-particle momentum is not.

## Relative motion stores an independent kinetic contribution

Let relative velocity be v1−v2 and reduced mass μ=m1m2/M. Expanding the squares gives

$$
K=\frac12m_1v_1^2+\frac12m_2v_2^2
=\frac{P^2}{2M}+\frac12\mu(v_1-v_2)^2.
$$

To see the algebra, write v1=P/M+(m2/M)(v1−v2) and v2=P/M−(m1/M)(v1−v2). The cross terms cancel after mass weighting. The first contribution describes center motion; the second describes relative motion. Both are nonnegative for positive masses.

If P=0, only the first contribution vanishes. The particles may still have relative speed and positive kinetic energy. Consequently, momentum conservation alone cannot certify kinetic-energy conservation, a collision type or the absence of an internal energy exchange.

## Internal forces can do nonzero total work

Let Δqi=qi(t)−qi(0). Since all prescribed forces are constant,

$$
W_{\mathrm{int}}=f(\Delta q_1-\Delta q_2),\qquad
W_{\mathrm{ext}}=E_1\Delta q_1+E_2\Delta q_2,
$$

$$
K(t)-K(0)=W_{\mathrm{int}}+W_{\mathrm{ext}}.
$$

Equal and opposite internal forces do not imply zero internal work: they act through generally different displacements. Adding forces is sufficient for momentum balance; calculating work additionally needs the motion at each point of application. This is why a fixed center can coexist with increasing kinetic energy.

## Declare the potential before claiming mechanical conservation

For this constant pair, choose the linear potential

$$
V(q_1,q_2)=-f(q_1-q_2).
$$

Its negative gradients give +f and −f. Thus ΔV=−Wint and

$$
\Delta(K+V)=W_{\mathrm{ext}}.
$$

With zero external work, the declared mechanical energy is conserved even if K changes. A different potential zero adds a constant without changing this balance. The linear potential is unbounded over unrestricted positions; it is a mathematical example, not a universal material law. An actuator with a finite reservoir or a spring with a displacement-dependent force would need additional state or another interaction law.

## Interpret the plots and boundary readouts

The position panel shows q1 in blue, q2 in orange and R in green against time. These curves share meters on the vertical axis. The momentum panel compares the actual total momentum with P0+(E1+E2)t in kg m/s. Agreement is expected because the same declared dynamics obey the balance; overlapping curves are not evidence from an independent experiment.

Use the query control for exact current values, and then change just the boundary code. Positions, velocities and total momentum should stay fixed while selected force and selected momentum may change. Reverse the internal pair, try unequal masses, and compare an opposing external pair with both external forces zero. These changes distinguish aggregate motion from relative dynamics.

## Six exercises with worked solutions

1. **Exercise: internal motion from rest.** Set equal1kg masses at−1m and1m, initial velocities zero, f=1N and E1=E2=0. At1s? **Solution:** accelerations are1 and−1m/s²; positions−1/2 and1/2m; velocities1 and−1m/s. P=0 and R=0, while K=1J.
2. **Exercise: energy exchange.** Find potential and mechanical energy for the same motion. **Solution:** initial V=2J and final V=1J. K rises from0 to1J. Therefore K+V remains2J, and internal work is1J.
3. **Exercise: add external force.** Set E1=1N in that example. **Solution:** a1=2 and a2=−1m/s². At1s, q1=0,q2=1/2m and P=1kg m/s. R=1/4m; its acceleration is1/2m/s².
4. **Exercise: external work.** Evaluate energy for exercise3. **Solution:** K=5/2J and V=1/2J, giving total3J instead of initial2J. E1 moves its particle by1m, so external work is1J, matching the increase.
5. **Exercise: select one particle.** Return to exercise1 and select particle1. **Solution:** selected momentum changes from0 to1kg m/s. The partner force is now external, supplying1N s. Total two-particle momentum is still0; there is no contradiction.
6. **Exercise: unequal masses.** Use m1=1kg,m2=2kg, the initial positions of exercise1 and f=1N. **Solution:** at1s, velocities1 and−1/2m/s give P=0. Positions−1/2 and3/4m give R=1/3m, equal to its initial value. K=3/4J and the arithmetic midpoint1/8m is not the center of mass.

## What a learned simulator should preserve

A network can predict the correct center trajectory while producing incorrect relative motion. Checking only total momentum will miss such errors. Conversely, a small kinetic residual does not identify a physically correct force law. Evaluate particle trajectories, the selected-system balance, center dynamics and the declared energy model together, with their assumptions recorded. Additional observations are needed to decide whether this constant-pair model fits an actual apparatus.

## Background and limits

The standard center-of-mass framework is presented in [MIT OpenCourseWare, Velocity and Acceleration of the Center of Mass](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/pages/week-5-momentum-and-impulse/17-6-velocity-and-acceleration-of-the-center-of-mass/). The constant-pair example and all numerical solutions here follow from the stated equations. Rigid-body rotation, variable mass and collision mechanics require further assumptions and are developed separately.
