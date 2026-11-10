## Rolling and sliding require different contact laws

A wheel can translate, spin and slip at the floor at the same time. Watching its center move does not reveal whether its contact is sliding. Static friction can point backward under a center-applied force and forward under an applied axle torque. Its direction must come from the contact constraint and equations of motion.

This lesson solves actual continuous motion of a rigid axisymmetric wheel on a fixed horizontal floor. Unlike the previous instantaneous rod impact, it evolves position, angle and velocity through a finite time interval. The center stays at height R; there is no vertical bounce or normal impact in this model.

## Declare the geometry, inertia and signs

Choose positive mass m and radius R, gravitational acceleration g=10m/s², and normal load N=mg. A constant horizontal force F acts at the center. A constant signed torque τ acts about the center.

We use clockwise angular velocity Ω and angle φ as positive. This convention is the opposite of the counterclockwise ω in the rod-impact lesson. Positive linear velocity V points right. With these signs, bottom contact slip is

$$
u=V-R\Omega.
$$

The inertia is I=κmR² with0<κ≤1, a realizable axisymmetric mass family within outer radius R. A uniform disk has κ=1/2; a thin hoop has κ=1. The floor is stationary, and the wheel remains circular and rigid. We omit rolling resistance, tire deformation, temperature evolution and lift-off.

## Derive translation and rotation together

Let f be the signed contact friction force, positive right. Then

$$
m\dot V=F+f, \qquad I\dot\Omega=\tau-Rf.
$$

A rightward bottom force creates a counterclockwise torque, explaining its minus sign in the clockwise equation. Differentiating slip gives

$$
\dot u=A_0+Cf, \qquad
A_0=\frac Fm-\frac{R\tau}{I}, \qquad
C=\frac1m+\frac{R^2}{I}>0.
$$

Both translation and rotation determine the relative contact acceleration. Reusing a point-particle friction formula with only F/m would miss the spin response and predict the wrong event time.

## Determine whether pure rolling is feasible

At u=0, persistent no-slip motion requires u̇=0. The required static friction is

$$
f_{\rm req}=-\frac{A_0}{C}
=\frac{R\tau/I-F/m}{1/m+R^2/I}.
$$

If |freq|≤μsN, select rolling with f=freq. Equality belongs to the rolling branch by the declared convention. Rolling means V=RΩ, not that V and Ω are zero. The wheel can accelerate from rest while satisfying this constraint.

The required friction may be positive, negative or zero. A nonzero force and torque can also produce compatible accelerations without friction. The normal force stays mg because the circular wheel's center has no vertical acceleration and the applied force and torque do not add a vertical force.

## Use kinetic friction when the contact slides

For u≠0, kinetic friction is

$$
f=-\mu_kN\operatorname{sgn}(u).
$$

It opposes relative slip at the floor. It need not oppose the direction of center motion. The controls define μs=μk+δ with δ≥0, so every coefficient combination respects the static-versus-kinetic ordering.

At u=0 when the required static force exceeds its limit, slip starts in the direction of A0. Kinetic friction then opposes that incipient slip. The excess threshold and μs≥μk ensure the new branch accelerates away from zero with the declared sign. A zero instantaneous slip value can therefore be either rolling or the exact onset of sliding; the regime flag distinguishes them.

## Locate the rolling or reversal event exactly

Within a fixed-friction branch, V and Ω are affine in time. Slip is also affine. If u0 and its branch derivative have opposite signs, the zero-slip event occurs at

$$
t_s=-\frac{u_0}{\dot u}.
$$

Integrate position and angle with the same constant accelerations. At the event, test static feasibility again. The wheel either joins pure rolling or begins sliding in the opposite relative direction. Under the constant drives and coefficient ordering used here, no additional zero-slip switching event follows this decision.

Position, angle, V and Ω stay continuous. Friction may jump finitely. A vertical segment on the friction-versus-time plot represents its two one-sided limits, not a finite impulse. An exact event-time query uses the post-event branch. Events beyond the displayed interval remain reported in the table without being inserted into the visible curves.

## Separate contact work from dissipated energy

The translational and rotational powers of friction are fV and −RfΩ. Their sum is

$$
P_{\rm contact}=f(V-R\Omega)=fu.
$$

For pure rolling this total is zero, even when friction is nonzero and its two individual work contributions are nonzero. For sliding, fu=−μkN|u|. The accumulated mechanical dissipation is therefore

$$
Q=\mu_kN\int_0^t|u(s)|\, ds.
$$

This is relative sliding distance at contact, not the center's travel and not always the absolute net difference |x−Rφ|. A reversal requires adding both traveled slip branches. The model calls Q contact heat or dissipated energy; it does not assign temperatures or decide how the two materials share that energy.

Zero kinetic friction gives Q=0 even when the contact slips. Conversely a moving wheel can roll with Q=0. Motion, slip and dissipation are related but distinct quantities.

## Close the full force-and-torque energy budget

The wheel's kinetic energy and external work are

$$
K=\frac12mV^2+\frac12I\Omega^2, \qquad
W=F\Delta x+\tau\Delta\phi.
$$

Multiplying the equations of motion by V and Ω yields

$$
\Delta K+Q=W.
$$

The calculation also integrates friction's translational and rotational works separately and verifies their sum is −Q. This exposes sign errors that a total-energy number alone might conceal. Linear impulse is Ft+∫fdt, while clockwise angular impulse is τt−R∫fdt.

The angle φ is unwrapped. Complete revolutions contribute to torque work. A circular drawing can use sinφ and cosφ for the current marked point, but reducing φ modulo2π before computing τφ would lose physical work.

## Follow an initially sliding disk into rolling

Set m=R=1, κ=1/2, μk=0.2, μs=0.4, F=τ=0, V0=1 and Ω0=0. Initially f=−2, V̇=−2 and Ω̇=4. Thus u̇=−6 and ts=1/6s.

At the event, V=Ω=2/3, x=5/36 and φ=1/18. Relative slip distance is x−Rφ=1/12, so Q=1/6. Thereafter friction becomes0 and the wheel rolls uniformly. At t=1, x=25/36 and φ=11/18, but Q remains1/6.

Using total center distance25/36 in the heat formula would produce an impossible loss exceeding the initial kinetic energy. The correct final energy is1/3, compared with initial1/2. This example makes the choice of contact-relative distance observable.

## Read the geometry and time curves

A wheel center, a geometric bottom contact and a marked material point are different objects. The geometric contact is at(x, 0). A material mark initially on the right rim is at(x+Rcosφ, R−Rsinφ). As the wheel spins, successive material points reach the bottom. The geometric contact marker is not one permanently tracked piece of material.

Time histories span0 to10s and explicitly include the query and any event in that interval. Linear speed V and rim speed RΩ share m/s and can be compared directly; raw Ω has rad/s. Friction is in N, angle in rad and energy in J. Signed angles and positions stay distinct from traveled distances.

The control values, piecewise states, switching times and work budgets use exact rational arithmetic. Trigonometric drawing coordinates are numerical. A continuous-motion experiment at fixed height does not supply the normal impact map from the previous lesson.

## Worked examples and exercises with solutions

1. **A center-driven rolling disk.** Use m=R=1, κ=1/2, μk=0.2, μs=0.4, F=3, τ=0 and start at rest. **Solution:** freq=−1, V̇=Ω̇=2. At1s, x=φ=1, V=Ω=2 and K=3=W. Static friction does−1J translational work and+1J rotational work, with zero net contact work and heat.

2. **The free sliding disk.** Use the same mass and coefficients with F=τ=0, V0=1, Ω0=0. **Solution:** ts=1/6s and the final rolling speed is2/3. At1s, Q=1/6, S=1/12 and K=1/3. The center's distance25/36 is not the contact slip distance.

3. **Torque drives the center through static friction.** Start at rest with F=0, τ=3 and the first example's coefficients. **Solution:** freq=+2, V̇=Ω̇=2. At1s, K=W=3. Friction does+2J translational work and−2J rotational work. Its forward direction is compatible with static rolling.

4. **A weak contact releases.** Keep F=0, τ=3 but use μk=μs=0.1. **Solution:** the required2N exceeds the1N static limit. Kinetic f=+1 gives V̇=1, Ω̇=4, u̇=−3. At1s, x=1/2, φ=2, K=9/2, Q=3/2 and W=6. By contrast, the center-force case F=3, τ=0 lies exactly at the static threshold with these coefficients and rolls.

5. **A relative-slip reversal.** Set m=R=1, κ=1/2, μk=μs=0.2, F=10, τ=0, V0=−1, Ω0=0. **Solution:** the contact reaches zero slip at1/16s, then reverses because static friction is insufficient. At1s, S=229/128 while x−Rφ=221/128; heat is229/64. Taking the absolute net slip would omit the first traveled branch.

6. **Keep events and revolutions outside a picture's limits.** A hoop with m=4, R=1, μk=μs=0.1, F=7.9, τ=0, V0=2, Ω0=0 has a rolling event at80s. **Solution:** it is still sliding at the10s plot limit, with u=7/4 and Q=75. Separately, a frictionless wheel with m=1, R=1/2, κ=0.1, τ=10, F=0 and zero initial speeds reaches φ=20000rad at10s. Torque work is200000J; wrapping the drawing phase must not erase those turns.

## Research connection and reading

A learned contact model should be tested on zero-slip acceleration, forward and backward friction, event-time equality, reversals and the difference between center travel and relative slip. Agreement on a wheel's outline alone cannot establish the correct force, torque, dissipation or branch selection. Exact budgets and independently integrated branch trajectories address different parts of that claim.

For background on rolling dynamics and energy, see [MIT OpenCourseWare's rolling lessons](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/pages/week-12-rotations-and-translation-rolling/). The constant force/clockwise-torque family, static-threshold convention and exact switching model above are declared specifically for this experiment. Frictional impact, multiple contacts and deformable tires require additional models and remain further study.
