## A stopping instant changes which question must be solved

A block sliding left can stop under a rightward drive and then either remain stuck or restart to the right. The same zero velocity occurs in both cases. To predict what happens next, compare the applied force with the static-friction limit rather than blindly continuing the old kinetic-friction direction.

This lesson uses one constant positive mass on a fixed horizontal surface. Initial position is zero, initial velocity is signed, and the horizontal drive F is constant. Vertical equilibrium gives N=mg with the declared teaching value g=10m/s². That normal-load formula is specific to this horizontal model; it is not a universal rule for inclined or vertically accelerated systems.

## State the static and kinetic laws separately

The kinetic coefficient is μk≥0. A nonnegative margin δ sets μs=μk+δ, so every available control combination satisfies μs≥μk. During nonzero-speed sliding,

$$
f_k=-\mu_kN\operatorname{sign}(v).
$$

At rest, static friction can adjust within its limit. We choose sticking when

$$
|F|\le\mu_sN,\qquad f_s=-F.
$$

The inequality matters: static friction is not always equal to its maximum. At the threshold equality this experiment explicitly selects sticking. If the drive exceeds the static limit, release begins in the drive direction and the kinetic branch supplies the right-hand acceleration.

These are declared Coulomb-model rules. They do not describe every microscopic surface or resolve temperature and speed dependence.

## Integrate a sliding segment before looking for events

While velocity retains a sign σ, acceleration is constant:

$$
a=\frac{F-\mu_kN\sigma}{m},\qquad
v(t)=v_0+at,\qquad x(t)=v_0t+\frac12at^2.
$$

If acceleration opposes the initial velocity, the first stopping time is

$$
t_s=-\frac{v_0}{a}>0.
$$

If acceleration is zero or points along velocity, there is no future stop in that segment. Zero acceleration permits nonzero uniform sliding. The static law must not be applied just because a moving block's drive happens to lie below the static threshold.

An initially resting block uses the static/release rule immediately; it does not need an invented earlier sliding segment.

## Restart the model at the exact stopping event

Evaluate the stop position xs from the first segment and set velocity to zero there. Recheck |F| against μsN. If sticking is allowed, x stays at xs and friction adjusts to−F. Otherwise motion restarts in sign(F), with

$$
a_+=\frac{F-\mu_kN\operatorname{sign}(F)}{m},\qquad
x=x_s+\frac12a_+(t-t_s)^2.
$$

The force value at the event uses this post-event branch. Position and velocity remain continuous. Friction and acceleration can have finite jumps, which do not insert an instantaneous finite momentum change.

The force plot includes the event time itself even if it lies between the ordinary samples. Vertical segments show one-sided limits, not a time interval traversing all intermediate force values. Events beyond ten seconds remain in the exact readout but are not inserted outside the plotted horizon.

## Zero velocity need not mean sticking

When an excessive drive reverses initially opposite motion, velocity is zero at the turning instant but post-event acceleration is nonzero. The interface labels this as incipient sliding. A sticking state instead has zero velocity and zero acceleration under a balancing static force.

A different distinction appears under force-balanced sliding. Suppose F equals kinetic friction and initial velocity is positive. The block continues uniformly. Start at rest under the same drive, and static friction may instead hold it fixed. Initial state and force history both matter.

## Heat uses distance while work uses displacement

Let D(t) be the total distance traveled, including both sides of any reversal. The mechanical dissipation quantity is

$$
Q(t)=\mu_kN D(t)\ge0,\qquad
W_{\mathrm{drive}}=F x(t).
$$

Kinetic friction opposes each segment's motion, so its work is−Q. Static friction during sticking does no work because displacement is zero. The energy balance is

$$
K(t)-K(0)+Q(t)=W_{\mathrm{drive}}(t).
$$

Replacing distance by signed net displacement would make reversed travel disappear from the heat calculation. Reversing direction does not reverse already generated heat. Q here is cumulative mechanical dissipation, not a modeled temperature or a complete thermal state.

The independent reference calculation integrates friction force times velocity over each constant-force segment. It checks the distance-based heat without defining heat as a missing energy residual.

## Momentum balance checks the force switch too

Integrate the drive and piecewise friction:

$$
m[v(t)-v_0]=Ft+\int_0^t f(s)\,ds.
$$

The event-time force value at a single instant changes neither impulse nor work. A correct implementation must nevertheless switch the subsequent force direction; extending pre-stop friction into reverse motion would accelerate in the wrong way and violate dissipation.

The exact table reports the momentum residual, friction-work/heat relation and total work/heat residual separately. Their agreement is a mathematical check of the declared model, not evidence that these coefficients fit an actual surface.

## Inspect the four time panels

Position is in meters, velocity in m/s, forces in newtons and energy in joules. The force panel compares blue friction with orange drive. The energy panel compares blue K, orange Q and green K0+Wdrive; K+Q should match that green budget.

Query time changes the displayed exact state while curves retain a ten-second horizon. Coefficients and other nonmass inputs are integer codes divided by ten. Stop time and stop position can be rational numbers between query-grid values. The graph samples those events explicitly, so the event is not rounded to the next simulation step.

## Six exercises with worked solutions

1. **Exercise: stop and stick.** Set m=1,μk=0.2,μs=0.4,F=3,v0=−1. **Solution:** N=10,kinetic friction initially+2N and acceleration5m/s². Stop occurs at1/5s with x=−1/10m. Since3≤4, static friction−3N holds the block afterward.
2. **Exercise: energy at that stop.** Evaluate the budget. **Solution:** initial K=1/2J, final K=0, distance1/10m gives Q=1/5J. Drive work is−3/10J. Indeed−1/2+1/5=−3/10.
3. **Exercise: stop and reverse.** Change only F to5N and query1s. **Solution:** stop time is1/7s. Post-stop acceleration is3m/s². Then x=101/98m and v=18/7m/s; total distance is115/98m. Heat is115/49J and drive work505/98J.
4. **Exercise: same drive, different initial state.** Set F=2N with the original coefficients. **Solution:** from v0=1m/s, kinetic friction−2N gives uniform sliding, x(1)=1m and Q=W=2J. From v0=0, static friction−2N gives rest and both accumulated quantities are zero.
5. **Exercise: threshold equality.** Start at rest with F=4N. **Solution:** the declared equality convention selects sticking with friction−4N. Increase F to4.1N and release occurs with acceleration(4.1−2)/1=2.1m/s². The static margin is a threshold, not an extra constant drag.
6. **Exercise: a later event.** Use m=4,μk=μs=0.1,F=3.9,v0=2. **Solution:** kinetic friction is−4N, acceleration−1/40m/s² and stop time80s. At10s velocity is7/4m/s, so the block is still sliding although its eventual stop is known.

## Why event handling matters for learned simulators

A fixed-step sign switch can overshoot zero velocity and chatter between friction directions. Exact event logic provides a useful reference for judging such methods. Energy-only fitting also misses the distinction between net displacement and distance. Evaluate event time, branch selection, momentum and work/heat together, and record which static-threshold convention was chosen.

## Background and limits

For the static/kinetic distinction, see [MIT OCW, Applications of Newton's Second Law, Section8.3](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/mit8_01scs22_chapter8.pdf). The piecewise constant-drive solution and exact examples above are derived here. Compliance, temperature, changing normal loads and contact release need additional models.
