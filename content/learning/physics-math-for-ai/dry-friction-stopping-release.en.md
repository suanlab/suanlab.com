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

## Research extension: identify friction with an experiment that changes direction

Suppose mass and normal load are known, but the drive is not independently calibrated and only position is measured. Can an arbitrarily accurate trajectory determine the kinetic coefficient? Not always. We construct three distinct, feasible parameter settings with the same entire position history, then choose a second experiment that distinguishes them.

The native lab is a forward model with known drive and coefficients. The inverse problem here deliberately hides some of those inputs from an observer; it is not an ambiguity in the evaluator. Keep the stated horizontal model, N=mg and g=10 m/s², the same static threshold convention and constant parameters within each run.

### See the missing force direction in a positive sliding experiment

Write C for kinetic-friction magnitude and S for the static limit. Define a_pos as acceleration while v>0; this differs from the post-release acceleration a_plus used earlier. A positive sliding segment provides only one linear combination of two unknown forces:

$$
C=\mu_kN,\quad S=\mu_sN,\qquad
m a_{\mathrm{pos}}=F-C
=\begin{pmatrix}1&-1\end{pmatrix}\begin{pmatrix}F\\C\end{pmatrix},\qquad
(F,C)\longmapsto(F+\eta,C+\eta).
$$

The observation row has rank one. Its null direction is (1,1): equal changes to drive and friction leave the measured acceleration unchanged, as long as the new parameters remain feasible. These are force changes in newtons, not arbitrary changes to two dimensionless coefficients.

Choose m=1 kg,N=10 N,S=3 N. The native control codes below use 0.1 units for every nonmass input; margin is the code for mu_s−mu_k. The time code 10 selects 1 s. The selected model permits mu_s=mu_k, including candidate C.

| case | F (N) | C (N) | mu_k | mu_s | codes (mass,kinetic,margin,drive,velocity,time) |
|---|---:|---:|---:|---:|---|
| A | 0 | 1 | 0.1 | 0.3 | (1,1,2,0,10,10) |
| B | 1 | 2 | 0.2 | 0.3 | (1,2,1,10,10,10) |
| C | 2 | 3 | 0.3 | 0.3 | (1,3,0,20,10,10) |

### Prove equality of the full stopped trajectory

All three settings start with v0=1 m/s and have a_pos=−1 m/s². Until the first stop they therefore have the same solution. Each stops at t_s=1 s and x=1/2 m. Since every drive lies below S, each then sticks. With positive deceleration magnitude A, the complete history for t≥0 is:

$$
u(t)=\min(t,t_s),\quad t_s=1\ \mathrm{s},\quad
x(t)=v_0u(t)-\frac A2u(t)^2,\quad v(t)=v_0-Au(t),\qquad
v_0=1\ \mathrm{m/s},\quad A=1\ \mathrm{m/s^2}.
$$

For t≤t_s this is the usual constant-acceleration expression. Thereafter u stays fixed, so position is 1/2 m and velocity zero. Infinitely dense, noiseless position data from this experiment still cannot distinguish the three settings. This is structural nonidentifiability, rather than an error caused only by sparse sampling.

### Keep the different work and heat budgets visible

The same trajectory also gives the same kinetic energy history. It does not give the same heat or drive work:

$$
Q=CD,\quad W=Fx,\quad \Delta K+Q=W,\qquad
x(t_s)=D(t_s)=\tfrac12\ \mathrm{m},\quad
Q(t_s)=C\left(\tfrac12\ \mathrm{m}\right),\quad
W(t_s)=F\left(\tfrac12\ \mathrm{m}\right).
$$

At the common stop, initial kinetic energy is 1/2 J and final kinetic energy is zero:

| case | dissipated heat Q (J) | drive work W (J) | Delta K+Q−W (J) |
|---|---:|---:|---:|
| A | 1/2 | 0 | 0 |
| B | 1 | 1/2 | 0 |
| C | 3/2 | 1 | 0 |

![Three candidate parameter settings share one forward position curve but have different dissipated heat and drive work at the common stop.](/assets/learning/classical-friction-identification.en.svg)

The upper panel draws their one shared position curve in metres, not three distinguishable trajectories. The lower panel separates heat and work in joules. Matching an energy residual supplies their difference; it does not independently measure either quantity. If F is calibrated separately, C=F−m a_pos becomes identifiable from this sliding segment. State what is observed before claiming an identification result.

### Prepare an opposite-direction experiment

Keep the same mass, drive, coefficients and normal load, but start with v0=−1 m/s. Friction now points right. Define a_neg during this nonzero negative-velocity segment:

$$
m a_{\mathrm{neg}}=F+C,\qquad
F=\frac m2(a_{\mathrm{pos}}+a_{\mathrm{neg}}),\quad
C=\frac m2(a_{\mathrm{neg}}-a_{\mathrm{pos}}),\quad
\mu_k=\frac C{mg}.
$$

The acceleration pairs (a_pos,a_neg) are (−1,1),(−1,3),(−1,5) m/s². The second experiment stops at 1,1/3,1/5 s and positions −1/2,−1/6,−1/10 m. These states distinguish the candidates. Use sliding accelerations, not the zero acceleration after sticking or a derivative spanning the force jump. In the native lab change only the initial-velocity code in each table row from 10 to −10 for this probe.

### Quantify deterministic acceleration uncertainty

Identifiability does not make noisy differentiation harmless. Suppose mass and normal load are known exactly and the two sliding-acceleration estimates have certified error magnitudes. Subtract the true two-direction inverse formulas from the estimated ones and apply the triangle inequality:

$$
|\widehat a_{\mathrm{pos}}-a_{\mathrm{pos}}|\le\varepsilon_{\mathrm{pos}},\quad
|\widehat a_{\mathrm{neg}}-a_{\mathrm{neg}}|\le\varepsilon_{\mathrm{neg}},\qquad
\max(|\widehat F-F|,|\widehat C-C|)\le\beta
=\frac m2(\varepsilon_{\mathrm{pos}}+\varepsilon_{\mathrm{neg}}),\quad
|\widehat\mu_k-\mu_k|\le\frac\beta N.
$$

With m=1 kg,epsilon_pos=0.1 m/s² and epsilon_neg=0.2 m/s², beta=3/20 N and the coefficient bound is 3/200. Same-sign endpoint errors attain the drive bound; opposite-sign errors attain the friction bound. These are deterministic bounds under stated assumptions, not confidence intervals. A position-noise tolerance alone does not certify an acceleration-error tolerance; that requires an estimator analysis. Changing mass, load or surface conditions between the two experiments also changes the inverse problem.

Even with F and C identified, the observed sticking and the model condition mu_s≥mu_k give only a lower bound on the static limit:

$$
S\ge\max(|F|,C),\qquad
\mu_s\ge\max\!\left(\frac{|F|}{mg},\mu_k\right).
$$

Additional calibration or release-threshold data is needed to determine mu_s itself. Prior bounds on the control space should be distinguished from information supplied by the trajectory.

### Reproduce the branch formulas and sharp error bounds

The code implements only the declared three-candidate family, in numerical SI units, with stop-and-stick in both directions. It checks six times per direction and candidate, the work budget and the two-direction inverse. It then checks all four acceleration-error corners for each drive using exact fractions. The general arguments above establish full-history equality and the error bound; finite checks verify these examples.

```python
from fractions import Fraction as F

def case(drive, direction, time):
    C = drive+1
    assert drive in (0, 1, 2) and direction in (-1, 1)
    a = drive-direction*C
    stop = -F(direction)/a
    u = min(time, stop)
    x = direction*u+a*u*u/2
    v = direction+a*u
    Q, W = C*abs(x), drive*x
    assert v*v/2-F(1, 2)+Q == W
    return x, v, stop, Q, W

for drive in map(F, [0, 1, 2]):
    a_pos, a_neg = drive-(drive+1), drive+(drive+1)
    assert (a_pos+a_neg)/2 == drive
    assert (a_neg-a_pos)/2 == drive+1
    for direction in [-1, 1]:
        for time in map(F, [0, "1/10", "1/2", 1, 2, 10]):
            x, v, stop, Q, W = case(drive, direction, time)
            if direction == 1:
                u = min(time, F(1))
                assert x == u-u*u/2 and stop == 1
            else:
                assert stop == 1/(2*drive+1)
        print("drive, direction, stop, x, heat, work:",
              drive, direction, stop, x, Q, W)

# Deterministic acceleration-error bounds, not a confidence interval.
m, N = F(1), F(10)
eps_pos, eps_neg = F(1, 10), F(1, 5)
beta = m*(eps_pos+eps_neg)/2
for drive in map(F, [0, 1, 2]):
    a_pos, a_neg = -F(1), 2*drive+1
    errors_F, errors_C = [], []
    for epos in [-eps_pos, eps_pos]:
        for eneg in [-eps_neg, eps_neg]:
            Fhat = m*(a_pos+epos+a_neg+eneg)/2
            Chat = m*(a_neg+eneg-a_pos-epos)/2
            errors_F.append(abs(Fhat-drive))
            errors_C.append(abs(Chat-(drive+1)))
            assert errors_F[-1] <= beta and errors_C[-1] <= beta
            assert abs(Chat/N-(drive+1)/N) <= beta/N
    assert max(errors_F) == max(errors_C) == beta
print("sharp force and coefficient error bounds:", beta, beta/N)
```

### Worked questions for contact-system identification

1. **Why do more forward samples not separate A,B,C?** Their analytic histories agree for every time. New measurement information or a different preparation is needed.
2. **Would a measured heat value help?** Yes. For the known monotone distance D=1/2 m, independently measured Q gives C=Q/D. A heat value reconstructed from a fitted energy residual is not an independent observation.
3. **What should a learned friction result report?** Specify calibrated inputs, directional coverage, acceleration-estimation error, stopping events and the static-threshold convention. A successful forward fit does not by itself identify every physical parameter.

The inverse family and uncertainty bounds are derived here using the same Coulomb law and system boundary as the native lesson.

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
