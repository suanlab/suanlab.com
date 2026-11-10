## A push has a duration and a history

Imagine pushing two identical carts on a frictionless track. In the first experiment you push steadily; in the second you start gently and increase the force. You arrange equal signed areas under the force curves. The carts finish with the same velocity, yet one has traveled farther. This is the distinction between a final change and the history that produced it.

We use a Newtonian point particle with constant positive mass m, initial position zero and known initial velocity v0. Forces are prescribed functions of time in an inertial laboratory. These are two separate experiments with identical initial data, not two forces simultaneously acting on one cart. The model contains no contact detector or collision rule.

## Specify both force profiles and the switching convention

Let duration τ be positive and amplitude F0 be signed. Experiment A uses a flat pulse; experiment B uses a linear ramp:

$$
F_A(t)=\begin{cases}F_0&0\le t<\tau,\\0&t\ge\tau,\end{cases}
\qquad
F_B(t)=\begin{cases}2F_0t/\tau&0\le t<\tau,\\0&t\ge\tau.\end{cases}
$$

The ramp reaches a left-hand limiting value of twice the flat amplitude. Its average over the pulse is still F0. We define force to be zero at the switching instant itself. Changing a finite value at a single instant would not change any integral below. Vertical force segments in the illustration mark one-sided jumps; they do not describe a physical interval during which all intermediate forces occur.

## Momentum balance integrates signed force

For constant mass, the momentum law is dp/dt=F. Integrate it from zero to the query time. Writing u=min(t,τ) gives

$$
J_A(t)=F_0u,\qquad J_B(t)=\frac{F_0u^2}{\tau},\qquad
p(t)-mv_0=J(t).
$$

Impulse has units N s, equivalent to kg m/s. It is signed in this one-dimensional problem. A negative pulse decreases signed momentum; it need not decrease speed throughout the experiment. Before the pulse finishes, the accumulated impulses generally differ. At and after the end, both equal F0τ. This is an external-impulse balance, rather than an assertion that this particle's momentum is conserved.

## Velocity follows accumulated area

Divide momentum by the same constant mass:

$$
v_A(t)=v_0+\frac{F_0u}{m},\qquad
v_B(t)=v_0+\frac{F_0u^2}{m\tau}.
$$

For positive amplitude, the flat experiment gains velocity earlier. At the pulse end the ramp catches up. Thereafter both particles move freely with their shared terminal velocity. For zero amplitude the two experiments reduce to the same uniform motion. No formula divides by amplitude, so the zero-force case remains well defined.

A finite force jump changes acceleration discontinuously. It does not create a finite jump in velocity: an interval shrinking to zero has vanishing force area when the force remains bounded. An ideal Dirac impulse is a different mathematical model and is not silently introduced here.

## Integrate again to obtain position

The time at which momentum was acquired matters for displacement. An infinitesimal force contribution at time s changes velocity for the remaining interval t−s. Therefore

$$
x(t)=v_0t+\frac1m\int_0^u(t-s)F(s)\,ds.
$$

Carrying out this integral yields the formulas used by the interactive model:

$$
x_A(t)=v_0t+\frac{F_0}{m}\left(ut-\frac{u^2}{2}\right),
$$

$$
x_B(t)=v_0t+\frac{F_0}{m\tau}\left(u^2t-\frac{2u^3}{3}\right).
$$

These expressions also cover times after the pulse. Simply substituting the query time into a during-pulse cubic would incorrectly keep accelerating the ramp particle forever. The clipped time u prevents that mistake while the unmodified t retains later free travel.

## Equal terminal impulse leaves a position gap

At t=τ the two positions differ by

$$
x_A(\tau)-x_B(\tau)=\frac{F_0\tau^2}{6m}.
$$

For positive amplitude, the earlier push puts A ahead. For negative amplitude the signed ordering reverses. After the pulse, the velocity difference is zero, so this position gap remains constant. The initial velocity cancels in the difference because both experiments started with the same value.

This gives a concrete inverse-problem warning: terminal momentum alone cannot identify the force history or final position. Even with duration known, the flat and ramp histories are distinct explanations of the same final momentum change. A peak force alone also cannot determine impulse without temporal information.

## Work uses force times actual velocity

Kinetic energy is K=mv²/2. Using dJ=F dt gives an exact expression for accumulated work in either experiment:

$$
W=\int_0^tF(s)v(s)\,ds
=\int_0^{J(t)}\left(v_0+\frac{j}{m}\right)dj
=v_0J+\frac{J^2}{2m}=K(t)-K(0).
$$

The antiderivative argument remains valid for negative impulse. Work may be negative when a force initially opposes motion. Equal terminal impulse and identical initial data imply equal terminal kinetic changes here, even though terminal positions differ. Multiplying endpoint force by total displacement fails for the time-varying ramp; after the switch endpoint force is zero despite earlier nonzero work.

## Read the three diagrams in order

The first diagram shows force against time in N. Compare signed areas, not just the heights. The second shows accumulated impulse in N s; the curves meet at the switch and remain together afterward. The third shows position in meters, where the surviving offset is visible. Each panel has its own vertical scale and unit.

Query time changes the exact table and readouts. The plotted histories retain a fixed ten-second horizon so a query change does not disguise later behavior by rescaling time. Set amplitude to zero, reverse its sign, and choose the shortest duration to inspect limiting cases. The exact rational table is the authority for current control values; sampled lines illustrate continuous histories.

## Six exercises with worked solutions

1. **Exercise: end of the pulse.** Set m=1kg, F0=2N, τ=2s, v0=0. Find both states at2s. **Solution:** J=4N s and v=4m/s in both. Positions are4m and8/3m. Both kinetic energies are8J; their separation is4/3m.
2. **Exercise: halfway.** Keep the settings and query1s. **Solution:** JA=2 and JB=1N s. Velocities are2 and1m/s; positions are1 and1/3m. Equal final impulse does not imply equal partial impulse.
3. **Exercise: later free motion.** Query4s. **Solution:** both velocities remain4m/s. Adding two seconds of free travel gives xA=12m and xB=32/3m. The gap remains4/3m.
4. **Exercise: braking work.** Take m=1kg, v0=2m/s, F0=−1N, τ=1s at the end. **Solution:** J=−1N s and v=1m/s. Work is−2+1/2=−3/2J in both, matching K changing from2 to1/2J. Their signed position difference is−1/6m.
5. **Exercise: mass scaling.** Double the first exercise's mass while holding force and duration fixed. **Solution:** impulse stays4N s, velocity becomes2m/s and the gap becomes2/3m. Impulse is prescribed by force history; its velocity effect depends on mass.
6. **Exercise: switching instant.** What are both forces exactly at τ, and can their zero values explain total work? **Solution:** both are zero by convention. Both still have4N s of accumulated impulse in the first example. A single endpoint value cannot replace the full work integral.

## Connection to learning dynamical models

A learned simulator supervised only on terminal velocity can fit both histories while getting intermediate positions wrong. Adding trajectory observations constrains a different part of the model. A momentum-balance residual and a work-balance residual are useful checks, but zero residuals do not prove that the prescribed force matches an experiment. Here the force law and initial data are assumptions; validating them requires observations beyond the algebraic identities.

## Background and scope

For the standard momentum, impulse and center-of-mass framework, consult [MIT OpenCourseWare, Classical Mechanics, Week5](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/pages/week-5-momentum-and-impulse/). The two pulse histories, their convolution formulas and the worked values above are derived for this lesson. Variable mass, relativistic momentum and contact-force reconstruction require additional models.
