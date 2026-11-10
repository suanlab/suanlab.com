## A slowing oscillator has not lost the energy accounting

A spring-mounted cart released from displacement oscillates less and less when a viscous resistance acts. Its mechanical energy decreases. That decrease is not a failure of accounting: the model transfers energy into a dissipated heat quantity. If an external actuator also pushes, input work must be included with its sign.

This lesson solves the actual time-dependent initial-value problem, rather than only drawing an energy-level set. Mass and spring stiffness are positive constants, damping is nonnegative, and the external drive is constant. We distinguish stored mechanical energy, drive work and integrated heat. The model specifies no temperature, heat capacity, thermal reservoir or periodic forcing.

## State which interactions store and transfer energy

Let x be displacement from the spring's undeformed position and v its velocity. With stiffness k, damping coefficient c and constant drive F,

$$
m\ddot x+c\dot x+kx=F,\qquad x(0)=x_0,\quad \dot x(0)=v_0.
$$

The spring's potential energy is V=kx²/2 and kinetic energy is K=mv²/2. We keep the constant drive outside the stored mechanical energy, so its work appears explicitly. One could instead include −Fx in a different potential convention, but then the corresponding work term must be removed to avoid double counting.

The viscous force is −cv. Its coefficient has units kg/s, not the units of spring stiffness N/m. Its power is −cv², nonpositive for every velocity sign. Resistance opposes velocity even when the cart moves left.

## Shift to equilibrium before solving the motion

The static equilibrium is xe=F/k. Set y=x−xe, y0=x0−xe and α=c/(2m). The equation becomes

$$
\ddot y+2\alpha\dot y+\omega_0^2y=0,\qquad \omega_0^2=k/m.
$$

The discriminant c²−4mk identifies the regimes. A constant drive shifts equilibrium without introducing a driving frequency. Calling it a resonant periodic experiment would therefore be incorrect. Changing the drive also changes y0 for fixed physical initial position; initial data must be transformed along with the coordinate shift.

## Underdamped motion retains oscillation

When c²<4mk, let ω=√(ω0²−α²). The initial-value solution is

$$
y(t)=e^{-\alpha t}\left[y_0\cos(\omega t)+\frac{v_0+\alpha y_0}{\omega}\sin(\omega t)\right],
$$

$$
v(t)=e^{-\alpha t}\left[v_0\cos(\omega t)-\frac{\alpha v_0+\omega_0^2y_0}{\omega}\sin(\omega t)\right].
$$

With c>0 the envelope decays. With c=0 the same branch reduces to undamped harmonic motion and dissipated heat is exactly zero. The interface names that limit separately. Mechanical energy may vary under a constant drive even in the zero-damping case because the actuator can perform work.

## Critical damping requires the repeated-root solution

At c²=4mk, the two characteristic roots coincide. Distinct-root expressions cannot be evaluated by dividing by their now-zero difference. Use

$$
y(t)=e^{-\alpha t}[y_0+(v_0+\alpha y_0)t],
$$

$$
v(t)=e^{-\alpha t}[v_0-\alpha(v_0+\alpha y_0)t].
$$

The t multiplying the exponential is essential. Both specified initial conditions are retained. “Critical” identifies a root structure, rather than a blanket claim about the fastest response for arbitrary initial data and arbitrary response metrics.

## Overdamped motion has two real decay rates

For c²>4mk, define roots rslow and rfast. Numerically, the slow root is evaluated without subtracting two nearly equal positive numbers:

$$
r_{\mathrm{fast}}=-\alpha-\sqrt{\alpha^2-\omega_0^2},\qquad
r_{\mathrm{slow}}=-\frac{\omega_0^2}{\alpha+\sqrt{\alpha^2-\omega_0^2}}.
$$

Then y=Cs exp(rslow t)+Cf exp(rfast t), where the two coefficients solve Cs+Cf=y0 and rslow Cs+rfast Cf=v0. Velocity is obtained by differentiating these exponentials. Both roots are negative under the stated positive-mass, positive-stiffness and positive-damping assumptions. Depending on initial data, sums of decays can still cross a position or reverse direction; “overdamped” does not itself specify every trajectory shape.

## Derive the instantaneous power balance

Multiply the original motion law by v:

$$
\frac{d}{dt}\left(\frac12mv^2+\frac12kx^2\right)=Fv-cv^2.
$$

Define mechanical energy E=K+V, drive power Pdrive=Fv and damping power Pdamp=−cv². Their sum is the rate of change of stored mechanical energy. Drive power can be positive or negative. Damping power cannot be positive in this model, while the associated heat generation rate is +cv².

A force value alone does not give power: the current velocity is also needed. An instant at rest has zero power even if the spring and drive forces are nonzero and acceleration is about to change the velocity.

## Integrate drive work and dissipated heat separately

Because the drive is constant, its accumulated work is F[x(t)−x0]. Heat is computed from the independent integral

$$
Q(t)=\int_0^t c\,v(s)^2ds\ge0,
\qquad E(t)-E(0)+Q(t)=W_{\mathrm{drive}}(t).
$$

The implementation does not define Q as whatever makes the residual vanish. It integrates the analytic squared-velocity expression. In the oscillatory branch this uses exponential sine/cosine integrals. At the repeated root it uses moments of an exponential times a quadratic polynomial. In the real-root branch it integrates the three exponential terms in the squared sum.

Short-time exponential moments use a convergent series to avoid subtracting nearly equal numbers. All transcendental formulas are evaluated in floating point. Their residual is a numerical diagnostic, not an exact-rational proof. An independent numerical integration of the differential equation and heat rate checks both motion and heat.

## Read the position, energy and power panels

Every horizontal axis is time in seconds. The position panel shows the actual analytic motion in meters. The energy panel compares E, Q and E0+Wdrive in joules; the sum E+Q should match the latter. The power panel shows drive power, negative damping power and their sum in watts.

The fixed ten-second horizon is an illustration of the same initial-value solution. Query time selects current readouts without changing that horizon. Try zero damping, exact critical damping, strong overdamping, a state already at driven equilibrium and a drive opposing an initially positive velocity. Distinguish a negative work value from a negative heat value: only the former is possible under these assumptions.

## Six exercises with worked solutions

1. **Exercise: undamped limit.** Use m=k=1,c=0,F=0,x0=1,v0=0. **Solution:** x=cos t and v=−sin t, so E=1/2J at every time and Q=Wdrive=0. Position changes while total mechanical energy stays fixed.
2. **Exercise: classify roots.** Keep m=k=1 and choose c=1,2,3. **Solution:** discriminants are−3,0,5, giving underdamped, critical and overdamped motion respectively. The critical case uses the repeated-root formula.
3. **Exercise: critical step drive.** Set m=k=1,c=2,F=1,x0=v0=0. At1s? **Solution:** x=1−2/e and v=1/e. Drive work equals x≈0.264241J; stored energy is about0.102579J.
4. **Exercise: heat for that step.** Integrate the heat rate directly. **Solution:** v=t exp(−t), so Q(1)=2∫0¹t²exp(−2t)dt=(1−5exp(−2))/2≈0.161662J. Adding it to stored energy reproduces the work in exercise3.
5. **Exercise: already at equilibrium.** Use F=k=1,x0=1,v0=0 and any allowed damping. **Solution:** y0=0 and v0=0 give x=1 permanently. Stored spring energy is1/2J, but drive work, heat and both powers are zero. A nonzero drive need not continually inject energy.
6. **Exercise: signed braking input.** Use m=k=c=1,F=−1,x0=0,v0=2 and query0.1s. **Solution:** velocity remains positive initially, so drive power and accumulated drive work are negative. Heat is positive because c v² is positive. The mechanical energy decreases by the magnitude of removed drive work plus generated heat.

## What an energy check can and cannot certify

A learned model can hide trajectory errors by defining an unobserved dissipation term from the residual. Integrating a declared nonnegative heat rate independently is a stronger check. Even then, an accurate balance alone does not identify the correct friction or drive law. Compare measured trajectories, inferred forces and the energy budget under stated assumptions. Real thermal modeling needs additional variables and constitutive relations beyond the cumulative Q used here.

## Background and scope

For the characteristic-root framework, see [MIT OCW, Damped Harmonic Oscillators](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-ii-second-order-constant-coefficient-linear-equations/damped-harmonic-oscillators/). The constant-drive shift, signed work and independent heat calculations above follow from the declared model. Periodic resonance and statistical thermal reservoirs are developed separately.
