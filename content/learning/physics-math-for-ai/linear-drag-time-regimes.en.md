## A model can be accurate for one quantity and inaccurate for another

Imagine predicting a falling object's speed and distance using free-fall formulas. When resistance has only just started acting, free fall may be a useful approximation. Much later, a terminal-speed description may be better. There is no universal dividing time that makes every observable accurate to every tolerance. The acceptable regime depends on what is predicted and how the error is measured.

This lesson uses a parent model that can be solved analytically. The object starts at rest with position zero, downward is positive, gravity is constant, and resistance is proportional to velocity. The comparison isolates approximation within that model. It does not establish that linear drag describes every real falling object, flow speed, shape or Reynolds number.

## Write the parent law with units and initial data

Let m be mass in kilograms, b the nonnegative linear drag coefficient in kg/s, and g positive gravitational acceleration in m/s squared. The equations are

$$
m v'(t)=mg-bv(t),\qquad x'(t)=v(t),
\qquad x(0)=v(0)=0.
$$

The drag term opposes positive downward velocity. The interface uses b equal to the drag code divided by ten, g equal to the gravity code divided by two, and time equal to the time code divided by ten seconds. The coefficient b is not a dimensionless fraction.

For b positive, define the relaxation time and terminal velocity:

$$
\tau=\frac{m}{b},\qquad v_T=g\tau,\qquad z=\frac{t}{\tau}.
$$

These definitions expose the relevant time ratio. A time of one second can be early for one mass-to-drag ratio and late for another. The phrase early must be interpreted relative to tau rather than a fixed clock value.

## Integrate the reference solution

Solving the linear velocity equation from rest gives

$$
v(t)=g\tau(1-e^{-z}).
$$

Integrating once more with x(0)=0 yields

$$
x(t)=g\tau^2(z-1+e^{-z})
=g\tau\left[t-\tau(1-e^{-z})\right].
$$

Direct differentiation verifies both the force law and the position-velocity relation. Both reference quantities are nonnegative for nonnegative time. For position, the inequality exp(-z) at least 1-z proves that property analytically. The formulas are exact expressions within the declared parent model; evaluating exponentials and coordinates on a computer remains numerical.

The first figure uses physical seconds and the chosen observable's units. The second uses dimensionless time z and either v/(g*tau) or x/(g*tau squared). Changing physical scales can leave those normalized curves unchanged. This is a useful consistency check rather than an assertion that omitted fluid physics has disappeared.

## Derive the early free-fall approximation

Expanding the exponential near z=0 gives

$$
v=gt\left(1-\frac{z}{2}+\frac{z^2}{6}+\cdots\right),
\qquad x=\frac{gt^2}{2}\left(1-\frac{z}{3}+\frac{z^2}{12}+\cdots\right).
$$

The leading terms are v approximately gt and x approximately gt squared/2. Their first relative corrections have different coefficients. There is already a reason to expect different error tolerances for speed and distance at the same time.

In the first guide m=1 kg, b=1 kg/s and g=10 m/s squared, so tau=1 second. At t=0.1 second the parent velocity is approximately 0.95162582 m/s, while free fall gives 1 m/s. Using the parent magnitude as denominator, the relative error is approximately 5.08332 percent. It fails a four-percent request.

## Change the observable without changing the physical experiment

The second guide retains the same mass, drag, gravity and time, but selects displacement. The reference is approximately 0.04837418 meters and the free-fall approximation is 0.05 meters. The relative error is about 3.36092 percent, so it passes the same four-percent requirement.

The trajectory did not change when the observable selector changed. The quantity being judged changed. A blanket statement that free fall is valid or invalid at this time would hide the application-specific distinction. The readout therefore names the observable, approximation, reference and requested tolerance separately.

At time zero, the parent observable is zero. Relative error defined by division by its magnitude is unavailable, including when the early approximation and absolute error are both zero. Replacing the denominator by one or an arbitrary small epsilon would define a different metric. The interface leaves the comparison unavailable instead of manufacturing a positive validity flag.

## Derive the late approximation and retain its offset

As z becomes large, the exponential becomes small. The late formulas are

$$
v_{\mathrm{late}}=g\tau,\qquad
x_{\mathrm{late}}=g\tau(t-\tau).
$$

The displacement includes a constant offset. Simply multiplying terminal velocity by time drops the integrated memory of the initial acceleration. That unshifted expression is another approximation, with a different error.

The third guide uses the same tau=1 second and g=10, but t=4 seconds. The reference displacement is approximately 30.183156 meters and the shifted late approximation is 30 meters. The relative error is about 0.60682 percent, passing a one-percent requirement. The terminal-velocity approximation at that same time has relative error about 1.86574 percent and fails it.

For t less than tau, the shifted late displacement is negative. The parent displacement is not negative. This result exposes misuse of the late approximation outside its regime. The figure retains it with the approximation label instead of clipping it to zero or presenting it as upward physical motion.

## Handle zero drag as a separate limit

When b=0, the original differential equation reduces to free fall. It has v=gt and x=gt squared/2 for these initial data. There is no finite relaxation time or terminal velocity under sustained positive gravity.

The interface uses that solution directly. The early approximation agrees with it, while tau, terminal velocity and the late approximation remain unavailable. A division by zero is not converted into an artificial finite terminal speed. The normalized terminal-scale figure is also unavailable because its scaling quantities do not exist as finite values.

The small-drag limit at fixed physical time is still consistent with free fall. Taking a long-time approximation while simultaneously letting the relaxation time grow without checking t/tau mixes limits and can give misleading results. A regime statement must say which quantities are held fixed.

## Separate numerical cancellation from physical approximation

For small z, directly computing 1-exp(-z) subtracts nearly equal numbers. The expm1 function evaluates exp(-z)-1 more accurately. Position contains an additional cancellation in z-1+exp(-z), so using expm1 alone is insufficient at very small z.

A stable implementation evaluates entire response functions: velocity is gt times (1-exp(-z))/z, and position is gt squared times (z-1+exp(-z))/z squared. Their limits at zero are one and one half. A convergent small-z series evaluates the second function without subtracting two terms of order z to obtain a term of order z squared.

Independent high-precision convolution integrals check the reference implementation. Those numerical checks concern evaluation error. Differences between the early or late approximation and that reference concern truncation of the declared model. Experimental agreement of the linear-drag law is a third question requiring physical evidence.

## Connect the validity criterion to AI research

A learned simulator might predict positions accurately while its finite-difference velocity estimates remain poor. Evaluating only one quantity can miss that issue. A benchmark should state its observable, units, normalization, time range and acceptable error rather than relying on a single label such as physically correct.

Dimensionless time provides a useful way to compare different masses and drag coefficients. It also warns against extrapolating a surrogate trained only on early motion into the terminal regime. Success inside the parent model does not automatically justify a different drag law or new physical setting. Separate approximation, numerical computation and empirical modeling claims when designing experiments and reporting results.

## Exercises with worked solutions

1. **Exercise:** For m=1,b=1,g=10, compute tau and terminal velocity. **Solution:** They are one second and ten meters per second.
2. **Exercise:** Why can early displacement pass a four-percent criterion when early velocity fails at t=0.1? **Solution:** They have different relative corrections and different reference magnitudes; their errors are about 3.36092 and 5.08332 percent.
3. **Exercise:** Is relative error zero when both values are zero? **Solution:** The specified ratio is 0/0 and unavailable; absolute error can still be zero.
4. **Exercise:** What offset belongs in the late displacement? **Solution:** Integrating the reference gives g*tau*(t-tau), retaining minus g*tau squared.
5. **Exercise:** Does a negative late displacement mean the parent object moved upward? **Solution:** No. It signals an out-of-regime approximation; the parent displacement stays nonnegative.
6. **Exercise:** What is the terminal velocity when b=0 with positive gravity? **Solution:** No finite terminal velocity exists; the parent solution is free fall.

## Sources and further study

[TU Graz's linear-drag mechanics notes](https://lampz.tugraz.at/~hadley/physikm/script/motion/terminal.en.php) present the analytic force, velocity and position law using an upward-positive convention. This lesson converts to downward-positive coordinates and derives the asymptotic errors directly. Continue with singular limits, dimensionless modeling and empirical tests of constitutive laws to study wider validity questions.
