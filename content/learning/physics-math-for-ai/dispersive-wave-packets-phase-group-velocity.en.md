## A moving ripple is not necessarily a moving message

Imagine a short train of ripples approaching a sensor. Individual crests can travel through the train while its overall envelope moves at another speed. The train can also broaden, lowering its peak even when an integrated quantity is conserved. A single animation of a moving sine wave cannot distinguish these effects.

We will build a localized packet from Fourier components and follow three measurable features: carrier phase, envelope center and intensity width. We will also identify what this calculation cannot establish about physical energy or the first arrival of information. This extends the earlier nondispersive wave equation instead of replacing its initial-data and energy-balance reasoning.

## Variables, units and model assumptions

Let position $x$ be in m and time $t$ in s. The carrier wave number $k_0$ has units rad/m, angular frequency $\omega_0$ rad/s, group-velocity coefficient $v_g$ m/s and quadratic coefficient $\beta$ m²/s. Radians are dimensionless in dimensional analysis. The initial amplitude width $\sigma>0$ and center $x_0$ are lengths.

The complex field is dimensionless in this lesson. Its real part is the displayed oscillation. Its squared magnitude is a normalized observation; without an additional physical constitutive model it is not an energy density in J/m. Assume a linear, translation-invariant, lossless envelope model on the whole real line. Absorption, nonlinear distortion, boundaries and frequency-dependent detector response are outside this calculation.

## Define a localized initial state

Use a Gaussian amplitude envelope with a carrier:

$$
\psi(x,0)=A\exp\!\left[-\frac{(x-x_0)^2}{2\sigma^2}\right]
\exp[i(k_0x+\phi_0)].
$$

Here $A\ge0$ is an amplitude, not intensity. The initial intensity is $A^2\exp[-(x-x_0)^2/\sigma^2]$, whose standard deviation is $\sigma/\sqrt2$. Calling both widths “sigma” without specifying the observable creates a factor-of-two error in variance.

The Gaussian is convenient because Fourier integration remains explicit. It does not have compact support: it is nonzero arbitrarily far from its center in exact arithmetic. A finite plotting window hides tails rather than creating a physical signal front.

## Fix the Fourier convention first

Write $p=k-k_0$ and use the inverse convention

$$
\psi(x,t)=\frac{A\sigma}{\sqrt{2\pi}}
\int_{-\infty}^{\infty}e^{-\sigma^2p^2/2}
e^{i[(k_0+p)x-\omega(k_0+p)t+\phi_0-px_0]}\,dp.
$$

At $t=0$, the integral gives the prescribed initial state. The spectral amplitude has width parameter $1/\sigma$; this is an amplitude-width convention, not an unspecified uncertainty formula. Increasing spatial localization requires a wider range of wave numbers.

The signs in $e^{i(kx-\omega t)}$ determine all later phase and group directions. Reversing a transform convention is valid only if the initial spectrum and evolution convention change consistently. Checking reconstruction at zero time catches many sign and normalization errors before any animation is drawn.

## Two nearby waves explain group motion

For two equal-amplitude real waves, the elementary identity is

$$
\cos(k_1x-\omega_1t)+\cos(k_2x-\omega_2t)
=2\cos\!\left(\frac{\Delta k\,x-\Delta\omega\,t}{2}\right)
\cos(\bar kx-\bar\omega t).
$$

The slowly varying modulation follows $\Delta\omega/\Delta k$, whereas the rapid carrier phase follows $\bar\omega/\bar k$. Taking a narrow-band limit yields $v_g=\omega'(k_0)$. Two waves give a repeating beat pattern rather than a single localized pulse; many components are needed for localization.

The derivation concerns motion of interference features. It does not independently prove an energy or information speed. Those require a physical energy balance and a specified signal-preparation problem.

## Phase velocity and a local dispersion expansion

Expand the dispersion relation near the carrier:

$$
\omega(k_0+p)=\omega_0+v_gp+\tfrac12\beta p^2,
\qquad v_p=\frac{\omega_0}{k_0},\quad
v_g=\omega'(k_0),\quad\beta=\omega''(k_0).
$$

For a monochromatic carrier with $k_0\ne0$, a constant-phase line moves at $v_p$. If $k_0=0$, division does not define a phase velocity; report an unavailable value rather than infinity. An affine dispersion relation can preserve packet shape even if its nonzero intercept makes $v_p\ne v_g$.

In this lesson the quadratic polynomial defines the model exactly. When it approximates another medium, its coefficients must be derived from that medium and its usable band must be justified. The same algebra alone does not validate a real optical material across arbitrary frequencies.

## Evaluate the Gaussian packet exactly

Let $y=x-x_0-v_gt$ and $D=\sigma^2+i\beta t$. Completing the square in the Fourier integral gives

$$
\psi(x,t)=\frac{A\sigma}{\sqrt D}
\exp\!\left[-\frac{y^2}{2D}\right]
e^{i(k_0x-\omega_0t+\phi_0)}.
$$

Choose the square root continuously from $\sqrt D=\sigma$ at $t=0$. Since the real part of $D$ is positive, its argument lies between $-\pi/2$ and $\pi/2$; the principal branch makes that choice continuously for all supported real times.

The prefactor is essential. Simply widening the exponent while holding its peak fixed would create extra integrated squared magnitude. Zero time, zero dispersion and negative-time propagation are direct checks of the expression.

## Center, peak, width and conserved norm

Define $\zeta=\beta t/\sigma^2$ and $W=\sigma\sqrt{1+\zeta^2}$. Then

$$
|\psi|^2=\frac{A^2}{\sqrt{1+\zeta^2}}
\exp\!\left[-\frac{y^2}{W^2}\right],
\qquad \int_{\mathbb R}|\psi|^2dx=A^2\sigma\sqrt\pi.
$$

For a nonzero field, the intensity center is $x_0+v_gt$ and its variance is $W^2/2$. Broadening lowers the peak by the reciprocal width factor. A sensor at one fixed location can therefore see a smaller signal without any loss of the integrated norm.

This norm has units m under the dimensionless field normalization. It has no automatic interpretation as total mechanical energy. When $A=0$, the field and norm are zero, but a normalized intensity centroid is undefined. The configured geometric center can still be reported separately from an observed centroid.

## Chirp changes the local phase geometry

Write $r=\sigma^2$, $b=\beta t$ and $Q=r^2+b^2$. The full phase is

$$
\theta=k_0x-\omega_0t+\phi_0-\tfrac12\arctan(b/r)
+\frac{b y^2}{2Q},\qquad
k_{\rm local}=\partial_x\theta=k_0+\frac{by}{Q}.
$$

The local wave number varies across the packet. Positive and negative $\beta$ give the same intensity width at equal $|t|$ but opposite envelope chirp. An intensity snapshot alone cannot identify this sign.

The actual local constant-phase velocity is $-\partial_t\theta/\partial_x\theta$ wherever the denominator and field are nonzero. It need not equal $\omega_0/k_0$. This distinction prevents a carrier reference line from being mislabeled as the trajectory of every visible crest. A zero field has no observable phase; an almost flat phase gradient also makes crest velocity a poorly conditioned description.

A small but nonzero phase gradient retains its finite ratio. For example, local wave number $10^{-13}$ rad/m and local angular frequency 8 rad/s give $8\times10^{13}$ m/s. This is a local constant-phase speed, not a claim about signal, information or energy transport. The model distinguishes a zero field, an exact zero gradient, a finite ratio and a ratio outside floating-point range. It does not replace a small nonzero gradient with a zero-gradient cutoff. The carrier reference also distinguishes zero wave number from an unrepresentable ratio.

## The envelope equation provides another check

Factor $\psi=B(x,t)e^{i(k_0x-\omega_0t+\phi_0)}$. Each Fourier component of $B$ obeys the same quadratic evolution, so

$$
i(\partial_tB+v_g\partial_xB)=-\frac{\beta}{2}\partial_x^2B.
$$

One can differentiate the exact Gaussian and substitute it into this equation. This checks the sign of spreading independently of the zero-time test. The implementation exposes the envelope derivatives multiplied by the same carrier factor, making comparison possible without mixing derivatives of $B$ and $\psi$.

The equation also has a squared-envelope conservation law with current $j=v_g|B|^2+\beta\operatorname{Im}(B^*B_x)$. This is the current of this envelope norm. Calling it electromagnetic power requires additional physics that this effective equation does not supply.

## Work through a numerical example

Take $A=1$, $\sigma=1$ m, $x_0=0$, $k_0=4$ rad/m, $\omega_0=8$ rad/s, $v_g=1$ m/s and $\beta=0.5$ m²/s. At $t=2$ s, $\zeta=1$, the center is2 m and the amplitude width is $\sqrt2$ m. The intensity standard deviation is1 m; the peak intensity is $1/\sqrt2\approx0.7071$.

The integrated norm remains $\sqrt\pi$ m. The carrier reference speed is2 m/s while the envelope center moves at1 m/s. At the center, the chirp gradient is zero but its time-dependent phase contributes to local frequency: $\omega_{\rm local}=8.125$ rad/s. Thus the local crest speed there is2.03125 m/s, distinct from both reference speeds.

Changing $\beta$ to $-0.5$ leaves the intensity snapshot unchanged at this time but reverses the chirp. Measuring intensity and measuring complex phase are different inverse problems.

## Conditions for using a Taylor approximation

If an actual relation contains a cubic remainder, a sufficient diagnostic on a chosen band $|p|\le P$ is that $t\sup|\omega'''|P^3/6$ be small compared with a radian. This controls an accumulated phase remainder on that band, not every possible observed error. A Gaussian has tails outside every finite band, so their spectral weight must also be estimated.

Longer propagation can invalidate a narrow-band approximation even when the initial packet looks smooth. Compare the exact material dispersion with its quadratic approximation at the same initial spectrum, rather than adjusting widths afterward to force agreement. Loss and boundaries can introduce entirely different errors.

An effective quadratic relation can also produce negative frequencies far outside its local band. That does not invalidate the stated mathematical envelope model, but it prevents interpreting the whole polynomial as a globally faithful positive-frequency material law.

## Information fronts and what a plot cannot show

A group peak is a feature of an already prepared waveform. An information-front experiment instead asks when a previously absent disturbance can first influence a receiver. The initial Gaussian does not supply an exactly undisturbed exterior region, so its moving peak cannot answer that question.

Nor does a finite observation window measure all norm or energy. Separately track the analytic total, the observed region and any numerical truncation. A low sampled amplitude can result from spreading, translation, phase cancellation in the real part or actual loss; this lossless model supports only the first three.

For research, record assumptions before comparing velocities. Neither a visual peak nor a fitted line through several maxima proves causality, a universal transport theorem or a result for an unspecified material.

## Research and AI experiments with negative controls

An operator-learning experiment can vary $\sigma$, $v_g$, $\beta$ and initial phase, then compare predictions against exact complex fields. Evaluate amplitude, phase and conserved norm separately. A model that predicts intensity alone cannot distinguish the two chirp signs in the symmetric example above.

Use $\beta=0$ as a no-spreading control and $A=0$ as an undefined-observable control. Hold the initial spectrum fixed when comparing propagation laws. Split training and evaluation by physical parameter regimes, not by randomly shuffling neighboring points from the same trajectory.

Independent Fourier quadrature tests the closed form, symbolic substitution tests its equation and real-space integration tests moments. Agreement among several rearrangements of one copied expression would supply weaker evidence. Rendering and meaningful interactions still require separate verification after these mathematics checks.

## Explained exercises

### Exercise 1 — Which width is measured?

An amplitude envelope is $e^{-x^2/(2\sigma^2)}$ with $\sigma=2$ m. Find the intensity variance.

**Solution:** Squaring gives $e^{-x^2/\sigma^2}$. Its variance is $\sigma^2/2=2$ m² and standard deviation $\sqrt2$ m. Using4 m² confuses amplitude width with intensity statistics.

### Exercise 2 — Spreading without losing norm

Let $\beta t/\sigma^2=3$. How do width and peak intensity change?

**Solution:** The width increases by $\sqrt{10}$ and the peak intensity decreases by $1/\sqrt{10}$. Their product stays fixed, preserving the Gaussian integral; keeping the peak fixed would violate that norm conservation.

### Exercise 3 — Equal snapshots, different phases

Can one intensity snapshot determine the sign of $\beta$ when the initial envelope is unchirped?

**Solution:** No. Intensity depends on $(\beta t)^2$, whereas the envelope phase changes sign with $\beta t$. A phase-sensitive measurement or additional known preparation is needed to break this ambiguity.

### Exercise 4 — Does the carrier label track every crest?

For the numerical example at its center and $t=2$ s, compare the three speeds.

**Solution:** The carrier reference is $8/4=2$ m/s; the envelope speed is1 m/s. The full phase gives local angular frequency8.125 rad/s and wave number4 rad/m, so the local crest speed is2.03125 m/s. The chirp phase cannot be omitted in actual crest tracking.

### Exercise 5 — A zero-amplitude packet

What center and phase should be inferred when $A=0$?

**Solution:** The field and norm vanish. Configured $x_0+v_gt$ still exists as a parameter, but dividing a zero first moment by a zero norm does not define an observed centroid. Phase and local crest speed are unavailable, rather than meaningful zeros.

### Exercise 6 — From a peak to a causal front

Does observing a Gaussian peak at $x=v_gt$ measure the earliest possible signal arrival?

**Solution:** No. The Gaussian already has nonzero tails everywhere. A front experiment needs a specified change of source or initially absent disturbance and a physical propagation model. Peak motion alone does not determine its causal domain.

## Sources and reproducible checks

[MIT 8.03SC lecture13](https://www.ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/432d23acf0bce355921a8b9a4eefb0b6_MIT8_03SCF16_Lec13.pdf) motivates dispersion and beating; [Harvard Matthew Schwartz lecture11](https://schwartz.scholars.harvard.edu/sites/g/files/omnuum7046/files/schwartz/files/lecture11-wavepackets.pdf) discusses Fourier packets and dispersion expansions. The conventions, full prefactor, local-phase calculation, examples and verification here are derived explicitly for this lesson.

`gaussianDispersivePacket` in `src/lib/learning-dispersion.ts` computes fields and diagnostics. The independent Python generator integrates the initial Fourier spectrum and real-space moments; the TypeScript tests compare the actual model with those references and check limits and the envelope equation. A successful model test does not by itself claim that the lesson has been registered, rendered or visually reviewed on the homepage.
