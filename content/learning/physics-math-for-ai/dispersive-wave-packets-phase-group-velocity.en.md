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

## Research extension: quantify omitted cubic dispersion

The quadratic envelope above is an exact solution of its stated model. Its accuracy for another dispersion relation is a different question. A small omitted coefficient can accumulate a large phase over a long observation. We compare two lossless models with exactly the same initial Gaussian spectrum, then separate field error, centroid motion and physical interpretation. The cubic coefficient is a supplementary calculation, not a new native slider.

### Choose a common initial state and error scale


$$
p=k-k_0,\quad\omega_2(p)=\omega_0+v_g p+\frac{\beta p^2}{2},\quad\omega_3(p)=\omega_2(p)+\frac{\gamma p^3}{6},\qquad h=\frac{\gamma t}{\sigma^3}.
$$

Assume A>0, sigma>0 and real coefficients. Gamma has units m³/s, beta has units m²/s and h is dimensionless. Each spectral component changes only its phase. The polynomial is a declared mathematical dispersion model; it need not describe all wave numbers of a real material. Let psi2 and psi3 denote the two complex fields, using the same unitary Fourier convention and initial amplitude width sigma.


$$
\rho_\sigma(p)=\frac{\sigma}{\sqrt\pi}e^{-\sigma^2p^2},\quad\|\psi_0\|_2^2=A^2\sigma\sqrt\pi,\qquad
D^2(t)=\frac{\|\psi_3-\psi_2\|_2^2}{\|\psi_0\|_2^2}=\int_{-\infty}^{\infty}\frac{e^{-q^2}}{\sqrt\pi}\,4\sin^2\!\left(\frac{hq^3}{12}\right)dq,\quad q=\sigma p.
$$

Plancherel turns a spatial field error into a Gaussian average of the squared phase difference. Common carrier, translation and quadratic phases cancel, so this relative error is independent of beta and vg. The nonnegative sine-squared integrand avoids subtracting two nearly equal overlaps. A conserved norm does not make the fields equal.

### Derive a bound before choosing the observation horizon


$$
\mathbb E_{\rho_\sigma}p^6=\frac{15}{8\sigma^6},\qquad D(t)\le\min\!\left(2,\sqrt{\frac5{96}}\frac{|\gamma t|}{\sigma^3}\right).
$$

Apply abs(exp(iz)−1)≤min(2,abs(z)) and integrate the sixth Gaussian moment. Repeated integration by parts gives the factor 15/8. This is an exact model bound, not a claim that the numerical integral is certified to every printed digit. A narrower initial packet has a broader spectrum and amplifies the omitted cubic term as sigma to the power −3.


$$
h_0\ne0,\quad t_\gamma=\frac{h_0\sigma^3}{\gamma}:\qquad D(t_\gamma)=D(h_0)>0\quad\text{as }\gamma\to0\text{ with }h_0/\gamma>0.
$$

For every fixed time the error tends to zero with gamma. Along the growing positive-time horizon above it stays nonzero: the sine-squared integrand is positive on a set of positive measure. A claim of uniform long-time accuracy therefore needs an explicit time scale.

### Compute what the intensity centroid actually measures


$$
\langle x\rangle_3=x_0+t\left(v_g+\frac{\gamma}{4\sigma^2}\right),\qquad\operatorname{Var}_3 x=\frac{\sigma^2}{2}+\frac{\beta^2t^2}{2\sigma^2}+\frac{\gamma^2t^2}{8\sigma^4}.
$$

In Fourier space position evolves as x(t)=x(0)+t omega-prime(p). For the unchirped initial Gaussian, the symmetrized covariance of initial position with p or p² is zero. Use E p=E p³=0, E p²=1/(2 sigma²) and Var(p²)=1/(2 sigma⁴). The centroid speed is the spectral average of omega-prime, which equals vg for the quadratic model but acquires the displayed finite-bandwidth shift here. The cubic intensity is generally not Gaussian: its mean and variance do not specify its full profile. These are squared-field moments, not a mechanical-energy center or a signal-front velocity.

### Keep Gaussian tails when using a band-limited remainder estimate


$$
|r(p)|\le K|p|^3/6\quad(|p|\le P),\quad\omega_{\rm actual}=\omega_2+r\text{ real}:\qquad D^2\le\frac{5K^2t^2}{96\sigma^6}+4\operatorname{erfc}(\sigma P).
$$

Inside the band, use the same phase inequality and bound its sixth moment by the full Gaussian moment. Outside, use the worst squared distance 4 between two unit phasors. The normalized spectral probability outside the band is erfc(sigma P), not the amplitude-tail expression with an extra square root of two. This estimate requires real dispersion outside the band too; unspecified absorption does not satisfy that unit-phasor assumption.


$$
\sup_x|\psi_3-\psi_2|\le\min\!\left(2A,\frac{2A}{3\sqrt{2\pi}}\frac{|\gamma t|}{\sigma^3}\right).
$$

The spectral L1 triangle inequality gives this absolute pointwise bound: integrate abs(p)³ exp(−sigma² p²/2), whose value is 4/sigma⁴. It does not control relative error or phase at field zeros. The following table uses the exact spectral error integral; values are rounded to nine significant digits. Simpson integration over q in [−8,8] omits at most 4 erfc(8) from D²; discretization error is checked separately by refinement.

| h | relative field error D | L2 upper bound | centroid shift / sigma |
|---|---|---|---|
| 0.25 | 0.0567041208 | 0.0570544331 | 0.0625 |
| 0.5 | 0.111459278 | 0.114108866 | 0.125 |
| 1 | 0.210457805 | 0.228217732 | 0.25 |
| 2 | 0.362955786 | 0.456435465 | 0.5 |

```python
from math import exp, sin, sqrt, pi

def error(h, n=40000):
    step = 16/n
    def f(q):
        return exp(-q*q)/sqrt(pi)*4*sin(h*q**3/12)**2
    squared = step/3*(f(-8)+f(8)+sum(
        (4 if j % 2 else 2)*f(-8+j*step) for j in range(1, n)))
    return sqrt(squared)

for h in [.25, .5, 1, 2]:
    print(h, *(format(value, '.9g') for value in
          [error(h), sqrt(5/96)*h, h/4]))
```

![Cubic dispersion changes a Gaussian intensity profile and accumulates a norm error below the analytic bound.](/assets/learning/waves-dispersion-cubic-remainder.en.svg)

### Research questions with explanations

**Is a small gamma sufficient?** **Answer:** Specify bandwidth and time: the governing dimensionless quantity is abs(gamma t)/sigma³. The growing-horizon sequence defeats a coefficient-only argument.

**Must the centroid move at vg?** **Answer:** It does in the stated quadratic Gaussian model. Cubic dispersion shifts the spectral average derivative by gamma/(4 sigma²), even with a symmetric initial spectrum.

**Does norm conservation certify physical energy or causality?** **Answer:** No. The norm belongs to the declared lossless field model. Mechanical energy, material validity and information fronts need additional physical structure.

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
