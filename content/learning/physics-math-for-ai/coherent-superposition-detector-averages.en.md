## A detector does not observe a signed wave

Two fields can each be nonzero while their sum vanishes. Conversely, two equal fields in phase have twice the amplitude and four times the normalized intensity of either field. The distinction is between a linear field and a quadratic observation. Adding the two separate intensities discards the phase information responsible for interference.

We will keep three questions separate: what field exists at a selected time, what a detector averages over a finite exposure, and what an ensemble of phase realizations predicts. They describe different operations. This lesson concerns independently prepared scalar waves, rather than energy exchange between spring-coupled masses.

## Units, normalization and physical assumptions

Assume linear superposition and the same polarization for both fields. Write dimensionless complex amplitudes $a, b\ge0$. A normalized intensity is a squared field magnitude. Converting it to optical irradiance in W/m² additionally requires an amplitude calibration, a specified medium and the appropriate cycle-average factor. It is not the mechanical string energy from the preceding lesson.

Angular frequencies multiply a chosen time unit to produce radians. For calculations labeled in seconds, their units are rad/s. Slow display frequencies illustrate the mathematics; they are not the physical frequencies of visible light. Orthogonal polarization, nonlinear detection and quantum photon statistics are separate subjects.

## Complex fields and actual real fields

For one phase realization $\theta$, use

$$
U(t)=a e^{-i\omega_1t}+b e^{i(\phi+\theta+\chi)}e^{-i\omega_2t},
\qquad E(t)=\operatorname{Re}U(t).
$$

Here $\phi$ is a controlled phase and $\chi$ an additional spatial path phase. A phasor is a convenient representation; its imaginary component is not an additional transverse displacement of the material. The actual real field is $E$.

At the exposure center $t_0$, define $\Delta\omega=\omega_2-\omega_1$ and $\psi=\phi+\chi-\Delta\omega t_0$. The instantaneous normalized intensity is $a^2+b^2+2ab\cos(\psi+\theta)$. This quantity differs from $E(t_0)^2$ because it includes the squared imaginary component as well.

## Phase populations are ensembles

Represent possible phases by $\theta_j$ with nonnegative weights $w_j$ and positive total weight. Normalize $p_j=w_j/\sum_kw_k$. An ensemble prediction averages observations, not amplitudes:

$$
\mathbb E|U|^2=\sum_jp_j|U_j|^2,
\qquad \left|\mathbb E U\right|^2\ne\mathbb E|U|^2
$$

in general. For $a=b=1$ and equally weighted phases0 and $\pi$, the mean second field is zero. Squaring the mean total field gives1, whereas the mean normalized intensity is2. Fluctuating realizations have been lost by averaging the field first.

A finite phase population is an explicit teaching distribution, not a claim to simulate every stochastic optical source. A single realization should not be presented as the unique physical field of a partially coherent ensemble.

## Derive the interference cross term

Expanding the complex magnitude gives

$$
|U_1+U_2|^2=|U_1|^2+|U_2|^2+
2\operatorname{Re}(U_1U_2^*).
$$

The cross term can be negative without making total intensity negative. With equal amplitudes and phase difference $\pi$, it cancels the two separate intensity terms. Changing a path changes relative phase and therefore the observation, even if the separate branch intensities stay fixed.

This does not create energy from nothing. The detector measures a particular spatial field and the source/boundary arrangement determines its full energy distribution. A two-phasor calculation at one observation point is not a complete global energy budget.

## Coherence and its normalization

Define the population phase moment

$$
M=\sum_jp_j e^{i\theta_j}.
$$

It describes phase alignment. Its magnitude is bounded by1 mathematically. Our convention for mutual coherence is $\gamma_{12}=\langle U_1U_2^*\rangle/(ab)$, so its imaginary sign is the conjugate of a moment defined with $e^{+i\theta}$. Stating the convention prevents a correct phase shift from acquiring the wrong sign.

If either branch has zero intensity, the normalized mutual coherence is undefined. A population moment can still be calculated, but it must not be labeled as measured mutual coherence of a missing field. If both branches vanish, fringe visibility is also undefined.

## Finite detector exposure

Average over $[t_0-T/2, t_0+T/2]$. Integrating the relative phase yields

$$
S=\operatorname{sinc}(\Delta\omega T/2),\qquad
\operatorname{sinc}(x)=\frac{\sin x}{x},\quad \operatorname{sinc}(0)=1,
$$

$$
\overline I=a^2+b^2+2abS\operatorname{Re}(e^{i\psi}M).
$$

An exposure of zero duration means the instantaneous limit. Equal frequencies have $S=1$ for every exposure: a stable phase difference survives averaging. Different frequencies can wash out the cross term even if their phases were prepared deterministically. This exposure effect is distinct from random population phases.

## A negative sinc changes fringe phase

The factor $S$ is not always positive. With $\Delta\omega=2\pi$ and $T=1.5$, $S=\sin(1.5\pi)/(1.5\pi)<0$. A phase that was bright for a short exposure can become darker than the incoherent baseline, while the phase shifted by $\pi$ becomes brighter.

The sign reverses the modulation; it does not imply negative total light. For a phase scan, contrast depends on $|S|$ while the sign changes the location of the maxima. Replacing $S$ with its absolute value in the intensity formula would put the fringes in the wrong place.

## Real-field averaging retains other frequencies

For a particular realization, let $\alpha_1=-\omega_1t_0$ and $\alpha_2=\phi+\theta+\chi-\omega_2t_0$. Directly averaging the actual real square gives

$$
\begin{aligned}
\overline{E^2}={}&\frac{a^2}{2}[1+\operatorname{sinc}(\omega_1T)\cos(2\alpha_1)]\\
&+\frac{b^2}{2}[1+\operatorname{sinc}(\omega_2T)\cos(2\alpha_2)]\\
&+ab\left[S\cos(\alpha_1-\alpha_2)
+\operatorname{sinc}((\omega_1+\omega_2)T/2)\cos(\alpha_1+\alpha_2)\right].
\end{aligned}
$$

Only after the fast double-frequency and sum-frequency terms average away does this become half the analytic-signal intensity. For $a=b=1$, equal frequency $2\pi$ and a one-second exposure, $\overline I=4$ but $\overline{E^2}=2$. At DC with relative phase $\pi/2$, the two values are2 and1 without any cycle to average.

## Preserve small dark-fringe values numerically

Subtracting large rounded bright terms can destroy a small positive intensity. A more stable coherent expression is

$$
I(\alpha)=(a-b)^2+4ab\cos^2(\alpha/2).
$$

The finite-exposure result is a nonnegative combination of this expression and its opposite phase, with weights $(1+S)/2$ and $(1-S)/2$. Population averaging preserves nonnegativity. No arbitrary epsilon floor turns a dark fringe into invented signal.

The calculator separately integrates the nonnegative real-field square with composite Gaussian quadrature and compares a refinement and the closed expression. A difference between refinements is a numerical diagnostic, not a certified error bound. Near exact cancellation, the closed expression itself can lose precision; the independently integrated physical square remains nonnegative.

The numerical average uses a unit exposure coordinate $t=t_0+Tv$,with $v\in[-1/2,1/2]$. It integrates the square in $v$ directly,so it does not subtract two nearly equal absolute times or divide a collapsed interval by $T$. For example,$T=10^{-18}$ around $t_0=16$ must approach the instantaneous value rather than a false zero. A separate70-decimal reference verifies this limit.

## Visibility requires a specified scan

For a complete relative-phase scan at fixed amplitudes, population and exposure,

$$
V=\frac{I_{\max}-I_{\min}}{I_{\max}+I_{\min}}
=\frac{2ab|SM|}{a^2+b^2}
$$

when the denominator is nonzero. A crop of a spatial pattern may not contain the true extrema, so its observed range need not estimate this visibility. A missing branch has zero contrast but undefined normalized mutual coherence; complete darkness has no defined contrast.

The implementation evaluates the two opposite extremal phases using positive intensity expressions. Their larger and smaller values define the observed range, avoiding a spurious negative minimum from subtracting rounded large quantities.

## Worked nonidentifiability example

Balanced branch intensities $I_1=I_2=1$ and phase weights3:1 at0 and $\pi$ give $M=1/2$ and

$$
I(\phi)=2+\cos\phi,\qquad V=1/2.
$$

Now choose fully coherent fields but unequal intensities

$$
I_1=\frac{2+\sqrt3}{2},\qquad I_2=\frac{2-\sqrt3}{2}.
$$

Their sum is2 and $2\sqrt{I_1I_2}=1$, producing the same entire fringe curve. Even a perfect measurement of this curve cannot identify coherence without additional branch information. Measuring each branch separately can remove this particular ambiguity.

## Worked exposure versus population contrast

Keep unit amplitudes and the3:1 population at0 and$\pi$,so the intrinsic phase moment remains1/2. Choose $\omega_1=2\pi$, $\omega_2=4\pi$, $t_0=0$ and $T=1.5$. Now $S=-2/(3\pi)$. The normalized intensity at controlled phase0 is $2+S\approx1.78779$; at phase$\pi$ it is $2-S\approx2.21221$. The scanned visibility is $|S|/2\approx0.106103$,not the population-only contrast1/2.

At exposure1,the difference-frequency contribution averages to zero although the phase population has not changed. A missing fringe can therefore result from the detector window rather than a different source ensemble. For exposure1.5,the actual real-square mean at phase0 is about0.929264,whereas half the analytic intensity is about0.893897: a sum-frequency term has not yet averaged away. Increasing exposure is an experimental change,not just smoothing the displayed line.

## Model controls and independent verification

The underlying teaching model supports amplitudes0–2, frequencies0–16, time centers−16–16, exposures0–8 and1–32 finite phase populations. Weights0–16 are normalized; their total must be positive. These bounds control the demonstration, not a universal optical bandwidth.

Independent reference calculations integrate actual complex magnitudes and real squares with SciPy. Three separately averaged phase settings recover modulation and contrast without calling the product's formula. Tests also cover zero exposure, absent fields, negative sinc, nearly dark fringes, uniform phases and the nonidentifiability example. For an AI surrogate, matching one bright fringe is weak evidence: test changes of exposure, branch amplitude and population independently.

## Exercises with explanations

### Exercise 1: Equal fields

Two unit-amplitude fields are in phase. What is normalized intensity?

**Solution:** The total amplitude is2, so intensity is4. The separate intensities sum to2; the remaining2 is the cross term.

### Exercise 2: Average before squaring?

Use equal phase weights at0 and $\pi$ with $a=b=1$. Compare the two averaging orders.

**Solution:** The mean second field vanishes, so the squared mean total is1. Averaging the two total intensities,4 and0, gives2. The operations differ because squaring is nonlinear.

### Exercise 3: Zero exposure

Does $T=0$ remove the signal?

**Solution:** No. It selects the instantaneous limit with $S=1$. Implementing division by $T$ directly would fail; evaluate the continuous limit instead.

### Exercise 4: Missing branch

If $b=0$ and $a>0$, what are visibility and mutual coherence?

**Solution:** Intensity is phase-independent, so visibility is0. Normalized mutual coherence divides by $ab=0$ and is undefined. These are different statements.

### Exercise 5: Finite exposure reversal

What goes wrong if a negative $S$ is replaced by $|S|$ everywhere?

**Solution:** Contrast stays the same but bright and dark phase locations are exchanged. The signed factor belongs in the intensity formula.

### Exercise 6: Identify coherence from contrast

Does $V=1/2$ prove a partially coherent source?

**Solution:** No. Balanced partially coherent fields and the fully coherent unequal intensities in the worked example produce the same curve. Measure branch intensities or impose justified additional assumptions.

## Primary framework and next connection

[TU Delft's interference and coherence chapter](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap5_Interference/InterferenceCoherence_2022_01Clean.html) supplies the complex-field, observation and coherence framework. The explicit population, finite-window comparison, positive numerical form and independent verification here follow from the stated experiment.

The next lesson combines contributions from a finite aperture. Its pattern depends on both the aperture and propagation approximation; two ideal point phasors alone do not describe a finite slit envelope.
