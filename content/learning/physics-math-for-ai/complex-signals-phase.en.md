## Encode an oscillation by its complex amplitude

An equal-frequency sinusoid A cos(ωt+φ) is the real part of Ae^{iφ}e^{iωt}. The phasor separates amplitude and phase from common rotation; it does not imply the measured signal must be complex.

$$
s(t)=\operatorname{Re}[(e^{i\phi_1}+A_2e^{i\phi_2})e^{it}].
$$

The lab uses dimensionless phase time with ω=1. Different frequencies generally cannot share one constant combined phasor.

## Add amplitudes as complex numbers

Compute S=e^{iφ₁}+A₂e^{iφ₂} in Cartesian components, then recover magnitude and argument:

$$
|S|^2=1+A_2^2+2A_2\cos(\phi_2-\phi_1).
$$

Two in-phase unit signals give amplitude two. Equal opposite phases cancel. A 90-degree difference gives magnitude √2 and phase 45 degrees. The cross term encodes interference; adding scalar magnitudes alone loses it.

## Zero has no phase

At S=0, the signal vanishes for every t and its argument is undefined. A floating-point sinπ residual should not manufacture a meaningful phase. The model recognizes exact equal-amplitude, 180-degree input cancellation analytically.

Near cancellation is kept distinct: a small nonzero amplitude is not erased with an arbitrary tolerance. Phase becomes sensitive to noise and rounding when magnitude is tiny, so inspect both together.

## Interpret the time plot

Blue is the first signal; orange is the combined signal, not the second component alone. The sum is evaluated as

$$
\operatorname{Re}[(a+ib)e^{it}]=a\cos t-b\sin t.
$$

Exact cancellation produces a flat zero curve. Degrees are converted to radians for trigonometry. These time–amplitude axes differ from a phasor’s complex-plane axes.

## Power, sampling and AI applications

The readout |S|² is squared amplitude. Physical power needs impedance, units and peak/RMS conventions. A unitless number is not automatically watts. A cosine’s squared time average is one half, so peak amplitude squared differs from mean square.

Fourier features, audio, radar and phase-aware models use this algebra. Sampling, aliasing and frequency alignment remain separate assumptions; finite plotted samples do not guarantee reconstruction of arbitrary signals.

## Exercises and answers

1. Unit phases zero and 90 degrees combine to **√2 at 45 degrees**.
2. Zero amplitude has **undefined phase**.
3. With peak-cosine convention, |S|²=4 gives **mean square two**, not four.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
