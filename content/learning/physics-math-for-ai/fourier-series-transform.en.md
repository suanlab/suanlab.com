## Why add waves?

A signal can be described by its values at each time or by contributions from oscillations. These descriptions answer different questions: when does the signal change, and which oscillation scales are needed? Fourier analysis changes coordinates through inner products. It does not make nonlinear interactions disappear.

Start with a 2π-periodic square wave: −1 on (−π,0), +1 on (0,π), with midpoint value zero at jumps. Odd symmetry removes cosine terms. Integrating against sine leaves only odd harmonics:

$$
b_k=\frac1\pi\int_{-\pi}^{\pi}f(t)\sin(kt)\,dt,
\qquad S_M(t)=\frac4\pi\sum_{j=0}^{M-1}\frac{\sin((2j+1)t)}{2j+1}.
$$

With one term the approximation is a smooth sine with peak 4/π, about 1.273. Four terms include frequencies 1,3,5,7; twelve terms extend through 23. More terms describe sharper transitions by adding faster oscillations.

## Read the discontinuity honestly

Blue horizontal segments show the target; open endpoints mark excluded one-sided values, and filled dots mark midpoint values. Orange shows the partial sum. At t=0 every sine vanishes, so every partial sum equals zero. For this piecewise smooth periodic target, the limit at each jump is the average of the one-sided limits.

Convergence is not uniform across the jump. Oscillations concentrate into a narrower neighborhood, while the overshoot does not vanish in the Gibbs limit. The displayed overshoot is a maximum over a finite grid, not an exact evaluation of that limit. Moving a query close to zero and increasing the term count illustrates why pointwise and uniform convergence differ.

## A transform for a nonperiodic pulse

Switch to Gaussian mode. For an integrable Gaussian and angular frequency ω, our convention is

$$
f(t)=e^{-t^2/(2\sigma^2)},\qquad
F(\omega)=\int_{\mathbb R}f(t)e^{-i\omega t}\,dt
=\sigma\sqrt{2\pi}e^{-\sigma^2\omega^2/2}.
$$

At σ=1, F(0)=√(2π), about 2.507, equal to the time-domain area. Increasing σ broadens the pulse and narrows its spectrum. The overlay uses time for blue and angular frequency for orange; matching horizontal positions are not matching physical units. Harmonic count is unused in this mode.

## Exercise and AI connection

Predict F(0) when σ doubles: it doubles because area doubles. Then verify the readout. Next explain why the square wave cannot be represented by finitely many smooth sines exactly. Spectral layers and positional features use related oscillatory coordinates, but their performance still depends on sampling, boundaries and the learned operation. Ordinary integrable transforms, L² extensions and distributional transforms have different existence statements; do not apply this Gaussian calculation to every signal without specifying one.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
