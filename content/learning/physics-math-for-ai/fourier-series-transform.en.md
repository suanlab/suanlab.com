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

## Positive averaging proves a different convergence statement

For the standard Fourier cutoff S_n retaining |k|≤n, define the Fejér mean σ_N=(S₀+⋯+S_{N−1})/N. It weights coefficient k by 1−|k|/N for |k|<N. Its convolution kernel is

$$
F_N(u)=\frac1N\left(\frac{\sin(Nu/2)}{\sin(u/2)}\right)^2,
\qquad F_N\ge0,\qquad\frac1{2\pi}\int_{-\pi}^{\pi}F_N=1.
$$

The value at u=0 is the removable limit N. Expanding the square as a finite sum proves the integral normalization. For δ≤|u|≤π, F_N≤1/[N sin²(δ/2)], so its mass outside any fixed neighborhood of zero tends to zero. Split the convolution error into that neighborhood and its complement. Uniform continuity controls the first part and the vanishing tail controls the second, proving uniform Fejér convergence for continuous periodic targets.

The square wave is discontinuous. Fejér averaging still stays between −1 and 1 because it averages that bounded target with nonnegative normalized weights. It is a different procedure from the displayed raw partial sum. A uniformly convergent sequence of continuous functions cannot have this discontinuous limit; more harmonics do not remove that obstruction.

## Derive the nonvanishing Gibbs overshoot on a shrinking scale

For the lab's M odd harmonics, differentiating the finite sum gives S′_M(t)=(2/π)sin(2Mt)/sin t. Integrate from zero and set y=2Mt:

$$
S_M\!\left(\frac y{2M}\right)=\frac2\pi\int_0^y
\frac{\sin u}{2M\sin(u/(2M))}\,du
\longrightarrow\frac2\pi\int_0^y\frac{\sin u}{u}\,du.
$$

The first positive maximum is at t=π/(2M). Its limiting height is about 1.1789797445, an excess .1789797445 above the upper level, or .0894898722 of the jump of size two. Do not call that excess 17.9% of the jump.

| Odd terms M | First maximum | Excess divided by jump |
|---|---|---|
| 1 | 1.27323954 | .13661977 |
| 4 | 1.18422513 | .09211257 |
| 16 | 1.17930541 | .08965270 |
| 64 | 1.17900009 | .08950005 |

![Square-wave partial sums on the rescaled coordinate y=2Mt retain an overshoot as M increases, while the finite-width transition approaches a sine-integral profile.](/assets/learning/harmonic-gibbs-scale-en.svg)

The code's M=16 and 64 and the fixed diagram extend the formula beyond the live maximum 12; they are separate examples. Quadrature computes the limiting sine integral, while exact jump coordinates 0 and ±π in the live sum remain zero instead of trigonometric roundoff.

**Exercise:** Why can a fixed positive t converge to one while this first maximum does not? **Explanation:** The maximizer moves toward the jump as M grows; it is not a fixed point.

```python
from math import sin, pi

def simpson(f,a,b,n=20000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))/3

si_pi=simpson(lambda u: 1 if u==0 else sin(u)/u,0,pi)
limit=2*si_pi/pi
for m in [1,4,16,64]:
    t=pi/(2*m)
    peak=sum(4*sin((2*j+1)*t)/(pi*(2*j+1)) for j in range(m))
    print('m, first peak, overshoot / jump:',m,round(peak,8),round((peak-1)/2,8))
print('scaled limit:',round(limit,10))
# Standard Fejer mean: weights are 1-|k|/N, not an unweighted partial sum.
N=32
def fejer(t):
    return sum(4*(1-k/N)*sin(k*t)/(pi*k) for k in range(1,N,2))
assert max(abs(fejer(-pi+2*pi*j/2000)) for j in range(2001))<=1+1e-12
# The finite grid is a check; the nonnegative convolution kernel proves the global bound.
```

## Exercise and AI connection

Predict F(0) when σ doubles: it doubles because area doubles. Then verify the readout. Next explain why the square wave cannot be represented by finitely many smooth sines exactly. Spectral layers and positional features use related oscillatory coordinates, but their performance still depends on sampling, boundaries and the learned operation. Ordinary integrable transforms, L² extensions and distributional transforms have different existence statements; do not apply this Gaussian calculation to every signal without specifying one.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
