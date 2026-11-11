## Samples can hide a different continuous signal

A sampler records values at t=n/fₛ. It does not directly record what happened between those instants. Consider a cosine with frequency f in Hz and phase φ. Adding any integer multiple of the sampling rate changes its phase at sample instants by an integer multiple of 2π:

$$
x[n]=\cos(2\pi f n/f_s+\phi)
=\cos(2\pi(f-mf_s)n/f_s+\phi),\quad m\in\mathbb Z.
$$

This identity is the source of alias ambiguity. A denser plotted line cannot remove information already lost in sampling. The experiment selects a signed representative in [−fₛ/2,fₛ/2), preserving phase.

## A seven-Hz tone at eight samples per second

With f=7 Hz and fₛ=8 samples/s, the signed alias is −1 Hz. At n=1 both signals give cos(7π/4)=cos(−π/4)=√2/2. At n=2 both give zero. The entire sample table agrees even though the continuous blue seven-Hz curve and orange negative-one-Hz curve differ between dots.

A negative frequency here is a convenient signed complex-exponential coordinate. For a real cosine it can be exchanged for positive frequency by reversing phase. Keeping the signed value makes the same-phase alias identity explicit rather than silently changing phase in the interface.

## Why the Nyquist boundary deserves its own example

Set f=4, fₛ=8 and phase quarter-turn one, meaning φ=π/2. Every sample is zero:

$$
\cos(\pi n+\pi/2)=0,\qquad n\in\mathbb Z.
$$

A nonzero continuous waveform has therefore become indistinguishable from zero at these instants. The readout's strictly-below-Nyquist indicator is false at equality. It should not become true merely because the numerical samples are finite or because one particular phase happens to look recoverable.

## Conditions for reconstruction

The ideal sampling theorem concerns an entire band-limited signal under suitable function-space assumptions, with sampling strictly above twice its highest frequency for the usual unambiguous setting. It also assumes exact samples over an unbounded sequence and ideal interpolation. This finite table of one tone is an illustration of ambiguity, not an implementation of that full theorem.

Before downsampling audio, images or sensor streams, suppress out-of-band components with an anti-alias filter. Filtering after the sample ambiguity has occurred cannot generally identify which original frequency generated the data.

## Prove sinc reconstruction in a specified function space

Let f∈L²(R) have Fourier transform supported in [−πf_s,πf_s] in the angular convention. This compactly supported L² transform is also L¹, so its inverse defines a continuous representative of f; arbitrary null-set changes are not the samples. Expand F in the orthogonal exponentials e^(−iωn/f_s) on that band. Its coefficient is f(n/f_s)/f_s by inverse Fourier transformation. Inverting the expansion yields

$$
f(t)=\sum_{n\in\mathbb Z}f(n/f_s)\operatorname{sinc}(f_st-n),
\qquad \operatorname{sinc}(u)=\frac{\sin(\pi u)}{\pi u},\quad\operatorname{sinc}(0)=1.
$$

Convergence of the band expansion in L² gives L² convergence in time by Plancherel. Cauchy–Schwarz on the fixed finite band also controls point evaluation of the inverse error, yielding uniform convergence for symmetric partial sums. The complete infinite sample sequence and the exact band restriction matter.

At the critical rate f_s=2B this theorem remains injective for this L² bandlimited class: band endpoints have measure zero. The live Nyquist sine counterexample has infinite energy on R and lies outside that class. Allowing pure tones or spectral point masses at the boundary changes the problem. A strict guard band is a useful sufficient engineering condition, but it is not the necessity statement for every version of a sampling theorem.

## Truncate a genuine critical-rate reconstruction

The separate signal f(t)=sinc(2t−.3) is L², has B=1 Hz and is sampled at f_s=2 Hz. At t=.25 its exact value is about .935489284. The code compares an independently integrated rectangular spectrum with the sinc formula, then truncates to n=−N,…,N.

| Half-count N | Reconstructed value | Absolute truncation error |
|---|---|---|
| 4 | .898916028 | .036573255 |
| 16 | .925550542 | .009938742 |
| 64 | .932947512 | .002541772 |
| 256 | .934850136 | .000639147 |

These counts exceed the live one-second tone table; no finite table is being called an exact infinite reconstruction. In the live alias calculation, reducing f modulo f_s before selecting its signed representative preserves tiny frequencies such as 10⁻²⁰ Hz. Quarter-turn phase is applied as an exact rotation, avoiding loss of a tiny sinusoidal component when adding it to π/2. Exact Nyquist quarter-phase samples remain zero.

**Exercise:** Does an L² function have well-defined sample values without any other assumption? **Explanation:** No. Here compact spectral support selects its continuous inverse-transform representative.

```python
from math import sin, cos, pi

def sinc(u): return 1.0 if u==0 else sin(pi*u)/(pi*u)
def simpson(f,a,b,n=10000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))/3

# f(t)=sinc(2t-0.3), band limit B=1 Hz, sampling rate fs=2 Hz.
t=.25
truth=sinc(2*t-.3)
# Inverse Fourier integral in cycles per second: spectrum is a shifted rectangle / 2.
reference=simpson(lambda nu:.5*cos(2*pi*nu*(t-.15)),-1,1)
assert abs(reference-truth)<1e-12
for N in [4,16,64,256]:
    recovered=sum(sinc(n-.3)*sinc(2*t-n) for n in range(-N,N+1))
    print('half-count, value, error:',N,round(recovered,9),round(abs(recovered-truth),9))
# sinc is L2 but not L1; use the L2 bandlimited theorem and its continuous representative.
```

## Exercise

Predict the signed alias of 9 Hz at eight samples/s: it is +1 Hz. Test phase zero and then a quarter turn. Explain why increasing the display resolution does not resolve the ambiguity, whereas increasing the actual sampling rate can. In research reports distinguish sample rate, observation duration, DFT bin spacing and plot pixel count; these are four different quantities.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
