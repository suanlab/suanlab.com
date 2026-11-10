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

## Exercise

Predict the signed alias of 9 Hz at eight samples/s: it is +1 Hz. Test phase zero and then a quarter turn. Explain why increasing the display resolution does not resolve the ambiguity, whereas increasing the actual sampling rate can. In research reports distinguish sample rate, observation duration, DFT bin spacing and plot pixel count; these are four different quantities.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
