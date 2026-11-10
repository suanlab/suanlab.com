## Ask when a frequency appears

A single global spectrum discards the location of a local change. Here a 32-sample signal uses cosine bin two in its first half and bin six in its second half. Moving a window lets us ask which oscillation is present near a selected center. The boundary is explicitly zero extended when a block reaches outside the available record.

For block length L, start=center−L/2 and symmetric Hann weights, the local Fourier calculation is

$$
w[j]=\tfrac12-\tfrac12\cos\frac{2\pi j}{L-1},\qquad
S_c[k]=\sum_{j=0}^{L-1}x[\mathrm{start}+j]w[j]
 e^{-2\pi i k(\mathrm{start}+j)/32}.
$$

The global index in the exponential fixes the phase convention. A local-index STFT would have the same magnitudes and different phase. The chart displays |S|/Σw on the fixed 32-bin grid, not an unnormalized spectrum or a general unbiased sinusoid-amplitude estimator.

## Compare two windows

With center eight and L=16, the source is bin two, but the largest nonnegative displayed bin is one: this block contains only one cycle, and positive/negative Hann lobes overlap. Magnitudes at bins zero, one and two are about 0.5365, 0.5371 and 0.5119. A peak bin is not automatically the true tone frequency. Move to center 24 and the dominant bin becomes six. Near center 16, the window mixes both halves. Shorter windows localize the change more tightly but broaden spectral features. Padding to 32 bins samples the windowed spectrum more densely; it does not create the resolving power of 32 nonzero observations.

The table exposes complex coefficients and the readout compares Σ|S|²/32 with the energy of the windowed input. This Parseval check uses the full spectrum, whereas the reported dominant bin searches only bins zero through sixteen.

## Haar uses localized averages and differences

Switch to Haar mode. Apply an orthonormal two-point transform recursively to the unwindowed block:

$$
a=(x_0+x_1)/\sqrt2,\qquad d=(x_0-x_1)/\sqrt2,
\qquad a^2+d^2=x_0^2+x_1^2.
$$

Transform the averages again until one coarse coefficient remains. Output order is the coarsest scaling coefficient followed by detail groups from coarse to fine. Unlike the STFT chart, coefficient index is not a frequency bin. The recursive transform preserves the unwindowed block energy, not the Hann-windowed energy. A selected index beyond L has no Haar coefficient and is displayed as undefined.

## Exercise and limits

For block [1,1], the transform is [√2,0]. For [1,−1], it is [0,√2]. Explain how a difference detects a local transition while an average captures a coarse level. Then inspect a window crossing the record edge and count padded samples.

Wavelet and STFT methods support multiscale features in audio and scientific AI. This lab implements a discrete orthonormal Haar transform and individual STFT frames. It does not implement every wavelet family or an overlap-add inverse. Exact STFT reconstruction additionally requires compatible analysis/synthesis windows and hop rules, which must be checked separately.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
