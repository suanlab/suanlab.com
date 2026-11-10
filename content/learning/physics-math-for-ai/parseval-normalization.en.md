## Coordinate changes must account for scale

Energy here means the sum of squared sample magnitudes. It is a mathematical norm, not automatically a measured physical energy in joules. Parseval's relation says that Fourier coordinates preserve this norm with the correct normalization. The transform convention determines the factor.

For our unnormalized forward DFT and inverse divided by N,

$$
\sum_{n=0}^{N-1}|x[n]|^2=\frac1N\sum_{k=0}^{N-1}|X[k]|^2.
$$

To see why, expand the right side and interchange the finite sums. The cross term contains Σₖexp(2πik(n−m)/N). It is N when n=m and zero otherwise. The factor 1/N precisely cancels that N. No probabilistic approximation or long-record limit is involved.

## Check the eight-sample cosine

Take N=8 with one complete cosine cycle and zero DC offset. Its time energy is four. The two nonzero raw DFT coefficients are four and four, so raw spectral energy is 16+16=32. Divide by eight to obtain four. Comparing raw sums without the divisor would falsely report an eight-fold energy increase.

Switch to unitary normalization, U=X/√N. Now the two coefficients have magnitude √2 and their squared magnitudes sum directly to four:

$$
U[k]=X[k]/\sqrt N,\qquad \sum_n|x[n]|^2=\sum_k|U[k]|^2.
$$

Both modes represent the same input; changing normalization does not change its energy. The readout exposes raw displayed-spectrum energy, the required divisor and the correctly normalized total separately.

## Read contributions without matching indices

Blue stems show x[n]² at time index n. Orange stems show correctly normalized spectral energy contributions at bin k. They share an index range only for drawing convenience. The theorem equates their sums, not their pointwise values. Time sample one and frequency bin one are not the same coordinate or the same physical quantity.

The example input is cos(2πcn/N)+d. For one complete cycle, its cosine sum is zero, so adding d=1 increases energy by N. With N=8, energy becomes 4+8=12. For arbitrary fractional cycles, cross terms need not vanish; the actual sample sum remains the reference.

## Exercise and research practice

Predict the energy of a constant signal with c=0 and d=−1: the input is identically zero, so both totals vanish. Then try c=1.25 and confirm equality despite leakage. Parseval does not require peaks to land on integer bins.

Spectral losses in machine learning can differ by a factor of N or by frequency weights. Real-only FFT storage also requires special treatment of omitted conjugate bins. Before comparing an image-space loss with a Fourier-space loss, write the exact transform convention, reduction rule and weighting. Equal-looking APIs do not guarantee equal optimization scales.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
