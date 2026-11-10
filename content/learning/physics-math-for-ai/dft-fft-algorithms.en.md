## One transform, two algorithms

The discrete Fourier transform maps N complex samples into N complex coefficients. Our forward convention has no normalization and our inverse divides by N:

$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-2\pi i kn/N},\qquad
x[n]=\frac1N\sum_{k=0}^{N-1}X[k]e^{2\pi i kn/N}.
$$

An FFT computes this same transform through a factorization. It does not introduce a new frequency representation or improve the information content of the input. The experiment calculates both direct DFT and radix-two FFT and compares their complex coefficients before drawing magnitudes.

## A cosine gives a useful numerical check

With N=8, one cycle per window and phase zero, x[n]=cos(2πn/8). Euler's identity splits it into equal positive and negative frequency exponentials. The nonzero coefficients are X[1]=4 and X[7]=4. The plotted two-sided magnitudes divide by N, so both peaks have height 0.5. They are not a one-sided amplitude display with doubled interior bins.

The table includes signed cycles per window: bin seven represents −1. Bin N/2 is the single Nyquist bin for even N; it must not be counted twice. Altering phase rotates complex coefficients, which a magnitude-only graph can conceal. Inspect the real and imaginary columns as well.

## How radix two saves work

Separate even and odd input indices and transform each half:

$$
X[k]=E[k]+W_N^kO[k],\qquad
X[k+N/2]=E[k]-W_N^kO[k],\quad W_N=e^{-2\pi i/N}.
$$

Each pair is a butterfly. For N=8 there are (N/2)log₂N=12 butterflies, compared with 64 terms in the direct sum. At N=64 the counts are 192 and 4096. These are algorithmic counts with different units, not measured wall-clock ratios or equal-cost machine operations. Implementations also depend on memory access, constants and hardware.

## A fractional cycle spreads across bins

Set cycles per window to 1.25. Periodic extension of that observed window generally has a boundary mismatch, and energy spreads over multiple DFT bins. This leakage is not evidence that the FFT is inaccurate: the direct DFT agrees. Increasing N while keeping cycles per window fixed changes sampling of the normalized window; it does not automatically represent a longer observation of the same physical tone.

## Exercise and interpretation

Predict the coefficients of the constant input obtained with zero cycles and zero phase. Only X[0]=N remains, giving normalized magnitude one. Verify both algorithms and inverse reconstruction. For physical frequency labels supply an actual sample interval: bin k corresponds to kfₛ/N before signed wrapping. An index, cycles per window, radians per sample and Hz should never be interchanged without their conversion factors.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
