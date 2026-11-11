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

## Derive leakage from a finite geometric sum

For the complex tone x[n]=exp(2πifn/f_s), 0≤n<M, the normalized finite-record transform at frequency ν is

$$
A_M(\nu)=\frac1M\sum_{n=0}^{M-1}e^{2\pi i(f-\nu)n/f_s}
=\frac{e^{\pi i(M-1)d}}M\frac{\sin(\pi Md)}{\sin(\pi d)},
\qquad d=\frac{f-\nu}{f_s}.
$$

Use the geometric-sum identity, with the removable limit when d is an integer. A real cosine combines two such contributions, with their complex phases; its lobes may interfere. The first zeros around an isolated complex tone are at ν=f±f_s/M. Their separation depends on the actual M-sample observation duration M/f_s.

## Zero padding changes where the spectrum is sampled

Keep M=8, f_s=8 Hz and f=1.25 Hz fixed, then pad to L≥M. The FFT samples the same finite-record transform on ν=kf_s/L. Its grid spacing is f_s/L, while the first zeros remain at .25 and 2.25 Hz.

| Padded transform length L | Grid spacing, Hz | Strongest sampled frequency, Hz | Normalized magnitude |
|---|---|---|---|
| 8 | 1 | 1 | .90176420 |
| 32 | .25 | 1.25 | 1 |
| 128 | .0625 | 1.25 | 1 |

The larger peak is a better sample of the same lobe, not new data or a narrower lobe. The code uses direct complex sums for all three padded lengths and checks the two analytic zeros. The tone is complex here to isolate one lobe; the live real cosine has conjugate components.

For the radix-two algorithm, the butterfly recurrence is B(N)=2B(N/2)+N/2 with B(1)=0. Each of log₂N levels has N/2 butterflies, proving B(N)=(N/2)log₂N. Dividing an inverse butterfly by two at each recursive level produces the required overall factor 1/N. An implementation may instead scale once at the end; these are equivalent conventions if used consistently.

**Exercise:** Does increasing L from 8 to 128 make the observation last sixteen times longer? **Explanation:** No. Only acquiring more actual samples at the same f_s increases duration; padding contributes no observations.

```python
from cmath import exp
from math import pi, sin

M,fs,f=8,8,1.25
samples=[exp(2j*pi*f*n/fs) for n in range(M)]
def slice_value(nu):
    return sum(v*exp(-2j*pi*nu*n/fs) for n,v in enumerate(samples))/M
for L in [8,32,128]:
    # Zero-pad the same eight observations. No extra observation time is added.
    padded=samples+[0j]*(L-M)
    spectrum=[sum(v*exp(-2j*pi*k*n/L) for n,v in enumerate(padded))/M for k in range(L)]
    k=max(range(L),key=lambda k:abs(spectrum[k]))
    print('FFT grid length, spacing Hz, peak Hz, magnitude:',
          L,fs/L,k*fs/L,round(abs(spectrum[k]),8))
for nu in [f-fs/M,f+fs/M]: assert abs(slice_value(nu))<1e-14
# These first zeros remain at f+-1 Hz for every padded length.
```

## Exercise and interpretation

Predict the coefficients of the constant input obtained with zero cycles and zero phase. Only X[0]=N remains, giving normalized magnitude one. Verify both algorithms and inverse reconstruction. For physical frequency labels supply an actual sample interval: bin k corresponds to kfₛ/N before signed wrapping. An index, cycles per window, radians per sample and Hz should never be interchanged without their conversion factors.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
