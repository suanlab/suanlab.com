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

## Extend the identity without claiming every Fourier integral exists

For Schwartz f,g, Fourier inversion for g and Fubini give the cross-inner-product identity

$$
\int_{\mathbb R}f\overline g
=\frac1{2\pi}\int_{\mathbb R}F\overline G,
\qquad \|f\|_2^2=\frac1{2\pi}\|F\|_2^2.
$$

These interchanges are justified by rapid decay. Inversion itself can be obtained by inserting a Gaussian frequency cutoff, identifying convolution with an approximate identity and taking the cutoff to its limit. The normalized map f↦F/√(2π) is therefore an isometry on a dense subspace. Extend it by L² completion; inversion extends similarly and gives Plancherel on L². The resulting transform need not be an absolutely convergent integral for an arbitrary L² representative.

Likewise, completeness of trigonometric polynomials on the circle upgrades Bessel's inequality to Parseval. Fejér approximation proves density for continuous periodic functions, which are dense in circle L². This distinguishes a finite orthogonal DFT identity from an infinite-dimensional completeness theorem.

## Weight the missing real-spectrum conjugate bins

For a real vector with even N, X[N−k]=conj(X[k]). If only bins 0 through N/2 are stored,

$$
\sum_n|x[n]|^2=\frac1N\left(|X[0]|^2+|X[N/2]|^2
+2\sum_{k=1}^{N/2-1}|X[k]|^2\right).
$$

DC and the Nyquist bin are singletons; only the interior bins receive weight two. For x[n]=1+cos(2πn/N), unweighted stored-bin energy misses half of the cosine contribution.

| N | Time energy | Unweighted stored spectrum / N | Correct weighted energy |
|---|---|---|---|
| 4 | 6 | 5 | 6 |
| 8 | 12 | 10 | 12 |
| 16 | 24 | 20 | 24 |

The code uses direct complex sums and checks the weights. For a time-domain MSE of a real residual vector, divide the weighted raw spectral sum by N², not N. Frequency weighting other than this conjugate accounting changes the loss rather than merely changing coordinates.

**Exercise:** Can all stored real-spectrum bins be doubled? **Explanation:** No. That double-counts the unpaired DC and even-N Nyquist components. Odd N needs its own endpoint rule.

```python
from cmath import exp
from math import cos, pi

for N in [4,8,16]:
    x=[1+cos(2*pi*n/N) for n in range(N)]
    X=[sum(v*exp(-2j*pi*k*n/N) for n,v in enumerate(x)) for k in range(N)]
    time=sum(v*v for v in x)
    wrong=sum(abs(v)**2 for v in X[:N//2+1])/N
    corrected=(abs(X[0])**2+abs(X[N//2])**2+
               2*sum(abs(X[k])**2 for k in range(1,N//2)))/N
    assert abs(time-1.5*N)<1e-12 and abs(corrected-time)<1e-12
    print('N, time energy, unweighted rFFT energy, corrected:',N,
          round(time,6),round(wrong,6),round(corrected,6))
# For time MSE rather than total energy, divide the corrected expression once more by N.
```

## Exercise and research practice

Predict the energy of a constant signal with c=0 and d=−1: the input is identically zero, so both totals vanish. Then try c=1.25 and confirm equality despite leakage. Parseval does not require peaks to land on integer bins.

Spectral losses in machine learning can differ by a factor of N or by frequency weights. Real-only FFT storage also requires special treatment of omitted conjugate bins. Before comparing an image-space loss with a Fourier-space loss, write the exact transform convention, reduction rule and weighting. Equal-looking APIs do not guarantee equal optimization scales.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
