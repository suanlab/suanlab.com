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

## Derive an inverse-window condition instead of assuming it

Suppose every analysis frame contains x[n]w[n−mR] and its complete inverse DFT is multiplied by a synthesis window v[n−mR]. Adding frames gives

$$
y[n]=x[n]D[n],\qquad D[n]=\sum_m w[n-mR]v[n-mR].
$$

Perfect reconstruction with no spectral modification requires D[n]=1, or division by a known nonzero D[n]. If a coordinate has D[n]=0 with identical real analysis/synthesis windows, every covering window is zero there and information is lost. A valid dual uses v_m[n]=w[n−mR]/Σ_j w²[n−jR] wherever the denominator is positive.

The code uses a separate periodic 16-sample record, length-eight windows, full frame transforms and overlap-add. It does not silently change the live zero-extended single-frame lab.

| Window applied at analysis and synthesis | Hop R | Minimum Σw² | Maximum Σw² |
|---|---|---|---|
| Periodic Hann, denominator L | 4 | .5 | 1 |
| Symmetric Hann, denominator L−1 | 4 | .409079 | .903421 |
| Square root of periodic Hann | 4 | 1 | 1 |
| Periodic Hann without overlap | 8 | 0 | 1 |

Half-overlapped periodic Hann has Σw=1, but applying it twice gives nonconstant Σw². Dividing by the positive overlap sum or using the square-root pair corrects this. The symmetric Hann used by the live lab does not inherit the periodic formula merely because both are called Hann.

## Connect the finite Haar transform to an L² basis

On [0,1), take φ=1 and ψ=1 on the first half, −1 on the second half. The functions ψ_jk(t)=2^(j/2)ψ(2^jt−k), supported on their dyadic cells, have norm one and zero integral. Disjoint supports give same-level orthogonality; a finer wavelet has zero integral where any coarser wavelet is constant, giving cross-level orthogonality. Include φ together with j≥0 and 0≤k<2^j.

Their span contains every dyadic step function. These are dense in L²[0,1], for example by approximating continuous functions uniformly on increasingly fine dyadic cells and using continuous-function density. Thus this is a complete orthonormal basis on the interval. The finite recursive average/detail transform is its finite-dimensional counterpart; a coefficient index is a scale/location coordinate, not a Fourier frequency.

**Exercise:** Is a good STFT forward spectrum enough to prove exact inverse reconstruction? **Explanation:** No. Window coverage, synthesis weights, hop size and boundary handling determine the inverse condition.

```python
from math import cos, sin, pi, sqrt
from cmath import exp

N,L=16,8
signal=[1+sin(.4*n) for n in range(N)]
def dft(x,inverse=False):
    n=len(x)
    return [sum(v*exp((1 if inverse else -1)*2j*pi*k*j/n) for j,v in enumerate(x))/(n if inverse else 1) for k in range(n)]
for name,hop,window in [
    ('periodic Hann',4,[.5-.5*cos(2*pi*j/L) for j in range(L)]),
    ('symmetric Hann',4,[.5-.5*cos(2*pi*j/(L-1)) for j in range(L)]),
    ('root periodic Hann',4,[sqrt(.5-.5*cos(2*pi*j/L)) for j in range(L)]),
    ('periodic Hann, no overlap',8,[.5-.5*cos(2*pi*j/L) for j in range(L)])]:
    output=[0.0]*N
    D=[0.0]*N
    for start in range(0,N,hop):
        frame=[signal[(start+j)%N]*window[j] for j in range(L)]
        restored=dft(dft(frame),True)
        for j in range(L):
            index=(start+j)%N
            output[index]+=restored[j].real*window[j]
            D[index]+=window[j]**2
    if min(D)>1e-14:
        recovered=[a/b for a,b in zip(output,D)]
        assert max(abs(a-b) for a,b in zip(recovered,signal))<1e-13
    else: assert D[0]==0  # An input coordinate is unobserved by all analysis windows.
    print(name,round(min(D),6),round(max(D),6),min(D)>1e-14)
# Periodic framing here is separate from the live lab's finite zero-extended single frame.
```

## Exercise and limits

For block [1,1], the transform is [√2,0]. For [1,−1], it is [0,√2]. Explain how a difference detects a local transition while an average captures a coarse level. Then inspect a window crossing the record edge and count padded samples.

Wavelet and STFT methods support multiscale features in audio and scientific AI. This lab implements a discrete orthonormal Haar transform and individual STFT frames. It does not implement every wavelet family or an overlap-add inverse. Exact STFT reconstruction additionally requires compatible analysis/synthesis windows and hop rules, which must be checked separately.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
