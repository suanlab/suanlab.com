## A filter responds differently to different tones

Consider the two-tap finite impulse response h=[1−r,r], with r between zero and one. It combines the current input and a one-sample-delayed input. With an input defined for every integer time, its steady-state response is

$$
y[n]=(1-r)x[n]+rx[n-1],\qquad
H(e^{i\omega})=1-r+re^{-i\omega}.
$$

Substitute x[n]=exp(iωn) to derive the second expression: the common exponential factors out, leaving H. Complex exponentials are eigenfunctions of a shift-invariant linear system; real cosines inherit magnitude scaling and phase shift.

## Interpret magnitude and phase numerically

At r=0.5 and ω=π/2, H=(1−i)/2. Its magnitude is 1/√2, about 0.707, and phase is −π/4. A cosine input therefore produces an attenuated cosine shifted by that phase. Switch from response view to tone view: blue is the input and orange the actual two-tap sum.

The input includes the sample at n=−1 and continues indefinitely. This is a steady-state calculation. A signal switched on at n=0 with zero earlier input would have a startup transient and a different first output. Do not infer that transient from the present plot.

## A zero has no phase

For the averaging filter r=0.5,

$$
H(e^{i\omega})=e^{-i\omega/2}\cos(\omega/2),\qquad
H(e^{i\pi})=0.
$$

At the Nyquist frequency the alternating input [1,−1,1,−1,…] cancels exactly. Its output is zero and phase is undefined, not a measured zero-radian delay. The interface leaves phase blank at this exact zero. Close to the zero, small numerical or measurement changes can make phase estimates unreliable even when magnitude is technically nonzero.

## Compare three endpoints

At r=0 the filter is the identity. At r=1 it is a pure one-sample delay with unit magnitude at every frequency; a magnitude-only display cannot distinguish it from identity. At r=0.5 it averages and suppresses high frequencies. For every allowed r, DC gain is one because the taps sum to one.

## A stable FIR need not have a stable causal inverse

The forward taps are nonnegative and sum to one, so the allowed FIR always has ℓ¹ norm one and BIBO gain at most one. Its inverse has a different stability question. For r≠1,

$$
H(z)=(1-r)+rz^{-1},\qquad
G(z)=\frac1{1-r}\frac1{1+\frac r{1-r}z^{-1}}.
$$

The causal geometric expansion has ratio −r/(1−r). It is absolutely summable only when r<1/2, giving inverse ℓ¹ norm 1/(1−2r). For 1/2<r<1, use the negative-time expansion instead; that noncausal inverse is stable with norm 1/(2r−1). At r=1 the inverse is a one-sample advance. At r=1/2, H is zero on the unit circle, so no ℓ¹ convolution inverse can satisfy G(e^iω)H(e^iω)=1 everywhere.

| r | Stable inverse | Inverse ℓ¹ norm |
|---|---|---|
| 0 | Identity | 1 |
| 1/4 | Causal geometric tail | 2 |
| 1/2 | No ℓ¹ inverse | ∞ |
| 3/4 | Negative-time tail | 2 |
| 1 | One-sample advance | 1 |

The code truncates the causal r=1/4 inverse and verifies every interior convolution coefficient exactly. Its final residual is (1/3)^N and its partial norm tends to two. This is an infinite-time inverse statement, separate from inversion of one finite periodic record.

## Interpret linear phase and tiny responses separately

Away from the r=1/2 zero, the factorization H=e^(−iω/2)cos(ω/2) on 0≤ω<π gives group delay −d arg H/dω=1/2 sample. The derivative is not defined at the zero. Other r generally produce frequency-dependent group delay; a single magnitude does not determine it.

Near π, computing 1−r+r cosω can erase the real component even when it is positive. The implementation uses (1−2r)+2r cos²(ω/2) there, with imaginary component −r sinω, and preserves the exact designated endpoint zero at r=1/2,ω=π. At ω=π−2⁻⁵¹, the real component is about 8.02×10⁻³² and the magnitude remains nonzero. Small real part, small gain and an exact zero are different claims.

**Exercise:** Is every noncausal inverse unstable? **Explanation:** No. The r=3/4 inverse is a stable counterexample, but it requires future samples and is unavailable as a zero-delay online causal filter.

```python
from fractions import Fraction as F

r=F(1,4)
for count in [1,2,4,8]:
    g=[(-r/(1-r))**n/(1-r) for n in range(count)]
    convolution=[F(0)]*(count+1)
    for n,v in enumerate(g):
        convolution[n]+=(1-r)*v
        convolution[n+1]+=r*v
    assert convolution[0]==1
    assert all(v==0 for v in convolution[1:count])
    assert abs(convolution[-1])==F(1,3)**count
    norm=sum(map(abs,g))
    assert norm==2*(1-F(1,3)**count)
    print('inverse taps, partial l1 norm, final residual:',count,str(norm),str(abs(convolution[-1])))
# At r=3/4 the stable inverse has negative-time support and the same l1 norm 2.
# At r=1/2 the unit-circle zero rules out any l1 convolution inverse.
```

## Exercise and AI connection

Predict y[n] for a constant input: the value stays constant in steady state. Verify the DC readout, then compare r=0 and r=1 in tone view to reveal the phase change hidden by their equal gain curves. For ω=π and r=0.25, H=0.5, so the alternating output retains half the amplitude with zero phase.

Learned temporal or spatial kernels also have frequency responses when treated as linear shift-invariant operators. Nonlinear activation, position-dependent weights and boundary handling break that simple global description. State those assumptions before using a frequency plot to explain a whole neural network.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
