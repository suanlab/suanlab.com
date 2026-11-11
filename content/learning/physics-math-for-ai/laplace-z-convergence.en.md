## A rational expression is only half the specification

Transforms accumulate contributions over an entire signal. A compact algebraic answer can hide whether the defining integral or series actually converges. The region of convergence, or ROC, is part of the answer. Here we use bilateral transforms; unilateral initial-value conventions are different.

For a>0, compare a causal exponential with a negative anti-causal exponential. The minus sign on the latter is intentional:

$$
\begin{aligned}
\mathcal L\{e^{-at}u(t)\}(s)&=\frac1{s+a},&&\Re s>-a,\\
\mathcal L\{-e^{-at}u(-t)\}(s)&=\frac1{s+a},&&\Re s<-a.
\end{aligned}
$$

To derive the first condition, the magnitude of the integrand decays like exp(−(Re s+a)t) for positive time. The second integral runs toward negative infinity, reversing the inequality. The value assigned at t=0 does not change either integral.

## The same formula, different signals

At a=1 and s=0, the causal integral is one. The anti-causal integral diverges even though substituting into 1/(s+1) still gives one. The readout separates the rational continuation from the actual convergent transform. At s=−1 there is an algebraic pole, so neither receives a finite value. At s=−2 the anti-causal transform exists and equals −1.

Blue shows the selected sided signal on a finite time interval. An exponentially growing left tail is not made integrable by its truncated drawing. Open and filled markers distinguish endpoint conventions.

## Discrete time changes the ROC geometry

For the bilateral Z transform, sum x[n]z⁻ⁿ over all integers. The corresponding pair is

$$
\begin{aligned}
\mathcal Z\{a^nu[n]\}(z)&=\frac z{z-a},&&|z|>a,\\
\mathcal Z\{-a^nu[-n-1]\}(z)&=\frac z{z-a},&&|z|<a.
\end{aligned}
$$

Geometric-series convergence now gives circles instead of vertical half-planes. With a=0.5 and z=0.25, the anti-causal sum is −1. Its negative-time series can be written as −Σₘ≥₁(z/a)ᵐ, which also defines value zero at z=0. The causal series does not converge there.

## Derive a two-sided stable response from an annular ROC

The same two-pole rational expression has three possible sidedness assignments:

$$
H(z)=\frac z{z-1/2}+\frac z{z-2}.
$$

If both geometric terms are right-sided, its ROC is |z|>2 and the impulse response grows to the right. If both are negative left-sided, the ROC is |z|<1/2 and one tail grows to the left. In the middle annulus 1/2<|z|<2, the first term is right-sided and the second negative left-sided:

$$
h[n]=(1/2)^n u[n]-2^n u[-n-1],
\qquad \sum_{n\in\mathbb Z}|h[n]|=3.
$$

The unit circle belongs to this middle ROC, so it has an ordinary continuous frequency response. The system is stable and noncausal even though one pole lies outside the unit circle. The familiar rule that all poles lie inside the unit circle presumes a causal rational system; removing that premise changes the conclusion.

| Symmetric tail cutoff N | Partial DC sum | Partial absolute sum |
|---|---|---|
| 1 | 1/2 | 3/2 |
| 2 | 3/4 | 9/4 |
| 4 | 15/16 | 45/16 |
| 8 | 255/256 | 765/256 |

The code sums n=0,…,N−1 on the right and n=−1,…,−N on the left. DC gain tends to one while the absolute sum tends to three. They are different quantities. This two-pole example is separate from the live single-pole control.

## Prove what bounded-input bounded-output stability means

For an absolutely summable discrete impulse response, |(h*x)[n]|≤‖h‖₁‖x‖∞ proves BIBO stability. Conversely, finite-support bounded test inputs can choose x[−k]=sign(h[k]) on a growing finite set. The output at zero becomes the corresponding sum of |h[k]|. If these sums are unbounded, no uniform BIBO gain exists. For complex h, use its conjugate phase instead of sign.

The continuous analogue uses ∫|h| for an ordinary impulse-response function. Initial conditions, unilateral transforms and distributional impulses need their own formulation. Do not infer integrability from a finite plotted tail or a finite rational value outside the ROC.

**Exercise:** Does substituting z=1 into H prove stability for all three ROCs? **Explanation:** No. Only the middle annulus contains the unit circle; algebraic continuation is not the defining convergent sum.

```python
from fractions import Fraction as F

# h[n]=(1/2)^n for n>=0, and -2^n for n<=-1.
# Its Z transform has ROC 1/2<|z|<2 and includes z=1.
for N in [1,2,4,8]:
    causal=sum((F(1,2)**n for n in range(N)),F(0))
    anticausal=-sum((F(1,2)**m for m in range(1,N+1)),F(0))
    dc=causal+anticausal
    absolute_sum=causal-anticausal
    assert dc==1-F(1,2)**N
    assert absolute_sum==3*(1-F(1,2)**N)
    print(N,str(dc),str(absolute_sum))
print('limits: DC gain 1; absolute impulse-response sum 3')
# Changing the ROC changes sidedness even though z/(z-1/2)+z/(z-2) stays the same.
```

## Exercise and systems interpretation

Choose a=1.5 and z=1. Which side has a convergent transform? Only the anti-causal side, since 1<1.5. Its rational value is −2. Verify both readouts, then move onto the pole z=1.5.

The imaginary axis in Laplace analysis and the unit circle in Z analysis matter for frequency responses. Whether they belong to the ROC must be checked before identifying an ordinary Fourier response. Causality, stability and pole location require a jointly specified signal or system and ROC; the algebraic fraction alone does not identify them.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
