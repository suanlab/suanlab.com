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

## Exercise and systems interpretation

Choose a=1.5 and z=1. Which side has a convergent transform? Only the anti-causal side, since 1<1.5. Its rational value is −2. Verify both readouts, then move onto the pole z=1.5.

The imaginary axis in Laplace analysis and the unit circle in Z analysis matter for frequency responses. Whether they belong to the ROC must be checked before identifying an ordinary Fourier response. Causality, stability and pole location require a jointly specified signal or system and ROC; the algebraic fraction alone does not identify them.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
