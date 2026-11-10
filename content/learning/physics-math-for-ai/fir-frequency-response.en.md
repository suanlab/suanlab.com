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

## Exercise and AI connection

Predict y[n] for a constant input: the value stays constant in steady state. Verify the DC readout, then compare r=0 and r=1 in tone view to reveal the phase change hidden by their equal gain curves. For ω=π and r=0.25, H=0.5, so the alternating output retains half the amplitude with zero phase.

Learned temporal or spatial kernels also have frequency responses when treated as linear shift-invariant operators. Nonlinear activation, position-dependent weights and boundary handling break that simple global description. State those assumptions before using a frequency plot to explain a whole neural network.

Further reading: [MIT 6.003, Signals and Systems lecture notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/lecture-notes/).
