## Recovering an input needs more than a forward formula

An inverse problem starts with observations and asks which input produced them. This experiment uses periodic eight-sample convolution, two taps h=[1−r,r], and alternating observation noise. The reference truth combines a low-frequency cosine with a Nyquist component:

$$
x[n]=\cos(2\pi n/8)+\tfrac12(-1)^n,\qquad y=h*_8x+\eta(-1)^n.
$$

The Fourier-domain observation is Y=HX+E. A direct inverse divides by H only if every response bin is nonzero. Small nonzero values amplify noise; exact zeros destroy identifiability of the corresponding component. These are distinct problems.

## An exact zero cannot be inverted

At r=0.5, H[4]=1−2r=0. All values of the truth's alternating component produce the same noiseless observation. The direct inverse is therefore undefined. The interface draws no invented reconstruction curve and gives no reconstruction MSE for that request. Infinity in reciprocal gain describes a failed inverse, not a finite computed estimate.

With zero regularization, the minimum-norm pseudoinverse instead sets the missing coefficient to zero. It recovers the visible cosine but cannot infer the reference alternating amplitude 0.5. Its reconstruction MSE is 0.5²=0.25 even without recoverable error in the cosine. Selecting a minimum-norm solution is additional structure, not evidence of uniquely recovered truth.

## Stabilization trades fit for size

For positive λ the Tikhonov estimate is

$$
\widehat X[k]=\frac{\overline{H[k]}Y[k]}{|H[k]|^2+\lambda}.
$$

With our DFT convention this minimizes Σₙ|(h*_8z)[n]−y[n]|²+λΣₙ|z[n]|²; the common Parseval factor cancels. Positive λ suppresses unstable gains but also shrinks identifiable components. At r=0.5, noise amplitude 0.05 and λ=0.1, MSE is about 0.2555. The missing component still dominates the error. Blue is truth and orange the selected estimate; the table exposes observed blur plus noise separately.

## Near a zero is also dangerous

Set r=0.49, λ=0 and choose direct inversion. The Nyquist gain is 0.02, so observation noise 0.05 becomes input error amplitude 2.5. Its MSE is 2.5²=6.25. A tiny observation residual can coexist with a very poor reconstruction: fitting the noisy data exactly is not the same as recovering its source.

## Exercise and research connection

First set r=0 and noise zero. Direct inversion is identity and MSE vanishes. Then restore r=0.5 and compare direct, pseudoinverse and positive-λ estimates. Explain which outputs are undefined, which choose among compatible inputs, and which introduce shrinkage.

Deblurring, sensor reconstruction and learned inverse solvers require explicit boundary conditions, noise assumptions and priors. Regularization can select or stabilize a solution, but it cannot create observational identifiability. Report reconstruction error and observation residual separately when a reference truth is available.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
