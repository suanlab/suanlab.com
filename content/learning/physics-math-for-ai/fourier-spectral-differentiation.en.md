## Differentiate using all periodic samples

A periodic function can be represented by oscillatory modes, and differentiation multiplies each mode by its wave number. A spectral method therefore uses the whole periodic sample set rather than only neighboring differences. It can be very accurate for smooth, resolved functions, but it cannot manufacture information absent from the samples.

The experiment samples f=A sin(kx)+0.2 cos(2x) at N points x_j=2πj/N on [0,2π). Do not duplicate the endpoint 2π, which is the same periodic location as zero. Blue shows the derivative from an actual directly computed DFT. Orange is the derivative of the original continuous function. The table separately reports periodic central differences.

## Normalization and signed wave numbers

$$
\widehat f_m=\frac1N\sum_{j=0}^{N-1}f_j e^{-imx_j},\qquad
(Df)_j=\sum_m i\kappa_m\widehat f_m e^{i\kappa_mx_j}.
$$

Indices in the latter half of the DFT represent negative frequencies. At N=16, index fifteen represents wave number minus one. The even-grid Nyquist index N/2 cannot uniquely distinguish positive from negative orientation from real samples. This implementation sets its derivative multiplier to zero, following a real-interpolant convention, while retaining its coefficient in energy calculations.

Because the forward transform includes 1/N, the sample mean square equals the sum of squared coefficient magnitudes. That Parseval check can detect normalization mistakes. It does not identify an unresolved continuous signal.

## Aliasing changes derivatives

At grid nodes, sin((k+N)x_j)=sin(kx_j). Nevertheless the continuous derivatives have different factors k+N and k, even at those same nodes. With N=16, a sine of frequency seventeen looks like frequency one. Its true derivative contribution at zero is seventeen, while the sampled spectral interpretation gives one. A smooth-looking numerical curve is therefore not evidence that the continuous derivative is correct.

The resolution flag concerns the specified single sine component. It does not mean unknown high frequencies can always be detected from samples alone. Nonperiodicity, discontinuities, Gibbs effects and boundary conditions require additional treatment beyond this periodic trigonometric example.

## Worked example and exercise

For N=16 and k=3, the spectral sine multiplier is three. The central-difference multiplier is sin(3h)/h≈2.35264, so it underestimates this derivative. The spectral derivative is near roundoff for this resolved trigonometric polynomial.

Exercise: compare samples and derivatives of the Nyquist sine k=N/2. Solution: sin(πj)=0 at every node, but its true derivative is (N/2)cos(πj), which is nonzero. The selected Nyquist convention cannot recover that derivative from zero sine samples. Also try A=0: the absent component contributes no error regardless of its specified frequency, illustrating why a resolution flag alone does not measure actual error.

## Connection to AI frequency operators

Fourier neural operators and spectral PDE solvers depend on normalization, truncation, boundary assumptions and resolution transfer. Accurate reconstruction does not automatically imply accurate derivatives, fluxes or high-frequency energy. Compare against continuous targets while changing resolution, and record wave-number conventions even when using an FFT library. A small direct DFT is used here to keep each operation inspectable.

Reference: [MIT numerical PDE materials](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
