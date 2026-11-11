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

## Skew differentiation does not supply a discrete chain rule

With the stated real Nyquist convention, periodic Fourier differentiation is skew-adjoint in the grid inner product: Dᵀ=−D. This follows from the purely imaginary signed multipliers and the unitary transform after normalization. Consequently uᵀDu=0, which conserves semidiscrete linear-advection energy. Multiplying samples to form nonlinear terms introduces aliasing, however, and D(u²)=2uDu is generally false.

For smooth Burgers dynamics, consider the split approximation uₜ=−[D(u²)+uDu]/3. Its grid energy derivative vanishes because uᵀD(u²)=−(Du)ᵀu². The cancellation is algebraic and does not assume a false discrete product rule. It does not restore unresolved modes, prove dealiased accuracy, or supply entropy dissipation after shock formation.

For the four-node real spectral matrix and u=(1,2,3,5), take the unweighted energy E=Σuᵢ²/2:

| Semidiscrete right-hand side | E′ |
|---|---:|
| −uDu | 9 |
| −D(u²)/2 | −9/2 |
| −[D(u²)+uDu]/3 | 0 |

```python
from fractions import Fraction as F
u=list(map(F,(1,2,3,5)))
def D(v):return [(v[(j+1)%4]-v[(j-1)%4])/2 for j in range(4)]
du,du2=D(u),D([x*x for x in u])
naive=[-x*y for x,y in zip(u,du)]
conservative=[-v/2 for v in du2]
split=[-(a+x*b)/3 for a,x,b in zip(du2,u,du)]
rates=[sum(x*y for x,y in zip(u,rhs)) for rhs in (naive,conservative,split)]
assert rates==[9,F(-9,2),0]
assert sum(u[i]*du[i] for i in range(4))==0
print(rates)
```

The four-node matrix below is an exact rational auxiliary example; the live DFT lab starts at eight nodes. Its Nyquist derivative is zero. Stable energy behavior and resolution are separate properties, so learned spectral operators should be checked for both. A conserved quadratic alone cannot certify a nonlinear PDE solution.

## Worked example and exercise

For N=16 and k=3, the spectral sine multiplier is three. The central-difference multiplier is sin(3h)/h≈2.35264, so it underestimates this derivative. The spectral derivative is near roundoff for this resolved trigonometric polynomial.

Exercise: compare samples and derivatives of the Nyquist sine k=N/2. Solution: sin(πj)=0 at every node, but its true derivative is (N/2)cos(πj), which is nonzero. The selected Nyquist convention cannot recover that derivative from zero sine samples. Also try A=0: the absent component contributes no error regardless of its specified frequency, illustrating why a resolution flag alone does not measure actual error.

## Connection to AI frequency operators

Fourier neural operators and spectral PDE solvers depend on normalization, truncation, boundary assumptions and resolution transfer. Accurate reconstruction does not automatically imply accurate derivatives, fluxes or high-frequency energy. Compare against continuous targets while changing resolution, and record wave-number conventions even when using an FFT library. A small direct DFT is used here to keep each operation inspectable.

Reference: [MIT numerical PDE materials](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
