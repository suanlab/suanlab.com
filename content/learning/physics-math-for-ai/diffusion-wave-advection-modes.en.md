## The same initial shape can obey different laws

On the periodic domain [0,2pi], take an integer spatial mode k and initial displacement u0=M+A cos(kx). The lab compares diffusion, waves and constant-speed transport. They share an initial graph but evolve through different equations and require different data. The wave additionally needs initial velocity v0=B cos(kx).

Blue shows the current field and orange the initial displacement. A snapshot close to zero does not mean the state has disappeared: in a second-order wave equation, velocity remains part of the state. This is a concrete reason to identify state variables before interpreting a visualization.

## Diffusion damps short spatial scales faster

For u_t=nu u_xx,

$$
u(x,t)=M+A e^{-\nu k^2t}\cos(kx).
$$

Here the symbol u on the left is the field, while nu denotes diffusivity. Each second spatial derivative contributes minus k², giving amplitude decay rate nu k². Setting diffusivity zero freezes the shape. Increasing k makes diffusion much faster at the same physical time.

The periodic mean stays M. The centered displacement energy E=one half integral of (u-M)² equals pi A² exp(-2nu k²t)/2 and satisfies dE/dt=-nu integral of u_x². The field loses this energy through smoothing rather than through a boundary flux.

## Waves exchange kinetic and spatial energy

For u_tt=c²u_xx, write omega=ck:

$$
u(x,t)=M+\left(A\cos\omega t+\frac B\omega\sin\omega t\right)\cos(kx),
\qquad
E_{\mathrm{wave}}=\frac12\int(u_t^2+c^2u_x^2)\,dx.
$$

The full energy is pi(B²+omega² A²)/2, constant in time. At A=c=k=1, B=0 and t=pi/2, displacement is nearly flat but velocity stores the energy. Displacement L2 alone is therefore not the conserved wave quantity.

## Advection translates without smoothing

For u_t+c u_x=0, the exact field is M+A cos(k(x-ct)). The shape shifts with unchanged amplitude, periodic mass and centered L2. In a coordinate system moving at c, this solution is stationary. It differs both from diffusion's amplitude loss and the standing wave's exchange between displacement and velocity.

The readout checks the selected PDE at x=0.37 using separately evaluated time and spatial derivatives. The mass is 2pi M in every model because the mean velocity and net periodic boundary flux are zero in this experiment.

## Energy estimates and why backward heat is ill-posed

For smooth periodic heat solutions, integration by parts gives d(‖u−M‖²/2)/dt=−ν‖uₓ‖². Applied to a difference of two solutions, this proves L² contraction and uniqueness. For wave solutions, multiplying uₜₜ=c²uₓₓ by uₜ gives conservation of (‖uₜ‖²+c²‖uₓ‖²)/2. The mean displacement requires its own initial value, since the gradient does not control constant modes. For advection, d‖u‖²/dt=−c∫(u²)ₓ=0. Periodicity removes boundary terms in each argument.

These identities extend the single-mode lab to finite Fourier sums by orthogonality, and to energy solutions through justified approximation. Forward heat is stable, but reconstructing an earlier state multiplies mode k by e^(νk²T). Take ν=T=1 and initial uₖ(x,0)=cos(kx). At T its L² norm is √πe⁻ᵏ²→0 while its initial norm stays √π. Therefore the inverse is not continuous in the L² norm, even though every individual finite mode can be reversed exactly.

| k | log of backward amplification at νT=1 |
|---|---:|
| 1 | 1 |
| 2 | 4 |
| 4 | 16 |

```python
from math import exp, pi, fsum
modes = [(1, 1.), (2, .5), (4, -.25)]
nu, T = .2, .7
initial_energy = pi/2*fsum(a*a for k, a in modes)
final_energy = pi/2*fsum(a*a*exp(-2*nu*k*k*T) for k, a in modes)
dissipation = -nu*pi*fsum(k*k*a*a*exp(-2*nu*k*k*T) for k, a in modes)
assert 0 < final_energy < initial_energy and dissipation < 0
for k in (1, 2, 4):
    terminal_norm = pi**.5*exp(-k*k)
    log_inverse_gain = k*k
    assert abs(terminal_norm*exp(log_inverse_gain)-pi**.5) < 1e-14
    print(k, terminal_norm, log_inverse_gain)
print('energy:', initial_energy, final_energy, 'derivative:', dissipation)
```

The code compares exact finite-mode energy factors and logarithmic amplification, avoiding overflow. Diffusion-model reverse dynamics includes a distribution-dependent score and is not simply unrestricted backward heat applied to noisy observations. Inverse PDE learning requires regularization or information about admissible states; a low forward residual alone cannot stabilize inversion.

## Exercise: compare the right invariant

Use A=1, M=0, k=2, nu=0.2 and t=0.5 in heat mode. The amplitude is exp(-0.4). In wave mode use k=c=1 and t=pi/2; compute kinetic energy to explain how total energy remains pi/2 despite near-zero displacement. Finally set transport mean M=0.5: mass is pi even though the moving cosine changes sign relative to its mean. State which invariant each comparison actually tests.

Further reading: [MIT PDE course](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/). All mode formulas and energy checks are independently derived.
