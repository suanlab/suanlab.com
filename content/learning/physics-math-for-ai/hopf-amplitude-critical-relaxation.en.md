## 1. Rotating is different from sustaining an oscillation

Think of z=x+iy as an arrow in a plane. Its length is amplitude and its direction is phase. An arrow that rotates while shrinking produces visible oscillations that eventually disappear. An arrow whose length approaches a positive constant while it keeps rotating describes a sustained oscillation. A few turns in a finite observation window do not distinguish these outcomes.

We separate amplitude growth and saturation from phase rotation. When saying that a system slows down near criticality, specify whether rotation stops or amplitude disturbances recover slowly. This also prepares the amplitude-stability conditions needed by later phase models.

## 2. Fix the model and its units

Use the following deterministic model. Amplitude is dimensionless, time is in seconds, and μ,ω,g,c have units s⁻¹. Require g>0.

$$
\dot z=(\mu+i\omega)z-(g+ic)|z|^2z,\qquad g>0.
$$

Here μ controls growth of small amplitudes, g limits growth, ω is the angular frequency near the origin, and c controls amplitude dependence of rotation. Setting c=0 does not remove amplitude growth or saturation. We use the polynomial field itself as our experimental model. Applying it to a physical system requires separate justification of the approximation region and error.

$$
\dot x=(\mu-g\rho)x-(\omega-c\rho)y,\qquad
\dot y=(\omega-c\rho)x+(\mu-g\rho)y.
$$

## 3. Read polar coordinates without inventing a phase at the origin

For r>0, substitute z=r exp(iθ) and separate real and imaginary parts.

$$
\dot r=\mu r-gr^3,\qquad \dot\theta=\omega-cr^2.
$$

The radial equation can be solved without θ, but the phase equation depends on radius. Amplitude and phase are therefore not completely independent. At r=0 there is no direction: phase and actual phase velocity remain unavailable. A coefficient ω in the complex equation does not give the zero vector an observable direction.

## 4. Squared amplitude obeys a Riccati equation

Set ρ=r² to compute radial dynamics without square roots. This is our chosen dimensionless observation, not a probability or automatically a physical energy.

$$
\dot\rho=2\mu\rho-2g\rho^2,\qquad
u=\rho^{-1}\Longrightarrow\dot u=-2\mu u+2g.
$$

The reciprocal variable in the second equation is u. Use this transformation only for ρ>0. The nonlinear equation becomes an affine linear equation for u, allowing both a closed solution and an independent matrix-exponential check. Treat the origin separately instead of forcing it through a reciprocal transformation.

## 5. An exact flow across the critical parameter

For initial squared amplitude ρ₀, the solution is

$$
\rho(t)=
\begin{cases}
\dfrac{\rho_0e^{2\mu t}}{1+(g\rho_0/\mu)(e^{2\mu t}-1)},&\mu\ne0,\\[6pt]
\dfrac{\rho_0}{1+2g\rho_0t},&\mu=0.
\end{cases}
$$

This is the solution of the specified model, not a result obtained by shrinking a time step. The implementation rewrites the expression according to the sign of μ to avoid forming a large positive exponential. The μ=0 branch uses exact integer/rational evidence. A short trajectory with μ=10⁻⁹ may resemble the critical trajectory without being classified as the same model.

## 6. Phase remembers accumulated amplitude

Let A(t) be the integral of squared amplitude. D is the denominator in the preceding solution; at μ=0 it is1+2gρ₀t.

$$
A(t)=\int_0^t\rho(s)\,ds=\frac{\ln D(t)}{2g},\qquad
\theta(t)-\theta(0)=\omega t-cA(t).
$$

Equal current radii need not imply equal accumulated phases if amplitude histories differ and c≠0. The computed phase shift is an unwrapped accumulated angle, distinct from coordinates folded into a circle. Reconstruct x,y by rotating the initial vector through this angle and scaling it by r/r₀. Do not project the state back onto an arbitrary circle at each record.

## 7. Below onset, specify what relaxes

For μ<0 the origin is stable. At small amplitude, r decays proportionally to exp(μt), while ρ decays proportionally to exp(2μt). Thus two observations of the same trajectory have different linear relaxation times.

| Observation and state | Linear growth rate | Positive relaxation time |
| --- | --- | --- |
| Small r at μ<0 | μ | 1/\|μ\| |
| Small ρ at μ<0 | 2μ | 1/(2\|μ\|) |
| Radial disturbance of the μ>0 circle | −2μ | 1/(2μ) |

These times describe local linearization. They are not universal half-decay times for a finite disturbance. At μ>0 the origin is unstable and is not assigned a positive recovery time.

## 8. At criticality, decay is algebraic

For μ=0 and ρ₀>0, amplitude and squared amplitude have different half-decay times.

$$
r(t)=\frac{r_0}{\sqrt{1+2g\rho_0t}},\qquad
t_{\rho/2}=\frac1{2g\rho_0},\qquad
t_{r/2}=\frac3{2g\rho_0}.
$$

Zero real part of a linear eigenvalue does not remove nonlinear decay. The origin remains weakly attracting, with long-time amplitude proportional to t⁻¹ᐟ². Divergence of a linear relaxation time does not make every finite-disturbance half-decay time infinite. Distinguish the order in which disturbance size and μ approach zero.

## 9. Above onset, separate radial stability from rotation

For μ>0 there is an invariant circle of positive amplitude. Its radial and tangential dynamics are different.

$$
\rho_* =\frac\mu g,\qquad
\Omega_* =\omega-c\frac\mu g,\qquad
\lambda_{\rm radial}=-2\mu.
$$

If Ω*≠0, the circle is a periodic orbit with period2π/|Ω*|. The phase direction is neutral because time origin is free. Radial stability does not send every initial phase to one point. If Ω*=0, every point on the circle is an equilibrium; a rotation period is unavailable. Negative Ω* reverses rotation and does not give a negative period.

## 10. Ordinary Hopf conditions and important exceptions

The linearization at the origin is

$$
J_0=\begin{pmatrix}\mu&-\omega\\\omega&\mu\end{pmatrix},\qquad
\lambda_\pm=\mu\pm i\omega.
$$

Ordinary Hopf requires ω≠0 at μ=0. Along this coefficient path, the real part crosses with μ and positive saturation gives the supercritical form. With ω=0, both linear eigenvalues vanish and the classification is different. See [Kuznetsov's original exposition](https://www.scholarpedia.org/article/Andronov-Hopf) for the definition and nondegeneracy conditions. The solutions, examples and observation comparisons here are derived directly for our specified model.

Another exception is the exact initial state z₀=0. Even for μ>0, the noiseless solution remains zero. Instability concerns growth of a perturbation. We do not insert unmodeled noise to make an oscillation appear spontaneously.

## 11. Critical relaxation and the conditions for phase reduction

Radial recovery slows as μ approaches zero. Even at positive μ, replacing r immediately by a constant needs a separation between recovery rate2μ and the changes caused by coupling or forcing. Later phase equations should be read as separate models with this assumption. A phase-only description is not automatically valid where amplitude disappears.

Critical relaxation here concerns a finite-dimensional normal form. Do not identify it with a thermodynamic limit or a noise-driven collective transition. Rotation can retain a finite frequency while amplitude recovery becomes slow.

## 12. A critical example you can calculate by hand

Choose μ=0,g=1,c=1,ω=2,z₀=1. Then ρ=1/(1+2t), and accumulated phase is2t−ln(1+2t)/2. Amplitude decreases while angular speed2−ρ increases. The spiral's appearance does not imply faster recovery of amplitude disturbances.

| t [s] | ρ | r | Phase shift [rad] |
| --- | --- | --- | --- |
| 0 | 1 | 1 | 0 |
| 1/2 | 1/2 | 1/√2 | 1−ln2/2 |
| 1 | 1/3 | 1/√3 | 2−ln3/2 |
| 3/2 | 1/4 | 1/2 | 3−ln2 |

We can separately measure sensitivity to the initial squared amplitude.

$$
\frac{\partial\rho(t)}{\partial\rho_0}
=\left(\frac{\rho(t)}{\rho_0}\right)^2e^{-2\mu t},\qquad\rho_0>0.
$$

This is sensitivity near the selected initial preparation. It does not by itself decide identifiability of all coefficients or stability of a learning algorithm.

## Research extension: the phase that survives amplitude recovery

### A clock reading is more than a polar angle

Suppose two arrows have the same direction but different lengths. With c≠0 they initially rotate at different speeds. After both lengths recover to the attracting circle, a permanent angular offset can remain. To predict the eventual clock reading, attach a correction to the current polar angle. This corrected variable is the asymptotic phase. Points with equal corrected phase form an isochron; they approach the same moving point on the cycle at the same physical time.

Assume μ>0, g>0 and Ω\*≠0. Set a=c/g and r\*=sqrt(μ/g). For every nonzero state define a circle-valued phase, with a continuous lift along a trajectory when needed.

$$
\phi=\theta-a\ln(r/r_*),\qquad
\dot\phi=\omega-cr^2-a(\mu-gr^2)=\Omega_*,\qquad
\Omega_*=\omega-c\mu/g.
$$

This cancellation is exact for our specified unforced polynomial model, even away from the cycle. It is not a general formula for every Hopf system after truncation. On the cycle φ=θ. Away from it, equal θ does not imply equal φ. An isochron is a logarithmic spiral when a≠0 and a radial ray when a=0.

$$
\theta=\phi+a\ln(r/r_*),\qquad
v=\frac{r_*^2}{r^2}-1,\qquad
\dot v=-2\mu v,\qquad -1<v<\infty.
$$

The radial coordinate v has an exactly exponential evolution. It is not r−r\*, which is only locally exponential. Since r→r\*, the angular difference between the trajectory and the cycle point with phase φ(t) tends to zero. The origin is excluded: its angle is undefined and the logarithm diverges. When Ω\*=0 the same algebra supplies a conserved label for a circle of equilibria, not a periodic clock. At μ≤0 this attracting-cycle construction is unavailable.

### From an impulse to a permanent phase shift

At a point (x,y), differentiate φ with respect to Cartesian state. Denote this gradient by Z. Phase is in radians; do not confuse Z with the complex population order parameter in the later chapter.

$$
Z(x,y)=\nabla\phi=\frac1{r^2}\begin{pmatrix}-y-ax\\x-ay\end{pmatrix},\qquad
\|Z\|=\frac{\sqrt{1+a^2}}r,\qquad
Z\cdot F=\Omega_*.
$$

Here F is the unforced Cartesian vector field. Along the cycle Z obeys the adjoint equation below. Its normalization is Ω\*, because φ is an angular phase. For a time coordinate φ/Ω\*, the corresponding normalization is 1.

$$
\frac{dZ}{dt}=-[DF]^\mathsf{T}Z,\qquad
Z_*(\phi)=\frac1{r_*}
\begin{pmatrix}-\sin\phi-a\cos\phi\\\cos\phi-a\sin\phi\end{pmatrix}.
$$

A brief impulse is modeled here as a prescribed instantaneous state displacement h, followed by the original unforced flow. It is not an additional control on the existing reader. Start on the cycle at polar angle θ, and write its dimensionless displacement in the local radial/tangential frame as w. For |w|<1 the branch connected to zero displacement is unambiguous.

$$
w=\frac{(h_x+ih_y)e^{-i\theta}}{r_*},\qquad
\Delta\phi=\operatorname{Im}\log(1+w)-a\operatorname{Re}\log(1+w).
$$

This is already the lasting phase difference: after the pulse both corrected phases advance at the same Ω\*. A radial outward displacement can therefore change the eventual clock reading even though it causes no immediate change in polar angle. The first-order phase response and an explicit remainder bound follow from the series for the complex logarithm.

$$
\Delta\phi_{\rm lin}=\operatorname{Im}w-a\operatorname{Re}w=Z_*\cdot h,\qquad
|\Delta\phi-\Delta\phi_{\rm lin}|
\leq\frac{\sqrt{1+a^2}|w|^2}{2(1-|w|)}.
$$

The bound is a finite-displacement statement for |w|<1; it becomes loose near that boundary. Larger displacements need a branch and basin check. A pulse landing exactly at the origin has no phase. Do not assign it a finite shift by extending the linear response formula.

### Why weak forcing becomes harder near onset

For a specified additive force f(x,y,t), the off-origin chain rule gives exact amplitude and phase equations. The radial component of f changes r, and r also changes the phase sensitivity.

$$
\dot r=\mu r-gr^3+e_r\cdot f,\qquad
\dot\phi=\Omega_*+Z(x,y)\cdot f,\qquad e_r=(x,y)/r.
$$

Replacing Z by its cycle value gives a phase-only approximation. Using r≈r\* and θ≈φ requires amplitude disturbances to stay small. A useful local scale check compares a sustained force with the radial restoring rate 2μ times the cycle radius. It is a consistency estimate, not an error theorem for arbitrary forcing.

$$
\frac{|f|}{2\mu r_*}\ll1,\qquad
\frac{|h|}{r_*}\ll1,\qquad
\|Z_*\|=\sqrt{1+a^2}\sqrt{g/\mu}.
$$

With g,c and the absolute forcing scale fixed, the sustained-force ratio grows as μ⁻³ᐟ² and impulse sensitivity grows as μ⁻¹ᐟ². Meanwhile radial recovery slows. An exact unforced phase coordinate therefore does not guarantee an accurate forced phase-only model near onset. This distinction matters when fitting oscillator models to neural or learned latent dynamics.

### A radial pulse you can reproduce

Choose μ=g=c=1, ω=2, so r\*=1 and Ω\*=1 rad/s. A radial pulse h=κ r\*eᵣ has w=κ. For κ>−1 the exact shift is −ln(1+κ); its first-order estimate is −κ. The code and table compare these, with the conservative bound above. No sampled trajectories or confidence intervals are used.

![Logarithmic isochrons meeting the attracting cycle and exact versus linear radial pulse phase shifts](/assets/learning/hopf-isochron-research-phase-response.en.svg)

The upper panel connects states with the same asymptotic clock reading; equal geometric angles on different circles generally have different readings. The lower panel compares exact and linear pulse responses for a=1. The marked example κ=0.5 has no instantaneous angular jump, yet its lasting phase shift is about −0.405 rad.

```python
from math import log1p, sqrt

for kappa in [0.01, 0.1, 0.5, -0.5]:
    exact = -log1p(kappa)
    linear = -kappa
    error = abs(exact - linear)
    bound = sqrt(2) * kappa**2 / (2 * (1 - abs(kappa)))
    print(f"{kappa:.2f} {exact:.6f} {linear:.6f} {error:.6f} {bound:.6f}")
```

| κ | Exact Δφ | Linear Δφ | Absolute error | Bound |
| --- | --- | --- | --- | --- |
| 0.01 | -0.009950 | -0.010000 | 0.000050 | 0.000071 |
| 0.10 | -0.095310 | -0.100000 | 0.004690 | 0.007857 |
| 0.50 | -0.405465 | -0.500000 | 0.094535 | 0.353553 |
| -0.50 | 0.693147 | 0.500000 | 0.193147 | 0.353553 |

The larger inward response comes from approaching the phase singularity at the origin. The same absolute pulse is more disruptive on a smaller cycle. For example, decreasing μ from 1 to 1/4 with g=1 halves r\*, doubles the infinitesimal sensitivity and multiplies the sustained-force ratio by 8.

### Research self-checks

#### 1. Can two different radii have the same eventual clock reading?

**Research explanation.** Yes. Choose θ₂−θ₁=a ln(r₂/r₁). Then φ₂=φ₁, so the two states have identical corrected phase at every later time. Both approach the same point of the moving cycle. If their initial angles were equal instead, the lasting shift would be −a ln(r₂/r₁).

#### 2. Does rapid radial relaxation erase a radial pulse's phase shift?

**Research explanation.** No. For a≠0 the transient change of rotation integrates to the fixed shift −a ln(1+κ). Faster recovery and the size of that integrated shift are distinct properties. For a=0 radial pulses leave phase unchanged, while tangential pulses can still shift it.

#### 3. Does the exact off-cycle phase remove all approximation error under forcing?

**Research explanation.** No. The exact equation retains Z(x,y), hence the amplitude and its history. Replacing it by Z\*(φ) discards that information. Near μ=0 a fixed pulse is no longer small relative to r\* and radial recovery is slow. A model containing only phase must justify those scales separately.

[Wedgwood et al., Introduction and Fig. 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3582465/) motivate isochrons, infinitesimal phase response and limits of strong-attraction assumptions. The formulas, remainder bound and numerical examples here are direct derivations for this chapter's coefficient convention. The reader already provides the original amplitude experiment, Worker calculation and fixed comparisons; this supplement adds standalone theory and Python, without an impulse simulator.

## 13. Six questions with explained solutions

**1. Find the two half-decay times in the critical example.** Solution:ρ reaches1/2 at1/2s, while r reaches1/2 at3/2s. Halving r quarters ρ.

**2. For μ=−1,g=1,z₀=1, find ρ(t) and the small-amplitude relaxation time.** Solution:ρ=1/(2e²ᵗ−1). At long times r is a constant times e⁻ᵗ. The local linear time is1s for r and1/2s for ρ.

**3. Is the attracting circle periodic for μ=1,g=1,c=2,ω=2?** Solution:ρ*=1 and Ω*=0. Its points are equilibria and a rotation period is unavailable. Radial disturbances still decay locally at rate2/s.

**4. What changes when c=3 instead?** Solution:Ω*=−1rad/s, so rotation reverses. The period is positive2πs. The radial stability does not change with c.

**5. Why does an exactly zero initial state fail to leave the origin for μ>0?** Solution:The right-hand side vanishes there. Uniqueness and instability are compatible. A small positive initial amplitude grows, but it defines a different initial preparation.

**6. Can one invariant-circle trajectory identify μ,g,ω,c individually?** Solution:No. It restricts the combinations μ/g and ω−cμ/g. Recovery and phase-speed measurements at other amplitudes add information. A small error over a short observation window also does not establish the exact sign of the critical parameter.

## 14. Numerical evidence and research connections

The production calculation uses exact rational sign/equality and directed256-bit exponential, logarithm, square-root and trigonometric enclosures. Large accumulated angles are neither wrapped nor used to renormalize the state. Independent enclosures for x²+y²−ρ and the log-amplitude balance are distinguished from algebraic identities. Record density does not change physical time or the analytic flow.

The independent reference imports no production modules. It computes a2×2matrix exponential for reciprocal squared amplitude at220digits. Finite reference cases are not a machine proof for all inputs. The reader, Worker, visualization and the pairwise and collective families are now registered; their integration evidence is linked in the research progress record.

For AI research, distinguish rotating latent dynamics, sustained amplitude and recovery after perturbation. This toy model is not a universal model of neural networks or measured systems. Next we study relative phase after amplitude stabilization and the information lost by collective order observations.
