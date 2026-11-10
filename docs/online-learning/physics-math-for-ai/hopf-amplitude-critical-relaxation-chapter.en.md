# The birth of amplitude and critical relaxation: a Hopf normal form

Authoring draft,2026-11-08. This is the first computational family for synchronization and critical dynamics. It is not registered in Interactive Book. Pairwise phase locking and collective order will follow.

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

## 13. Six questions with explained solutions

**1. Find the two half-decay times in the critical example.** Solution:ρ reaches1/2 at1/2s, while r reaches1/2 at3/2s. Halving r quarters ρ.

**2. For μ=−1,g=1,z₀=1, find ρ(t) and the small-amplitude relaxation time.** Solution:ρ=1/(2e²ᵗ−1). At long times r is a constant times e⁻ᵗ. The local linear time is1s for r and1/2s for ρ.

**3. Is the attracting circle periodic for μ=1,g=1,c=2,ω=2?** Solution:ρ*=1 and Ω*=0. Its points are equilibria and a rotation period is unavailable. Radial disturbances still decay locally at rate2/s.

**4. What changes when c=3 instead?** Solution:Ω*=−1rad/s, so rotation reverses. The period is positive2πs. The radial stability does not change with c.

**5. Why does an exactly zero initial state fail to leave the origin for μ>0?** Solution:The right-hand side vanishes there. Uniqueness and instability are compatible. A small positive initial amplitude grows, but it defines a different initial preparation.

**6. Can one invariant-circle trajectory identify μ,g,ω,c individually?** Solution:No. It restricts the combinations μ/g and ω−cμ/g. Recovery and phase-speed measurements at other amplitudes add information. A small error over a short observation window also does not establish the exact sign of the critical parameter.

## 14. Numerical evidence and research connections

The production calculation uses exact rational sign/equality and directed256-bit exponential, logarithm, square-root and trigonometric enclosures. Large accumulated angles are neither wrapped nor used to renormalize the state. Independent enclosures for x²+y²−ρ and the log-amplitude balance are distinguished from algebraic identities. Record density does not change physical time or the analytic flow.

The independent reference imports no production modules. It computes a2×2matrix exponential for reciprocal squared amplitude at220digits. Finite reference cases are not a machine proof for all inputs. Reader, Worker and visualization registration are still pending, as are the pairwise and collective families.

For AI research, distinguish rotating latent dynamics, sustained amplitude and recovery after perturbation. This toy model is not a universal model of neural networks or measured systems. Next we study relative phase after amplitude stabilization and the information lost by collective order observations.
