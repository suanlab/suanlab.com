## 1. A good amplitude can hide the wrong clock

Two bounded oscillations can have almost the same amplitude and energy but reach their peaks at different times. A tiny frequency error repeated over many cycles becomes a large phase error. A perturbation expansion that is accurate at a fixed time may therefore fail when we follow the motion for longer.

Think of comparing two clocks that differ by a small amount per day. Neither clock hand needs to grow without bound for the clocks to disagree. A growing term in an expansion can be the footprint of a shifted clock, rather than a growing physical oscillation.

We will compare a naive time expansion, two frequency-corrected approximations and an independently enclosed nonlinear periodic reference. No numerical time step is labelled an exact solution.

## 2. Declare the oscillator and its effective coupling

The dimensionless initial-value problem is

$$
 x''+x+\lambda x^3=0,\qquad x(0)=A\ge0,\quad x'(0)=0,\quad\lambda\ge0.
$$

The mass and linear stiffness have been scaled so that the linear angular frequency is 1. Time is measured in those units. For A>0, set x=Aq and define

$$
 g=\lambda A^2,\qquad q''+q+gq^3=0,\quad q(0)=1,\quad q'(0)=0.
$$

Small λ alone does not guarantee weak nonlinearity: amplitude matters. With λ=0.01, A=1 gives g=0.01, whereas A=4 gives g=0.16. At A=0 the physical state is the equilibrium x=0. Its trajectory has no physical oscillation period, even though the linearization has frequency 1.

## 3. Establish what the true motion can do

Multiplying the equation by x′ gives conserved energy

$$
 E=\frac{x'^2}{2}+\frac{x^2}{2}+\frac{\lambda x^4}{4}
 =\frac{A^2}{2}+\frac{\lambda A^4}{4}.
$$

For λ≥0 the potential increases with |x|. Consequently |x|≤A and |x′|≤A√(1+g/2). A nonzero turning amplitude produces a periodic orbit between −A and A. These facts provide range checks before any approximation is trusted.

A term that grows like t in a perturbative formula cannot represent an indefinitely growing amplitude of this conservative orbit. It signals a limitation of that formula. This conclusion would need reconsideration for forcing, damping or a different potential.

## 4. Derive the first naive correction

Try q=q₀+gq₁+O(g²) while keeping the original time coordinate. The leading solution is q₀=cos t. At the next order,

$$
 q_1''+q_1=-\cos^3t=-\frac34\cos t-\frac14\cos3t,
 \qquad q_1(0)=q_1'(0)=0.
$$

The forcing contains cos t, a homogeneous solution of the same differential operator. This resonance generates a term proportional to t sin t. Imposing the initial conditions yields

$$
 q_{\mathrm N}(t)=\cos t+
 g\left[\frac{\cos3t-\cos t}{32}-\frac38t\sin t\right].
$$

The third harmonic is a bounded shape correction. The second contribution is secular: its envelope grows linearly in time. Distinguish the correction itself from its envelope; sin t vanishes at some times even when the approximation has lost uniform validity.

## 5. Read the nonuniform limit correctly

For each fixed t the correction tends to zero with g. On intervals whose length grows with 1/g, the product gt need not be small. An elementary bound is

$$
 |q_{\mathrm N}(t)-\cos t|\le
 g\left(\frac1{16}+\frac38|t|\right).
$$

This is a bound on the displayed correction, not on the error relative to the exact Duffing solution. For physical displacement multiply it by A.

The clock example cos((1+α)t)=cos t−αt sin t+… exposes the same mechanism. Expanding a shifted phase generates a secular term even though the unexpanded function is bounded. Keeping that phase shift inside a cosine is a reorganization of the approximation.

## 6. Let the frequency change instead of expanding the clock

Introduce τ=ωt and seek a function periodic in τ. Now q″ in physical time becomes ω²q_ττ. Set ω=1+ω₁g+… and q=cos τ+gY₁(τ)+… . The first correction satisfies

$$
 Y_1''+Y_1=
 \left(2\omega_1-\frac34\right)\cos\tau-\frac14\cos3\tau.
$$

A periodic correction cannot contain resonant forcing at frequency 1. Cancel it by choosing ω₁=3/8. The remaining third harmonic gives a bounded waveform correction. This is the Poincaré–Lindstedt construction: remove the secular forcing by choosing the clock consistently.

## 7. The first frequency-corrected waveform

The resulting approximation is

$$
 \omega_1^{\mathrm{app}}=1+\frac38g,\qquad
 q_{\mathrm{PL1}}(t)=\cos\tau+
 \frac{g}{32}(\cos3\tau-\cos\tau),\quad
 \tau=\omega_1^{\mathrm{app}}t.
$$

All occurrences of τ use the same corrected clock. Differentiating for velocity requires the factors ω and 3ω. Correcting only the main cosine while leaving the third harmonic on the old clock would be a different approximation.

The waveform is bounded for fixed g and satisfies the turning-point initial conditions. Neither boundedness nor the initial conditions prove that its frequency is accurate for arbitrarily long times.

## 8. Continue the calculation to second order

At the next order, removing the resonant cosine gives

$$
 \omega_2^{\mathrm{app}}=1+\frac38g-\frac{21}{256}g^2,
 \qquad
 Y_2(\tau)=\frac{23\cos\tau-24\cos3\tau+\cos5\tau}{1024}.
$$

Use τ=ω₂_app t in both the leading and first-order terms, then add g²Y₂. The coefficients 23−24+1=0 preserve q(0)=1. All sine derivatives vanish initially, preserving q′(0)=0.

The second-order frequency is a truncated weak-coupling expansion. At g=16 it equals −14, while the exact hardening oscillator still has a positive frequency. The calculator retains the formal expression but marks its frequency interpretation invalid; it does not declare the physical orbit absent.

## 9. Multiple scales gives a complementary explanation

Let T=gt and treat t and T as independent variables during the expansion. Write q₀=a(T)cos(t+φ(T)). Using d/dt=D₀+gD₁ gives the first correction equation (D₀²+1)q₁=−2D₀D₁q₀−q₀³.

Its resonant sine and cosine coefficients vanish when

$$
 \frac{da}{dT}=0,\qquad \frac{d\phi}{dT}=\frac38a^2.
$$

For the normalized initial amplitude a(0)=1, the leading phase is t+3gt/8, agreeing with the first Lindstedt clock. Constant amplitude reflects conservative energy exchange; the phase changes slowly.

This example explains solvability and slow modulation. It is not a proof of an averaging theorem for arbitrary systems or arbitrarily long intervals. Damping, forcing and additional resonances change the modulation equations.

## 10. A small residual is a separate certificate

For a finite waveform Y(τ), define its physical equation residual by substituting x=A Y(ωt) into x″+x+λx³. In normalized form,

$$
 R(\tau)=\omega^2Y''+Y+gY^3.
$$

Cosines of odd multiples are Chebyshev polynomials of z=cos τ. Thus R is a finite polynomial in z, of degree at most 15 for the second-order waveform. If R(z)=∑rⱼzʲ, then |R|≤∑|rⱼ| because |z|≤1. The physical residual bound is A times that normalized bound.

The first and second constructions cancel powers below g² and g³ respectively. A residual bound controls how accurately the equation is satisfied. It does not, by itself, control accumulated phase or the whole-trajectory error.

## 11. Define the phase diagnostic before comparing it

Let ν be the exact fundamental angular frequency. At a horizon H the calculator reports

$$
 \Delta\Phi_j(H)=(\omega_j^{\mathrm{app}}-\nu)H.
$$

For the naive clock use ω_app=1. This is an unwrapped fundamental-phase difference. It is not atan2(x,x′), not a displacement error and not reduced modulo 2π. Nonpositive truncated frequencies are formal expressions with invalid physical-frequency interpretation.

At g=0.01 the reference frequency is approximately 1.00374183617774233. The first approximation is 1.00375; the second is 1.003741796875. Rounded illustrative phase differences are:

| H | Naive clock | First corrected clock | Second corrected clock |
|---:|---:|---:|---:|
| 100 | −0.374184 | 0.000816382 | −0.00000393027 |
| 10,000 | −37.4184 | 0.0816382 | −0.000393027 |
| 1,000,000 | −3741.84 | 8.16382 | −0.0393027 |

Read exact enclosures from the result tables. These values demonstrate that a bounded waveform can still drift away from the correct timing.

## 12. Why the relevant time scales change with order

The exact weak-coupling frequency starts as 1+3g/8−21g²/256+… . The naive frequency error begins at order g, the first corrected error at g² and the second at g³. Multiplying by time suggests nonvanishing accumulated phase on scales 1/g, 1/g² and 1/g³ respectively.

These are asymptotic scaling statements for weak coupling with other data fixed. Their constants matter: a phase of 0.04 radians and one of 4 radians are different practical errors even though both can be order one in an asymptotic statement. The interval diagnostic at the requested horizon is more informative than applying a slogan about “long-time accuracy.”

## 13. Build an exact periodic reference

For this initial-value family the analytic solution can be written

$$
 x(t)=A\operatorname{cn}\!\left(\sqrt{1+g}\,t\mid m\right),
 \qquad m=\frac{g}{2(1+g)}.
$$

Here m is the elliptic parameter. Sources using a modulus k write m=k²; confusing the two changes the result. The differential identity for cn verifies the Duffing equation after substituting this parameter. Its initial value and derivative give A and 0.

The complete elliptic integral K(m) determines the period: T=4K(m)/√(1+g). The reference is a nonlinear periodic waveform, not simply a cosine at an improved frequency. Waveform and clock corrections are distinct.

## 14. Enclose the frequency using arithmetic and geometric means

For M=AGM(1,√(1−m)), K=π/(2M), so

$$
 \nu=\sqrt{1+g}\,M,\qquad T=\frac{2\pi}{\nu}.
$$

AGM repeatedly replaces two positive numbers by their arithmetic and geometric means. The exact geometric sequence approaches the limit from below and the arithmetic sequence from above. Directed intervals preserve those bounds even with rounding.

The implementation uses 18 iterations and reports the resulting interval. Square-root arithmetic and the nome calculation use 512-bit components; trigonometry uses a 768-bit working grid. Final returned values are rounded outward to a 192-bit grid. These are arithmetic settings, not a blanket guarantee of 192 correct relative bits.

## 15. Enclose the waveform without accumulating time steps

The nome Q=exp(−πK(1−m)/K(m)) lies below 1/16 for our range 0<m<1/2. With C=4M√(Q/m), the cn waveform has the convergent representation

$$
 \frac{x(t)}A=C\sum_{n=0}^{\infty}
 \frac{Q^n}{1+Q^{2n+1}}\cos((2n+1)\nu t).
$$

Keeping N=64 terms gives a position-tail bound A C Qᴺ/(1−Q). Differentiation introduces (2n+1)ν; summing the corresponding geometric tail gives a separate velocity-tail bound. Arithmetic intervals are propagated in addition to these tails.

This reference does not integrate a small time step repeatedly to reach H. Its phase uncertainty still grows with time, and its returned interval must still be read. Independent checks use direct high precision elliptic cn/sn/dn rather than this Fourier construction.

## 16. Energy accuracy does not determine phase accuracy

A point on the right energy curve can be at the wrong location along that curve. Consequently nearly conserved approximate energy can coexist with a substantial timing error. Conversely a large energy defect is useful evidence that an approximation has lost other properties as well.

The probe tables report position, velocity, their errors relative to the periodic reference, equation residual, energy and energy defect separately. The exact reference conserves the initial energy. Do not replace one diagnostic with another or identify a phase difference with a displacement bound.

## 17. A long record needs a locally resolved view

The full record contains 65 time samples over [0,H]. At a very large horizon these samples cannot show every oscillation. Joining them with a smooth-looking line can manufacture an apparent period.

A second 65-point window is placed around the probe, clipped to the observation interval. Its width is at most 32/ceil(ν_upper), chosen to display several cycles with a useful number of samples. The full-grid flag checks a conservative eight-points-per-cycle criterion. These are sampling conventions, not ODE discretization error proofs.

Use full-horizon phase diagnostics for accumulation and the local waveform view for shape and timing. Neither a sparse picture nor a visually closed phase portrait certifies the entire continuous trajectory.

## 18. Respect special cases and input limits

The inputs are nonnegative rational λ,A≤16, a horizon H≤999999999999 and a probe fraction in [0,1]. Numerators and denominators have at most 12 digits. The effective coupling can reach 4096; strong examples deliberately expose approximation failure.

Zero amplitude has no period. Zero λ at nonzero amplitude is the harmonic limit and all three approximations coincide with the exact solution. At H=0 the initial state and zero approximation errors are treated algebraically. Tiny positive coupling is not rounded to zero, including at very long horizons.

The model excludes damping, periodic driving, negative-coupling escape, multiple wells and chaos. Those phenomena need different assumptions and reference problems.

## 19. Worked questions with explanations

1. **Amplitude dependence.** For λ=0.01 compare A=1 and A=4. Their effective couplings are 0.01 and 0.16, so the second experiment is less weakly nonlinear despite the same λ.
2. **Find the secular forcing.** Expand cos³t=(3cos t+cos3t)/4. The cos t term resonates with q₁″+q₁, creating t sin t. The third harmonic alone would remain bounded.
3. **Derive the first clock.** The resonant coefficient is 2ω₁−3/4. Setting it to zero gives ω₁=3/8, rather than choosing a frequency by fitting a plot.
4. **Preserve initial data.** The second waveform correction has coefficients 23,−24,1. Their sum is zero, and sine derivatives vanish at t=0. This checks the turning-point conditions, not the entire equation.
5. **Estimate accumulated drift.** At g=0.01 and H=10,000, the first clock drifts by about 0.08164 radians. At H=1,000,000 it drifts by about 8.164 radians. Bounded amplitude has not prevented this increase.
6. **Interpret a negative frequency.** At g=16 the second truncated expression is −14. The exact orbit remains periodic. The invalid approximation is not evidence that the physical oscillator has a negative angular frequency or no orbit.
7. **Separate residual from error.** A smaller polynomial residual does not give a whole-time position bound without a stability/accumulation argument. Compare its residual, phase drift and actual probe errors separately.
8. **Choose a picture.** For H=1,000,000, 65 samples cannot resolve all cycles. Use the local window to see oscillations and the unwrapped phase diagnostic to see accumulation; do not infer a slow oscillation from aliased points.

## 20. Research connections and the next scope

### Compare the four perturbation problems before choosing a tool

The same letter does not imply the same physical parameter across chapters. In the equilibrium problem g=λh²/k³ uses the unperturbed displacement; in Duffing motion g=λA² uses the prescribed turning amplitude. The quartic integral defines its own dimensionless g through gX⁴/4. The layer parameter δ=ε/|a| is a spatial width relative to a unit interval. Always restate the equation, data and limiting process when transferring an argument.

| Problem and reading | Small quantity and limit | Certificate available here | Failure that another term alone cannot fix |
|---|---|---|---|
| [Convex equilibrium](../regular-perturbation-residual-certificates/) | g=λh²/k³; fixed k>0 and h | Root bracket and displacement error ≤ residual magnitude/k | A Taylor series outside its radius; loss of the slope bound at k=0 |
| [Quartic integral](../quartic-integral-asymptotic-truncation/) | g→0⁺ at fixed retained order | Signed finite remainder and separate quadrature interval | Terms eventually grow at every fixed g>0; an unjustified infinite sum/integral exchange |
| [Boundary layer](../singular-perturbation-boundary-layer-matching/) | δ→0 with fixed nonzero drift and fixed data | Composite uniform value error = mismatch magnitude·exp(−1/δ) | The reduced equation loses a boundary condition; stretched coordinates and matching are needed |
| Duffing motion | g→0 with an explicitly chosen horizon H | Reference enclosures, residual bounds and fundamental-phase drift | A fixed-time expansion can fail on a growing time interval; the clock needs reorganization |

A useful worked comparison is to receive four proposed solver outputs and ask what evidence each needs. These numbers belong to different quantities and units; do not rank methods by comparing their magnitudes across rows.

**Equilibrium output.** At k=h=1 and λ=0.1, the second-order output p=0.93 has residual 0.0104357. The positive slope bound k=1 turns this into a displacement-error upper bound of 0.0104357. The argument requires the declared convex equation; it cannot be copied to an arbitrary differential equation.

**Integral output.** At g=0.1, S₂=0.9578125 has the first-omitted upper bound 0.0270703125. This is a finite remainder certificate even though the infinite series diverges. Its derivation uses the signed exponential remainder and finite Gaussian moments, not an equilibrium residual divided by k.

**Boundary output.** For ε=0.1, a=1, f=0, A=0 and B=1, the proposed profile u=1 has zero interior equation residual and a boundary error of one. The composite 1−exp(−10x) instead has the proved uniform value error exp(−10). Specify which boundary conditions were tested before trusting a residual score.

**Long-time output.** At g=0.01 and H=1,000,000, the bounded first Lindstedt waveform has about 8.16382 radians of fundamental-phase drift. Boundedness, a correct initial state and a small residual are distinct checks; none alone establishes that peaks occur at the correct long-time instants.

For a research comparison, first name the observable and norm, then the parameter/time or spatial regime, then the proof assumptions and reference uncertainty. Keep boundary defects, phase accumulation, truncation remainders and stability constants in the report. This four-problem diagnostic is the introductory synthesis; it does not supply WKB/Stokes analysis, general averaging theorems, resummation or continuum field-theory existence proofs.


This family is a controlled benchmark for neural ODEs, learned integrators, reduced models and long rollouts: verify initial data, invariants, local shape, derivatives, fundamental frequency and accumulated timing. Passing a short rollout does not establish success over a horizon that grows as coupling decreases.

Background on secular terms and time reorganization is in [Hunter, Chapter 5](https://math.ucdavis.edu/~hunter/notes/asy.pdf). The nonlinear reference uses [NIST DLMF §22.19(ii)](https://dlmf.nist.gov/22.19#ii), the [AGM construction](https://dlmf.nist.gov/22.20#ii) and the [cn Fourier series](https://dlmf.nist.gov/22.11.E2). Their modulus notation is distinguished from this chapter’s parameter notation.

This completes the planned types of perturbation examples only after implementation, reader verification and a scope audit. General averaging theorems, WKB/Stokes phenomena, nonlinear resonance and field-theoretic resummation remain deeper topics. The broader curriculum continues with correlation functions and the subsequent field/multiscale subjects.
