# Finite Kuramoto populations: order, locking and limits

This manuscript accompanies a numerical model under review. It is not yet registered as an Interactive Book unit and does not itself provide a browser lab.

## 1. Sharing a direction and sharing a speed

Imagine people walking around a circular track. They can gather at one point and immediately separate because they walk at different speeds. Two groups on opposite sides can instead maintain their separation while walking at the same speed. Instantaneous alignment in the first scene and common speed in the second are different observations.

Here each oscillator has a fixed amplitude and a changing phase. The previous Hopf amplitude model suggests a phase reduction when amplitude recovery is sufficiently fast and coupling appropriately weak. That does not automatically derive sinusoidal coupling for every oscillator. We specify the coupling and its assumptions first.

## 2. Why divide the coupling by N?

Write the phase of oscillator i as theta and its natural angular frequency as omega. Natural frequency means its speed without coupling. Actual speed generally changes when coupling is present. Our population is fully connected, with a zero self-interaction term.

$$
\dot\theta_i=\omega_i+\frac{K}{N}\sum_{j=1}^{N}\sin(\theta_j-\theta_i),\qquad K\geq0.
$$

Phase is dimensionless and time is measured in seconds. Both omega and K have units of inverse seconds. Dividing by N keeps the total influence on one oscillator on the scale of K as the population grows. Omitting it changes the coupling strength while changing population size. The computation contract admits N=1 through 32, natural frequencies of magnitude at most 8, K up to 16, and observation times up to 4 seconds. These are implementation limits rather than limits of the theory.

## 3. What the complex average shows

Turn each phase into a unit arrow and average the arrows. The length measures alignment, and the direction is a mean phase. Long individual arrows can cancel one another.

$$
Z_m=\frac1N\sum_{j=1}^N e^{im\theta_j},\qquad R_m=|Z_m|,\qquad \Psi_m=\arg Z_m\quad(Z_m\ne0).
$$

The first harmonic averages the original arrows. The second doubles their angles before averaging. Opposite groups therefore cancel in the first average and overlap in the second. An exactly zero first complex average has no mean direction. Retaining an old angle or presenting the direction of a tiny numerical residual as meaningful would mislead the reader. A small numerical magnitude is not automatically an exact zero either.

## 4. One order parameter cannot identify a population

All four oscillators in the table have natural frequency 1 and K=3. The phases are initial values; each subsequently increases by t. The zeros and ones follow from exact symmetry identities.

| Initial phases | R₁ | R₂ | Another observation | All actual speeds |
| --- | --- | --- | --- | --- |
| 0, 0, 0, 0 | 1 | 1 | R₄=1 | 1 |
| 0, 0, π, π | 0 | 1 | R₄=1 | 1 |
| 0, π/2, π, 3π/2 | 0 | 0 | R₄=1 | 1 |

The last two rows have the same first order parameter and different group structure. Even the second harmonic misses the regular arrangement in the last row. Inspect points and higher harmonics together. This table does not establish equal stability: existence of an exact symmetric solution and return after a small perturbation are separate questions.

## 5. Instantaneous speed and a finite observation window

Distinguish natural frequency, instantaneous velocity and average velocity over a finite window. Wrapping a phase discards completed turns, so window averages must use the continuous phase lift.

$$
\overline\Omega_i(T)=\frac{\theta_i(T)-\theta_i(0)}{T}\ (T>0),\qquad S_{\rm inst}(t)=\max_i\dot\theta_i(t)-\min_i\dot\theta_i(t).
$$

At zero time the window average is undefined and is returned as an empty value. Similar velocities during a short window do not prove asymptotic locking. Initially aligned oscillators have first order parameter 1, but all initial coupling terms vanish. Different natural frequencies immediately give them different actual speeds. Conversely, the balanced groups in the table have zero first order and exactly equal velocities. Measurements also require an explicit window and uncertainty model.

## 6. Why the mean velocity is conserved

The sine contributions from a pair have opposite signs. Adding all equations cancels internal interactions. This does not say that every individual velocity equals its natural frequency.

$$
\frac1N\sum_i\dot\theta_i=\frac1N\sum_i\omega_i,\qquad \frac1N\sum_i\theta_i(t)=\frac1N\sum_i\theta_i(0)+t\frac1N\sum_i\omega_i.
$$

The implementation calculates each pair force once and adds and subtracts it. Floating-point sums can still leave residuals. The exact mean-frequency identity and the observed numerical balance residual are returned separately. The integrator does not recenter phases at every step to hide errors. Adding one constant to all natural frequencies changes common rotation while preserving relative phases and harmonic magnitudes.

## 7. Exact symmetry and numerical symmetry errors

Balanced m-group profiles with identical frequencies place equal numbers of points at the vertices of a regular polygon. Their complex sum vanishes, so each point receives zero total coupling. The whole configuration rotates at its common natural frequency. The fully aligned one-group profile also retains its shape.


$$
\theta_j(t)=\theta_j(0)+\omega_0t,\qquad \sum_{q=0}^{m-1}e^{2\pi i q/m}=0\ (m>1).
$$

These identities are recognized from exact integer relations when initial phases are supplied as rational multiples of pi. Nearby phases are not rounded onto a symmetric profile. A small perturbation is a different preparation. When K=0, even heterogeneous natural frequencies allow independent analytic rotations.

An unmodified RK 4 trajectory is retained separately. With zero natural frequencies, initial phases 0 and pi, K=16 and time 4, the exact antiphase solution is stationary. A floating-point sine residual at pi can seed an unstable direction and grow into a large numerical departure. This comparison does not introduce physical noise. A noise experiment needs its own stochastic model and specified strength.

## 8. Three grids do not prove an error bound

Generic preparations are integrated with classical RK 4 on n, 2 n and 4 n steps; the finest trajectory supplies the primary numerical result. Unwrapped phases are compared at shared physical times.

$$
d_1=\max_{i,k}|\theta_i^{(n)}(t_k)-\theta_i^{(2n)}(t_k)|,\quad d_2=\max_{i,k}|\theta_i^{(2n)}(t_k)-\theta_i^{(4n)}(t_k)|,\quad E_{\rm formal}=d_2/15.
$$

If the leading error is proportional to the fourth power of step size, the ratio of successive differences approaches 16 and the finest error can be estimated by the second difference divided by 15. Whether those assumptions apply is another question. Against an independent DOP 853 reference, a deliberately coarse preparation has phase error greater than 1 radian and more than ten times the formal estimate. Producing an estimate does not validate a trajectory.

The number S of display intervals differs from the integration step count. Record i selects index floor(in/S) on the base grid. When division is not exact, record spacing can vary; actual times must be shown. Changing record density preserves the endpoint and global refinement diagnostics. The independent reference is itself a floating-point numerical solution with tight tolerances, rather than a rigorous enclosure for all inputs.

## 9. A decreasing function for identical frequencies

When every natural frequency is identical, remove that common rotation and use the resulting phases phi. The remaining coupling is a gradient flow.

$$
V=-\frac{K}{2N}\sum_{i,j}\cos(\phi_j-\phi_i)=-\frac{KN}{2}R_1^2,\qquad \dot\phi_i=-\frac{\partial V}{\partial\phi_i},\qquad \dot V=-\sum_i\dot\phi_i^2\leq0.
$$

Differentiate to check the sine sign and normalization. A decreasing potential does not mean every initial configuration becomes fully aligned. Exact balanced profiles also have zero relative velocity. With different natural frequencies, the final identity does not apply unchanged. This is a dynamical potential of a dimensionless phase model; physical heat, energy and entropy have not been specified.

## 10. Recovering the two-oscillator model

Set N=2 and compare the coefficients with the preceding unit. Each directed interaction now has coefficient K/2.

$$
\delta=\theta_1-\theta_2,\quad \Delta=\omega_1-\omega_2,\qquad \dot\delta=\Delta-K\sin\delta,\qquad R_1^2=\frac{1+\cos\delta}{2}.
$$

Compare stable and unstable locked phases above the mismatch, the critical bottleneck at equality, and slipping below it with the previous analytic solution. First order only reveals the cosine of the relative phase; it loses its sign and accumulated turns. Missing the factor of one half changes the locking threshold and relaxation rate. This reduction therefore checks a consequential modeling convention.

## 11. A separate infinite Lorentzian comparison

Now introduce assumptions different from a finite array. Specify a Lorentzian natural-frequency density in an infinite population with center and positive width. Its center is not an arithmetic mean: the Cauchy distribution has no defined mean. Keep it distinct from the empirical mean of the finite array.

$$
g(\omega)=\frac{\gamma}{\pi[(\omega-\Omega_c)^2+\gamma^2]},\qquad \dot R=\left(\frac K2-\gamma\right)R-\frac K2R^3.
$$

This reduction requires an infinite population, a particular invariant manifold and analyticity assumptions. Sections II–III of the [Ott–Antonsen original paper](https://arxiv.org/pdf/0806.0004) specify the power relation between Fourier coefficients of the phase density and conditions in the complex frequency plane. An arbitrary small finite population does not obey this equation exactly. The reference width is supplied separately; it is not fitted to the finite frequency list.

## 12. Deriving onset and relaxation times

Set the linear coefficient to K/2 minus the width. Solve for a positive equilibrium and linearize around that branch.

$$
K_c=2\gamma,\quad R_*^2=1-\frac{2\gamma}{K}\ (K>2\gamma),\quad f'(R_*)=-(K-2\gamma),\quad \tau_+=\frac1{K-2\gamma}.
$$

The comparison table uses width 1. Each relaxation time concerns small perturbations of the specified equilibrium.

| K | μ | Positive equilibrium R*² | Applicable exponential relaxation time |
| --- | --- | --- | --- |
| 1 | −1/2 | absent | 2 at the origin |
| 2 | 0 | absent | undefined at criticality |
| 3 | 1/2 | 1/3 | 1 on the positive branch |

The decay rate below onset is the width minus half the coupling. On the positive branch it is coupling minus twice the width. Mixing them gives a factor-of-two error. At criticality the linear rate vanishes and the remaining cubic equation gives algebraic decay from positive initial order. Do not replace the absent exponential time with zero or an arbitrary large number. Threshold comparisons use exact rational inputs even when two rounded machine values appear equal.

## 13. What can an AI representation infer?

Reducing a population to a statistic resembles learning a representation. A model receiving only first order cannot distinguish the two balanced configurations in the first table. That is information removed by the observation map. Add second and fourth harmonics, the point configuration and velocities over specified windows to discover which hypotheses become distinguishable.

Collective order in the Lorentzian reduction and individual amplitude in the Hopf normal form are different observables. A shared cubic saturation term does not license exchanging their meanings. A zero-center order-parameter bifurcation is not automatically an ordinary Hopf bifurcation either. Graphs, inertia, delay, noise, general coupling, finite-size effects, other frequency distributions, stability and the order of limits remain research topics. Passing tests for this small model does not complete those topics.

## 14. Exercises with explanations

**1. Hidden antiphase order.** Find first and second order for phases 0,0,pi,pi. Solution: the first arrows sum to zero, while doubling every angle makes all four arrows equal to 1. Thus first order is 0 and second order is 1. A first-harmonic mean direction is undefined.

**2. Alignment and speed.** What are initial first order and actual velocities for aligned phases and natural frequencies 1 and 3? Solution: first order is 1. Every sine interaction is zero, so velocities are 1 and 3. Alignment at one instant cannot establish frequency locking.

**3. Mean conservation and pair reduction.** Take natural frequencies 3 and 0, population 2 and coupling 5. Solution: mean velocity is 3/2 and relative velocity is 3 minus 5 times the relative-phase sine. Giving each edge coefficient 5 instead of 5/2 produces a different model with relative coefficient 10.

**4. Grid diagnostics.** Suppose successive differences are 0.016 and 0.001 radians. Solution: their ratio is 16 and the formal estimate is 1/15000 radian. This is consistent with fourth-order error behavior but does not guarantee it. Use an independent reference or further refinement. More display records cannot replace that check.

**5. Time scales on both sides.** Compare width 1 with coupling 3 and coupling 1. Solution: at coupling 3 the positive equilibrium is 1 over the square root of 3 and its radial relaxation time is 1. At coupling 1 the origin decays at rate 1/2 with time 2. Copying one branch formula to the other gives a wrong answer.

**6. Finite arrays and continuous references.** Must every finite population approach first order 1 over the square root of 3 when the supplied reference width is 1 and coupling is 3? Solution: no. An exact quarter-root configuration with identical frequencies retains first order 0 and fourth order 1. The infinite Lorentzian branch assumes a different preparation. A numerical departure from symmetry alone measures neither physical noise nor a finite-population critical value.
