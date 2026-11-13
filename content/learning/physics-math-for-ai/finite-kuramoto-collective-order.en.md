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

## Research extension: zero collective order can hide growing disturbances

### A stationary picture is not a recovery experiment

The balanced configurations earlier in this chapter rotate without changing shape. That exact solution answers an existence question. To ask whether a nearby configuration returns, perturb each phase and differentiate the equations. The resulting Jacobian describes the initial growth or decay of small disturbances. Equal order observations can conceal different growth rates.

For a frequency-locked solution, write θᵢ(t)=Ωt+αᵢ in a common rotating frame. Summing the equations first fixes Ω. The remaining equations must hold at the proposed phase configuration; an arbitrary snapshot is not a locked solution.

$$
\Omega=\overline\omega,\qquad
0=\omega_i-\overline\omega+\frac K N\sum_j\sin(\alpha_j-\alpha_i),\qquad
\theta_i=\Omega t+\alpha_i.
$$

Put θᵢ=Ωt+αᵢ+ηᵢ and keep terms linear in η. The Jacobian J is real symmetric for this all-to-all sine model. Its row sums vanish because a common phase shift changes no relative angle.

$$
\dot\eta=J\eta,\qquad
J_{ij}=\frac K N\cos(\alpha_j-\alpha_i)\ (i\ne j),\qquad
J_{ii}=-\frac K N\sum_{j\ne i}\cos(\alpha_j-\alpha_i),\qquad J\mathbf1=0.
$$

One zero eigenvalue always represents common rotation. Stability of relative phases concerns the transverse subspace whose components sum to zero. Additional zero eigenvalues need further analysis; do not silently classify them as attracting.

### Positive edge weights give a local sufficient condition

Group the terms for each unordered pair. The Jacobian has the quadratic form of a negative weighted graph Laplacian. Cosines can be negative, so the word Laplacian alone does not establish stability.

$$
\eta^\mathsf{T}J\eta
=-\frac K N\sum_{i<j}\cos(\alpha_j-\alpha_i)(\eta_i-\eta_j)^2.
$$

If all phases fit in an arc of width D<π/2, every cosine is at least cos D>0. For zero-mean η, the sum of squared pair differences is N times its squared norm. Consequently the linearized perturbation obeys

$$
\eta^\mathsf{T}\mathbf1=0\Longrightarrow
\eta^\mathsf{T}J\eta\leq-K\cos D\,\|\eta\|^2,\qquad
\|\eta(t)\|\leq e^{-K\cos D\,t}\|\eta(0)\|\quad(K>0).
$$

This is a sufficient local test at an existing locked solution, not a proof that a solution exists for every frequency list or that a large disturbance returns. Other stable configurations need not satisfy this simple arc criterion. For identical natural frequencies, the aligned state gives the exact transverse rates.

$$
J_{\rm aligned}=-K\left(I-\frac{\mathbf1\mathbf1^\mathsf{T}}N\right),\qquad
\lambda=0\text{ once},\quad -K\text{ with multiplicity }N-1.
$$

The common phase remains free while all relative perturbations decay locally at rate K. For N=1 there is no transverse direction. At K=0 there is no such attraction.

### Balanced states reduce to a two-dimensional Gram matrix

Now assume identical natural frequencies, N≥2 and exact Z₁=0. Every coupling sum vanishes, so the configuration is a relative equilibrium. Let cᵢ=cosαᵢ and sᵢ=sinαᵢ. Balance makes the full cosine row sum zero. Including its diagonal entry then gives a particularly simple Jacobian.

$$
J_{\rm balanced}=\frac K N(cc^\mathsf{T}+ss^\mathsf{T}),\qquad
\operatorname{rank}J\leq2,\qquad
c^\mathsf{T}\mathbf1=s^\mathsf{T}\mathbf1=0.
$$

The nonzero eigenvalues of BBᵀ and BᵀB agree, with B=[c s]. The smaller Gram matrix depends only on the second harmonic. Use cos²α=(1+cos2α)/2 and 2sinα cosα=sin2α.

$$
\frac K N B^\mathsf{T}B
=\frac K2\begin{pmatrix}1+\operatorname{Re}Z_2&\operatorname{Im}Z_2\\
\operatorname{Im}Z_2&1-\operatorname{Re}Z_2\end{pmatrix},\qquad
Z_2=\frac1N\sum_j e^{2i\alpha_j},\qquad R_2=|Z_2|.
$$

Its two eigenvalues determine the possible growing modes. Zero entries in this pair add to the zero modes of the full matrix.

$$
\lambda_+=\frac K2(1+R_2),\qquad
\lambda_-=\frac K2(1-R_2),\qquad
\lambda_{3},\ldots,\lambda_N=0\quad\text{as a multiset for }N\geq2.
$$

At K>0 every exactly balanced state has at least one positive transverse eigenvalue, so it is unstable. For equal antiphase groups, R₂=1 gives one growth rate K. A regular polygon with N≥3 has R₂=0 and two growth rates K/2. The remaining zeros do not remove these growing directions. For N=2 the balanced antipodal pair has rates K and 0; there are no further entries.

The spectrum formula requires exact first-harmonic balance and identical frequencies. R₂ alone does not determine the stability of a generic state. The aligned state also has R₂=1 yet has negative transverse rates. Thus the additional observation is useful only together with the hypotheses under which it identifies a property.

### Four populations with reproducible rates

Take N=4, K=3 and identical natural frequencies. Represent a phase by its unit vector (cosα,sinα). The tilted balanced configuration consists of two antipodal pairs whose directions have cosine 3/5. It has exact Z₁=0 and R₂=3/5. These specified configurations are distinct from random finite populations.

![Three balanced four-oscillator configurations and the two possible growth rates as a function of second-harmonic order](/assets/learning/synchronization-statistics-research-balanced-spectrum.en.svg)

The upper panels show the antiphase, square and tilted-pair configurations. All have R₁=0. The lower panel plots their normalized growth rates from the Gram matrix. The aligned comparison in the table is locally attracting in relative phase; the three balanced configurations are unstable despite their exact constant shapes.

```python
from math import hypot

cases = {
    "aligned": [(1, 0)] * 4,
    "antiphase": [(1, 0), (1, 0), (-1, 0), (-1, 0)],
    "square": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "tilted": [(1, 0), (-1, 0), (0.6, 0.8), (-0.6, -0.8)],
}
for label, points in cases.items():
    n = len(points)
    first = hypot(sum(x for x, y in points), sum(y for x, y in points)) / n
    second = hypot(sum(x*x-y*y for x, y in points), sum(2*x*y for x, y in points)) / n
    rates = (-3.0, -3.0) if label == "aligned" else (1.5*(1+second), 1.5*(1-second))
    print(f"{label} {first:.6f} {second:.6f} {rates[0]:.6f} {rates[1]:.6f}")
```

| Configuration | R₁ | R₂ | First indicated rate | Second indicated rate |
| --- | --- | --- | --- | --- |
| aligned | 1.000000 | 1.000000 | -3.000000 | -3.000000 |
| antiphase | 0.000000 | 1.000000 | 3.000000 | 0.000000 |
| square | 0.000000 | 0.000000 | 1.500000 | 1.500000 |
| tilted | 0.000000 | 0.600000 | 2.400000 | 0.600000 |

Rates are in s⁻¹. One common-phase zero is omitted from each row. The third transverse rate is −3 for alignment and 0 for each balanced example. The code uses the derived formulas for these preparations; an independent check diagonalizes the full Jacobian and compares perturbations of the original sine equations. Positive eigenvalues describe local growth, not unbounded exponential separation after a nonlinear trajectory has left the neighborhood.

### Research self-checks

#### 1. Does R₁=0 determine the fastest disturbance growth?

**Research explanation.** No. The antiphase and square examples both have R₁=0, but their largest rates are K and K/2. At exact balance with identical frequencies, R₂ supplies the missing spectral information. Outside those assumptions, inspect the full configuration and Jacobian instead.

#### 2. Does a decreasing potential make a balanced state stable?

**Research explanation.** No. The earlier potential is −KN R₁²/2. Balanced states have value 0, while nearby states with nonzero R₁ have lower potential. A trajectory can move away while the potential decreases. A constant potential along one exact trajectory proves neither return nor stability.

#### 3. What does the common-phase zero eigenvalue mean?

**Research explanation.** It expresses rotational symmetry. Aligned populations can recover their relative phases without recovering one prescribed absolute phase. Additional zeros at balanced states are not equivalent to this symmetry, and the positive rates already prove instability. At K=0 every rate is zero, requiring a separate neutral classification.

[The finite-size and stability discussion in Acebrón et al., section II.B](https://doi.org/10.1103/RevModPhys.77.137) supplies the model context. The finite-N quadratic form, balanced Gram spectrum and examples above are derived directly here. They do not assume the infinite Lorentzian reduction or establish stability for arbitrary learned networks. The existing reader retains its original population experiment; these spectral calculations are separate executable examples.

## 14. Exercises with explanations

**1. Hidden antiphase order.** Find first and second order for phases 0,0,pi,pi. Solution: the first arrows sum to zero, while doubling every angle makes all four arrows equal to 1. Thus first order is 0 and second order is 1. A first-harmonic mean direction is undefined.

**2. Alignment and speed.** What are initial first order and actual velocities for aligned phases and natural frequencies 1 and 3? Solution: first order is 1. Every sine interaction is zero, so velocities are 1 and 3. Alignment at one instant cannot establish frequency locking.

**3. Mean conservation and pair reduction.** Take natural frequencies 3 and 0, population 2 and coupling 5. Solution: mean velocity is 3/2 and relative velocity is 3 minus 5 times the relative-phase sine. Giving each edge coefficient 5 instead of 5/2 produces a different model with relative coefficient 10.

**4. Grid diagnostics.** Suppose successive differences are 0.016 and 0.001 radians. Solution: their ratio is 16 and the formal estimate is 1/15000 radian. This is consistent with fourth-order error behavior but does not guarantee it. Use an independent reference or further refinement. More display records cannot replace that check.

**5. Time scales on both sides.** Compare width 1 with coupling 3 and coupling 1. Solution: at coupling 3 the positive equilibrium is 1 over the square root of 3 and its radial relaxation time is 1. At coupling 1 the origin decays at rate 1/2 with time 2. Copying one branch formula to the other gives a wrong answer.

**6. Finite arrays and continuous references.** Must every finite population approach first order 1 over the square root of 3 when the supplied reference width is 1 and coupling is 3? Solution: no. An exact quarter-root configuration with identical frequencies retains first order 0 and fourth order 1. The infinite Lorentzian branch assumes a different preparation. A numerical departure from symmetry alone measures neither physical noise nor a finite-population critical value.
