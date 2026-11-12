## Accuracy and structure answer different questions

A numerical trajectory can look smooth, retain nearly constant energy and still distort the phase flow. Conversely, an integrator can preserve the symplectic form while its physical energy oscillates. Neither a pretty orbit nor one favorable diagnostic determines whether a method is appropriate.

This lesson studies an actual nonlinear oscillator, rather than applying another constant matrix to a linear oscillator. It compares velocity Verlet, classical RK4 and a deliberately energy-corrected Verlet step. The third method makes the distinction especially clear: repairing an energy value can change the derivative of the map and can even fail to produce a real next momentum.

Every plotted numerical point is an executed update. Interrupted trajectories, unavailable derivatives and uncomputed query indices stay explicit. They are not completed by copying the last valid point.

## Define the Hamiltonian and physical units

For mass $m>0$, stiffness $k\ge0$ and quartic coefficient $a\ge0$, use

$$
U(q)=\frac k2q^2+\frac a4q^4,\qquad
H(q,p)=\frac{p^2}{2m}+U(q).
$$

Here $q$ is in m, $p$ in kg m/s, $k$ in N/m and $a$ in N/m³. The equations are $\dot q=p/m$ and $\dot p=-kq-aq^3$. Position and momentum errors have different units and must not be added as raw coordinates.

The nonnegative potential makes the physical energy a useful bound on motion when the potential grows. If both coefficients vanish, the problem becomes a free particle. Numerical blowup at a coarse step in this bounded mechanical problem is not evidence that the continuous solution blows up.

## Execute the kick–drift–kick step

With constant step size $h$, velocity Verlet updates

$$
p_{1/2}=p-\frac h2U'(q),\qquad
Q=q+\frac hmp_{1/2},\qquad
P=p_{1/2}-\frac h2U'(Q).
$$

A kick changes momentum with position fixed. A drift moves position with momentum fixed. Each substep is a canonical map, and their composition retains that property. Applying the same sequence with $-h$ reverses it in exact arithmetic.

For a fixed duration $T$ and step count $N$, set $h=T/N$. Refinement must change both step size and number of steps so that the final physical time stays the same. Comparing different durations would mix discretization error with a different physical experiment.

## Derive Verlet from a discrete action

For $h\ne0$, approximate the action between two positions by

$$
L_d(q,Q;h)=\frac{m(Q-q)^2}{2h}
-\frac h2\left[U(q)+U(Q)\right].
$$

The discrete endpoint momenta satisfy

$$
p=-D_1L_d=\frac mh(Q-q)+\frac h2U'(q),\qquad
P=D_2L_d=\frac mh(Q-q)-\frac h2U'(Q).
$$

Solving these relations recovers the kick–drift–kick formulas. Thus the method is not merely a sequence with a small measured energy error: its map follows from a discrete variational construction.

The model evaluates both discrete Legendre residuals for each candidate method. They vanish for Verlet up to roundoff. RK4 and the energy-corrected final momentum generally do not satisfy this particular discrete action relation. That does not rule out every possible alternative variational formulation; it identifies which map the declared $L_d$ generates.

## Differentiate the full numerical map

Write the one-step Jacobian in phase order $(q,p)$ as $M=D\Phi_h$. Its entries have units

$$
[M]=\begin{pmatrix}1&\mathrm{s/kg}\\\mathrm{kg/s}&1\end{pmatrix}.
$$

For one degree of freedom, $M^TJM=J$ reduces to $\det M=1$. In higher dimensions determinant one alone would not suffice. The product code propagates derivatives of every algebraic stage with forward-mode differentiation, including all dependencies of an energy correction.

A trajectory Jacobian is the ordered product of its step Jacobians. A small determinant defect measured in floating-point arithmetic is a diagnostic, not a proof over all initial conditions. Large tangent entries can also make determinant subtraction ill-conditioned. Keep the analytic method property separate from the numerical determinant trace.

## Compare high order with symplecticity

Classical RK4 evaluates four force/velocity stages and combines them with weights $1,2,2,1$. It is fourth order for sufficiently smooth solutions on a fixed interval. That often gives a smaller short-time state error than second-order Verlet at the same step size.

RK4 is not generally symplectic for this oscillator. One useful experiment therefore compares state error and Jacobian error side by side. A method need not win both comparisons. The free-particle limit is a special exception: RK4, Verlet and the stated correction all reduce to the same exact shear.

Method order is an asymptotic statement under regularity and stability assumptions. It does not promise that arbitrarily large steps, every time horizon or every roundoff-dominated comparison will exhibit the nominal error ratio.

## Repair energy while keeping the provisional position

Start with a Verlet proposal $(Q,P_v)$. Keep its position and choose a new momentum from

$$
W=H(q,p)-U(Q),\qquad
P=\operatorname{sign}(P_v)\sqrt{2mW}.
$$

This is a specific postprocessing rule, not a physical impulse derived from the original equations. It preserves the input energy when a real branch is available. If $W<0$, no real momentum at this fixed position can have the requested energy. The algorithm stops rather than shrinking the step or clamping the square root.

At $W=0$, a finite momentum $P=0$ exists, but the generic square-root derivative is unavailable. The model returns the state and a null derivative. It does not claim that every specially simplified zero-root map is mathematically nonsmooth; free motion and zero steps are handled by their explicit identity/shear formulas. If the provisional sign is zero while $W>0$, the sign prescription does not select a branch.

## Include the input-energy derivative in the correction

The energy target is the current input energy $H(q,p)$, not one external constant independent of the perturbed input. This matters when differentiating the map.

Let $A=Q_q=1-h^2U''(q)/(2m)$ and $B=Q_p=h/m$. At a regular nonzero corrected momentum,

$$
P_q=\frac mP\left[U'(q)-U'(Q)A\right],\qquad
P_p=\frac mP\left[p/m-U'(Q)B\right].
$$

Consequently

$$
\det D\Phi_h=\frac{Ap-hU'(q)}P,
$$

which need not equal one. Treating the energy target as fixed during this differentiation would describe a different map, often a projection onto one fixed energy curve. A correct energy value is therefore insufficient evidence for canonicality.

## Interpret a truncated modified Hamiltonian carefully

For velocity Verlet, a leading modified Hamiltonian in the stated convention is

$$
H_{\mathrm{trunc}}=H+h^2\left[
\frac{U''(q)p^2}{12m^2}-\frac{(U'(q))^2}{24m}\right].
$$

A symbolic expansion cancels its one-step drift through order $h^4$, leaving an $O(h^5)$ local defect. This explains why its observed drift can be substantially smaller than the physical energy drift for small fixed steps in a bounded region.

It is a truncated asymptotic expression, not an exactly preserved quantity and not a universal stability certificate. Backward error arguments require hypotheses about step size, regularity and the region visited. The model reports the actual truncated drift without declaring it zero. It also does not apply an oscillator's exact quadratic invariant to a nonlinear quartic potential.

## Establish a separate physical reference

The browser-side reference uses adaptive RK4 step doubling with Richardson correction. It compares one full step with two half steps and controls position and momentum errors separately with declared absolute and relative scales. Absolute tolerances are 1e-12 m and 1e-12 kg m/s, with relative tolerance 1e-11. These are local error estimates, not rigorous global bounds.

The independent verification uses SciPy DOP853 with tighter tolerances and a second tolerance-refinement solve. It checks the entire reference grid, not only the last sample. Symbolic map derivatives and high-precision evaluations separately check the method stages, so the reference solver is not used to certify symplecticity.

The experiment can display a dimensionless state error only after dividing position and momentum discrepancies by their reported physical scales. Raw m and kg m/s errors remain available on separate curves. A reference-budget failure remains explicit and cannot become an exact trajectory by relabeling it.

## Refine at one duration and read observed order

The model compares $N$, $2N$ and $4N$ steps over the same $T$. If terminal normalized errors are $e_N,e_{2N}$, a measured order is

$$
r=\log_2(e_N/e_{2N}).
$$

For the default nonlinear experiment, Verlet gives approximately order two and RK4 approximately order four. The corrected map need not have a uniform measured order near turning points, branch failures or changes in its sign rule.

No order is reported when a run is incomplete or when the errors lie below the declared comparison floor. That floor prevents unstable ratios near the reference/roundoff scale; it is not a theorem that every smaller error is zero.

## Test reversal as well as conservation

Run the selected method for $N$ forward steps and then $N$ steps with $-h$. Report the final position and momentum differences separately. Verlet returns to the initial state up to roundoff because its composition is symmetric.

The energy-corrected rule can have a nonzero roundtrip error even while preserving energy in both directions. It can also finish the forward run and have no real corrected momentum during the reverse run. In that case reverse errors are null with an explicit failure reason. Converting null into zero would turn an algorithmic failure into a false success.

RK4 also generally has a roundtrip defect. Reversibility, symplecticity, physical energy behavior and convergence order are related topics but distinct tests.

## Calculate a representative step and read the controls

Use $m=k=a=1$, $q_0=1$ m, $p_0=0$, $T=2$ s and $N=16$. Then $h=1/8$ s. The first half-kick gives $p_{1/2}=-1/8$, the drift gives $Q=63/64$ m, and the final Verlet momentum is approximately $-0.246139288$ kg m/s.

Initial energy is $3/4$ J; first-step energy is about $0.749526756$ J. The energy correction instead selects approximately $-0.248054504$ kg m/s, restores energy, and has one-step determinant about $1.007843$. Its value is correct for energy but different for phase geometry.

Controls use $N=4\,2^{\mathrm{level}}$, duration code divided by two seconds, and initial position/momentum codes divided by two in their SI units. Finite updates beyond |q|=1e6 m or |p|=1e9 kg m/s stop at a declared display limit; that guard is not a physical singularity. A query is an actual integer step index. An index greater than $N$, or beyond an interrupted run, is unavailable. At $T=0$ all updates are identities, while the formula for $L_d$ itself remains undefined because it divides by $h$.

## Turn these distinctions into research tests

A learned correction can reduce an energy loss while changing phase volume, reversibility or state accuracy. Record each property instead of optimizing one scalar and assuming the others follow.

For differentiable simulators, compare the differentiated implemented map with independent symbolic or high-precision derivatives. Include the input dependence of correction targets. For conservation claims, state the exact model, method, step rule, floating-point precision, completed horizon and failure domain.

The experiment does not establish global bounds for arbitrary neural integrators or prove long-time stability from a short trace. It provides a nonlinear, independently checked example in which such claims can be formulated and falsified.

## Research extension: an exact invariant does not give uniform phase accuracy

The native quartic experiment correctly labels its modified Hamiltonian as truncated. A distinct calculation becomes available in the harmonic limit a=0: Verlet then has an exact quadratic invariant and an exactly computable phase. Comparing them shows why an excellent energy trace does not certify the clock over increasing time horizons.

We use the native model's harmonic limit, but evaluate the very long horizons below by closed formulas. The finite-step laboratory is not being claimed to execute those millions of steps. The exact-evolution approach is also discussed by Chin and Scuro in [sections III–V](https://arxiv.org/html/math-ph/0408004v2); all mass factors, units and examples below follow the stated kick–drift–kick convention.

### Form the actual one-step matrix

Use the following positive parameters and define two dimensionless quantities:

$$
a=0,\quad k=m\omega^2>0,\quad H=\frac{p^2}{2m}+\frac{m\omega^2q^2}{2},\qquad
x=h\omega,\quad \rho=1-\frac{x^2}{4},\quad m,\omega,h>0.
$$

Inserting the quadratic force into the native Verlet substeps gives

$$
\begin{pmatrix}q'\\p'\end{pmatrix}=M_h\begin{pmatrix}q\\p\end{pmatrix},\qquad
M_h=\begin{pmatrix}1-x^2/2&h/m\\-hm\omega^2\rho&1-x^2/2\end{pmatrix},\quad \det M_h=1.
$$

The off-diagonal entries have units s/kg and kg/s. Determinant one proves symplecticity in this single canonical pair, including step sizes that later turn out unstable. It does not prove boundedness of its iterates.

### Prove the exact invariant and identify its domain

Direct matrix multiplication gives

$$
K_h=\frac{p^2}{2m}+\frac{m\omega^2\rho q^2}{2},\qquad
M_h^T\begin{pmatrix}m\omega^2\rho&0\\0&1/m\end{pmatrix}M_h
=\begin{pmatrix}m\omega^2\rho&0\\0&1/m\end{pmatrix}.
$$

Thus K_h is preserved exactly in exact arithmetic. It is positive definite precisely when 0<h omega<2. In that domain its ellipse bounds both q and p for a fixed initial state. The invariant differs from the physical energy and is specific to this quadratic potential. Applying it unchanged to the native nonlinear a>0 case would be wrong.

An invariant and a Hamiltonian generating a map at a specified time are also different objects. On the stable principal branch,

$$
0<x<2:\quad\vartheta=2\arcsin(x/2),\quad \sin\vartheta=x\sqrt\rho,\qquad
H_{\rm mod}=\frac{\vartheta}{\sin\vartheta}K_h,\quad \omega_h=\frac\vartheta h.
$$

The time-h flow of H_mod is M_h. To check the scale, the natural frequency of K_h is omega sqrt(rho); multiplying by theta/sin(theta) gives theta/h. Removing that factor preserves the same ellipses but changes their traversal time. As h tends to zero the factor tends to one; a zero step itself is the identity.

### Separate an energy bound from a phase bound

For an initial displacement Q and zero initial momentum, the closed iterates are

$$
q_0=Q,\ p_0=0:\quad
q_n=Q\cos(n\vartheta),\quad p_n=-m\omega\sqrt\rho\,Q\sin(n\vartheta),\qquad
E_n=E_0\left[1-\frac{x^2}{4}\sin^2(n\vartheta)\right],\quad
|E_n-E_0|\le E_0\frac{x^2}{4},\quad E_0=\frac{m\omega^2Q^2}{2}.
$$

The energy estimate holds for every integer n in the stable domain. In the unit reference m=omega=Q=1 it bounds physical-energy drift by h²/8 J. Nevertheless the numerical rotation angle differs from the physical one:

$$
\omega_h=\omega+\frac{h^2\omega^3}{24}+O(h^4),\qquad
\Delta\theta_n=n(\vartheta-h\omega).
$$

At a fixed physical duration T=nh, the phase error is second order under the usual small-step limit. That is compatible with the original refinement experiment. It does not say the same small bound holds for durations growing without limit as h changes.

### Test a growing horizon explicitly

Choose the following horizon, using an integer N_h where the displayed ratio is integral, or its nearest integer for the asymptotic limit:

$$
T_h=\frac{24}{\omega(h\omega)^2},\qquad N_h=\frac{24}{(h\omega)^3},\qquad
N_h(\vartheta-h\omega)\longrightarrow1\quad(h\to0).
$$

The leading phase term tends to one radian while the energy bound tends to zero. This is deliberately a growing-horizon comparison, not a fixed-duration convergence-order experiment. It tests whether an inference from small energy drift to uniformly accurate phase would be valid.

The table uses m=omega=Q=1, p0=0 in SI reference units. Each row has its own duration N_h h. The code computes the closed formulas, rather than simulating the listed step counts.

| h (s) | N_h | unwrapped phase error (rad) | energy-error bound (J) |
|---|---|---|---|
| 0.2 | 3000 | 1.004527 | 0.005 |
| 0.1 | 24000 | 1.001127 | 0.00125 |
| 0.05 | 192000 | 1.000281 | 0.0003125 |
| 0.025 | 1536000 | 1.000070 | 0.000078125 |

```python
from math import asin

# SI reference m=omega=Q=1, p0=0. Closed formulas, not millions of updates.
for h in [.2, .1, .05, .025]:
    theta = 2*asin(h/2)
    n = round(24/h**3)
    phase_error = n*(theta-h)
    energy_bound = h*h/8
    print(h, n, round(phase_error, 6), round(energy_bound, 9))
    assert 1 < phase_error < 1.01
```

### Examine the stability boundary rather than extending an ellipse

At the boundary the matrix and invariant become

$$
h\omega=2:\quad M_h=\begin{pmatrix}-1&h/m\\0&-1\end{pmatrix},\quad K_h=\frac{p^2}{2m},\qquad
p_n=(-1)^np_0,\quad q_n=(-1)^n(q_0-nhp_0/m).
$$

For nonzero p0, position grows linearly in step count while the invariant stays constant. It no longer bounds position because its q term has vanished. The special orbit p0=0 is bounded, but does not certify stability of a neighborhood. Beyond this boundary, the matrix has real reciprocal eigenvalues and a growing direction. Symplecticity still holds; the numerical update can be unstable even when the continuous oscillator is bounded.

![On horizons proportional to the inverse square of the step, phase error stays near one radian while the exact physical-energy drift bound decreases.](/assets/learning/analytical-structure-long-phase.en.svg)

### Research questions with explanations

**Does an exactly preserved positive quadratic quantity certify accurate timing?** **Answer:** It bounds states on its ellipses, but does not set the physical angular speed. The modified generator includes a scale factor that changes this clock, producing the accumulated phase error.

**Why not use this exact invariant for the quartic oscillator?** **Answer:** Its derivation uses the constant linear force derivative of a quadratic potential. The native a>0 model instead has a nonlinear map and a truncated modified-energy diagnostic with separately checked defects and assumptions.

**What should a long-time learned-integrator comparison report?** **Answer:** State the horizon and step rule, distinguish fixed-duration refinement from a growing-horizon test, and report unwrapped phase, state error, invariant behavior and stability domain. This solvable limit does not establish a global nonlinear shadow-energy theorem.

## Work through six checks with explanations

### Exercise 1

Recover the first Verlet position for the default example.

**Solution:** $U'(1)=2$ N, so the half-kick is $-1/8$ kg m/s. Multiplying by $h/m=1/8$ moves position by $-1/64$ m. Thus $Q=63/64$ m. The two endpoint momentum formulas from $L_d$ give the same result.

### Exercise 2

Why is the corrected method not certified symplectic by its restored energy?

**Solution:** Its determinant includes derivatives of both the provisional position and the input energy. In the example it is approximately $1.007843$, not one. A trajectory can stay on an energy level while the map distorts neighboring phase states.

### Exercise 3

What does a negative $W$ mean physically and numerically?

**Solution:** The chosen provisional position has potential energy above the available total energy. No real momentum can repair that fixed position. This is failure of the stated update, not failure of the continuous initial-value problem. Stop and report it rather than altering the model silently.

### Exercise 4

Why must both $h$ and $N$ change in a refinement experiment?

**Solution:** Keeping $T=Nh$ fixed isolates discretization error. Halving $h$ without doubling $N$ stops at an earlier physical time, so a changed terminal state no longer measures only numerical accuracy.

### Exercise 5

Does a small truncated-shadow drift prove exact energy conservation?

**Solution:** No. The displayed function contains only the leading $h^2$ correction. Its symbolic local defect begins later than that of $H$, but it is not zero to all orders. The actual recorded drift and the assumptions of the asymptotic argument still matter.

### Exercise 6

A reverse correction fails after a completed forward run. What error should be reported?

**Solution:** No completed reverse position or momentum exists, so both reverse errors are unavailable with the failure reason. Zero would incorrectly certify a roundtrip. A different completed reverse example can instead report its nonzero component errors.

## Sources and independent verification

See [Hairer, Symplectic Integrators, sections 1 and 7](https://unige.ch/~hairer/poly_geoint/week2.pdf) for splitting and discrete variational principles, and [Hairer, Lubich and Wanner, Geometric Numerical Integration Illustrated by the Störmer–Verlet Method](https://courses.physics.ucsd.edu/2019/Winter/physics141/Assignments/stoermer-verlet.pdf) for modified equations and their assumptions. [SciPy's solve_ivp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) specifies the independent DOP853 solver. The particular energy correction, its derivative, guard domains and numerical comparisons are derived and checked here.
