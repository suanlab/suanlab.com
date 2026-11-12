## One photograph does not determine a Newtonian future

Imagine balancing a bead at the top of a smooth hill. A photograph gives its position, but two beads at that position can be moving in opposite directions. The force law supplies acceleration; it does not supply the missing velocity. A second-order Newton equation becomes a first-order evolution law only after position and velocity are included in the state.

This lesson uses an ideal destabilizing spring to separate three questions: whether a complete state has one future, how differences between states evolve, and whether a recorded future state contains enough precision to recover the past. These questions are related, but their answers need not be the same. A unique evolution can amplify initial errors, and an invertible evolution can be hard to reverse from rounded observations.

## Specify a physical law and its range of interpretation

Take mass m>0 and destabilizing stiffness k≥0:

$$
m\ddot x=kx, \qquad z=(x, v), \qquad
\dot z=\begin{pmatrix}v\\(k/m)x\end{pmatrix}.
$$

The force points away from the origin. This is the sign of a local unstable equilibrium, not the restoring force of an ordinary spring. Position is in meters, velocity in meters per second, mass in kilograms and k in newtons per meter. The potential is U(x)=−kx²/2, with its arbitrary zero chosen at the origin.

An actual hill or apparatus need not retain a quadratic potential at large displacement. The displayed trajectories solve this ideal mathematical model throughout the chosen interval. Extending their exponential growth indefinitely is not a prediction that an unlimited physical energy reservoir exists.

## A complete state selects a unique trajectory

The vector field is linear and smooth. For fixed m and k, giving x(0) and v(0) selects one global solution. Giving only x(0) leaves a family. Giving acceleration as an additional independent datum would be redundant here: the force law already fixes it from x(0).

Two position-time curves can meet while their velocities differ. They have the same projected position at that time, but occupy different points in phase space. Under this smooth autonomous law, two different complete-state trajectories cannot meet at the same time and then acquire different futures. This statement depends on the law and regularity assumptions; it is not a consequence of drawing a graph that looks orderly.

## Resolve growing and decaying directions

For k>0, set lambda=sqrt(k/m). Its unit is inverse seconds. Solving the characteristic equation gives

$$
x(t)=c_+e^{\lambda t}+c_-e^{-\lambda t}, \qquad
v(t)=\lambda c_+e^{\lambda t}-\lambda c_-e^{-\lambda t},
$$

$$
c_+=\frac{x_0+v_0/\lambda}{2}, \qquad
c_-=\frac{x_0-v_0/\lambda}{2}.
$$

Both modal coefficients have units of length. The growing direction has v0=lambda*x0; the decaying direction has v0=−lambda*x0. A generic initial state contains both. The existence of the decaying direction does not make the equilibrium stable: every neighborhood also contains states with a nonzero growing component.

For m=k=1 in the stated units, initial states (1, −1) and (1, 1) have identical positions but give x=e^(−t) and x=e^t. At two seconds their positions are about 0.1353 m and 7.3891 m. Neither result is ambiguous after the complete state is specified.

## Energy is an invariant, not a full state description

The conserved mechanical energy is

$$
E=\frac{m v^2}{2}-\frac{k x^2}{2}.
$$

It is indefinite: positive kinetic and negative potential terms can cancel. Both example states above have E=0. Thus even the pair “position and energy” fails to distinguish their velocity signs. An energy level is generally a set of phase-space states, not a replacement for an initial state.

Read the energy plots together with the position and velocity plots. A flat zero-energy curve does not imply no motion or no sensitivity. At large growth, evaluating the difference of two large floating-point terms can leave a small numerical residual. The initial energy is calculated as an exact rational from control codes; propagated energies and their absolute and scaled residuals are numerical diagnostics.

## Compare perturbations with consistent units

Adding a squared position error directly to a squared velocity error mixes dimensions. Introduce a stated reference time T=1 second and compare y=(x, Tv), whose two components both have units of length. For this convention,

$$
\delta y(t)=M_T(t)\delta y(0), \quad
M_T(t)=\begin{pmatrix}
\cosh(\lambda t)&\sinh(\lambda t)/(\lambda T)\\
T\lambda\sinh(\lambda t)&\cosh(\lambda t)
\end{pmatrix}.
$$

The difference equation uses the same matrix because the system is linear. Its largest singular value gives the Euclidean norm bound

$$
\|\delta y(t)\|_2\leq\sigma_{\max}(M_T(t))\|\delta y(0)\|_2.
$$

This is an upper bound, not the observed amplification for every perturbation. The direction matters. For lambda*T=1, growing and decaying eigenvectors are orthogonal, and their amplification factors are e^(lambda*t) and e^(−lambda*t). Changing T changes the norm and its finite-time singular values; it does not change the physical trajectory.

## Area preservation can coexist with instability

The analytic determinant of M_T is one. An expanding direction is accompanied by a contracting direction, so an initially small phase-space area is preserved even though lengths can change greatly. The two singular values are reciprocal. This property does not mean every distance or every coordinate uncertainty is preserved.

The model has exponential separation in one direction, yet it is an unbounded linear system with a completely explicit solution. Exponential growth alone is insufficient evidence of chaos. Use the example to understand instability and conditioning; do not infer mixing, bounded recurrent chaotic motion or unpredictable nonlinear interactions that this equation does not contain.

## Preserve the free-particle limit

At k=0, avoid dividing by lambda. The correct solution is

$$
x(t)=x_0+v_0t, \qquad v(t)=v_0, \qquad
M_T(t)=\begin{pmatrix}1&t/T\\0&1\end{pmatrix}.
$$

A velocity uncertainty produces a growing position uncertainty even though lambda=0. The finite-time singular norm of this shear is generally greater than one. Substituting e^(lambda*t)=1 as a universal error bound would therefore be wrong in this state norm. The modal coefficients are undefined at zero lambda and should not be displayed as finite physical quantities there.

## Invertible dynamics do not undo measurement rounding

The analytic inverse flow is M_T(−t). Applied to an exact state, it reconstructs the exact initial state. The lab also rounds both observed components to increments of 0.001 in their respective stated units before reconstructing. This is an explicit synthetic observation operation, not a claim that the underlying motion loses information.

If the observation error is eta, the reconstructed initial error is M_T(−t)*eta. Its norm is bounded by sigma_max(M_T(−t))*||eta||. Because the singular values are reciprocal and their product is one, the largest inverse singular value equals the largest forward singular value here. A square rounding cell gives the separate worst-case bound sigma_max*sqrt(2)*0.0005 m for T=1 s.

For the decaying example at five seconds, the state is approximately (0.00673795, −0.00673795). Rounding gives (0.007, −0.007). Inverse propagation yields approximately (1.03889, −1.03889), rather than (1, −1). Numerical inverse residuals without observation rounding are reported separately. Analytic invertibility, finite precision and observation uncertainty are distinct claims.

## Use this distinction in AI and research

A dynamics model that receives position alone may predict a conditional average over hidden velocities rather than a valid single Newton trajectory. An energy-conserving learned model can still choose the wrong branch. Supplying a complete state and checking state transitions addresses a different failure from checking an energy residual.

When fitting or reversing trajectories, report the state variables, observation precision, time horizon, units and norm. Compare sensitivity directions instead of reporting one universal amplification number. Independent matrix exponentials and ODE integrations validate the formulas; a small residual from the same implementation is not independent evidence of correctness.

## Research extension: observe the missing velocity without overlooking prediction error

### Two photographs can provide a complete state

One position observation leaves velocity unknown. A second observation at a known different time can supply the missing information if the force law is known. Keep the exact ideal law m x''=kx with m>0 and k≥0; measurement bias, uncertain mass or stiffness, and timing error are not included in the calculation below.

For k>0 let λ=√(k/m)>0 in s⁻¹, and let the observation gap be Δ>0 seconds. The two positions obey

$$
\begin{pmatrix}x(0)\\x(\Delta)\end{pmatrix}
=\begin{pmatrix}1&0\\\cosh(\lambda\Delta)&\sinh(\lambda\Delta)/\lambda\end{pmatrix}
\begin{pmatrix}x_0\\v_0\end{pmatrix},\qquad
v_0=\lambda\frac{x(\Delta)-x_0\cosh(\lambda\Delta)}{\sinh(\lambda\Delta)}.
$$

The observation matrix has determinant sinh(λΔ)/λ>0. Thus the exact positions determine x₀ and v₀ uniquely. At Δ=0 the two observations coincide and cannot identify velocity. This is a linear observability statement for this known model, not a claim that two photographs recover an unknown force law.

Do not attach a numerical condition number to this matrix without first specifying scales for position and velocity. Instead, propagate observation errors in their physical units.

### Derive a sharp velocity error bound

Suppose y₀=x₀+d₀ and yΔ=x(Δ)+dΔ, with independent bounds |d₀|,|dΔ|≤ε metres. Subtract the exact inversion formula from the measured one and apply the triangle inequality:

$$
|\widehat v_0-v_0|\le\lambda\epsilon\frac{1+\cosh(\lambda\Delta)}{\sinh(\lambda\Delta)}
=\lambda\epsilon\coth(\lambda\Delta/2).
$$

Opposite error signs attain the bound, so it is sharp for the stated error box. When λΔ is small, the bound behaves like 2ε/Δ: two almost simultaneous photographs require subtracting two noisy position values. Exact invertibility does not make that subtraction reliable. The hyperbolic ratio tends to one at large gap, giving the limiting initial-velocity bound λε within this ideal model.

Longer waiting also extends the time during which a real apparatus must satisfy the assumed law. These bounds do not absorb model error or certify that a quadratic unstable potential remains physically applicable at large displacement.

### Interpolation and extrapolation have different uncertainty

Eliminate velocity and write a position prediction directly from the two observations:

$$
\widehat x(t)=\frac{\sinh(\lambda(\Delta-t))}{\sinh(\lambda\Delta)}y_0+
\frac{\sinh(\lambda t)}{\sinh(\lambda\Delta)}y_\Delta.
$$

For 0≤t≤Δ both coefficients are nonnegative. Their sum is cosh(λ(t−Δ/2))/cosh(λΔ/2)≤1. Therefore the interpolated position error is at most ε; at the midpoint the worst-case bound is ε/cosh(λΔ/2). Instability of the dynamics does not force every constrained interpolation error to grow.

For t≥Δ the first coefficient changes sign. The sharp box-error bound becomes

$$
|\widehat x(t)-x(t)|\le\epsilon
\frac{\sinh(\lambda(t-\Delta))+\sinh(\lambda t)}{\sinh(\lambda\Delta)}
=\epsilon\frac{\sinh(\lambda(t-\Delta/2))}{\sinh(\lambda\Delta/2)}\qquad(t\ge\Delta).
$$

It grows with the extrapolation horizon. Fitting both observations exactly gives zero observation residual even when the future state is uncertain. A learned model can therefore fit the two images perfectly while its extrapolation remains poorly constrained.

At λ=1 s⁻¹ and ε=0.001 m, extrapolating one second beyond the second observation gives:

| Gap Δ (s) | Initial velocity error bound (m/s) | Position error bound at t=Δ+1 s (m) |
|---|---:|---:|
| 0.1 | 0.020016664 | 0.025066688 |
| 1 | 0.002163953 | 0.004086161 |
| 2 | 0.001313035 | 0.003086161 |

![Observation spacing controls velocity recovery and future position error](/assets/learning/classical-initial-observation.en.svg)

Both panels use the parameters above and logarithmic vertical axes. The upper quantity is a velocity error in m/s; the lower is a position error in m. These are worst-case bounds over the declared observation errors, not errors realized by every pair of measurements.

### Keep the zero-stiffness limit and the native controls explicit

At λ=0 the free-particle formulas are

$$
v_0=\frac{x(\Delta)-x_0}{\Delta},\quad
|\widehat v_0-v_0|\le\frac{2\epsilon}{\Delta},\qquad
\widehat x(t)=\left(1-\frac t\Delta\right)y_0+\frac t\Delta y_\Delta,\quad
|\widehat x(t)-x(t)|\le\epsilon\left(\left|1-\frac t\Delta\right|+\left|\frac t\Delta\right|\right)
\quad(\lambda=0).
$$

For interpolation the position bound is ε, and for t≥Δ it is ε(2t/Δ−1). Computing these formulas directly avoids division by zero in the hyperbolic expression.

Use native input `[1,1,10,-10,10,10,10]`: both initial positions are 1 m, but initial velocities are −1 and +1 m/s. At t=1 s their positions are e⁻¹ and e. A second photograph distinguishes the two complete states. Change the time code to 20 to compare at t=2 s. For the free limit, `[2,0,10,5,10,-5,20]` gives positions 2 and 0 m at t=2 s, recovering velocities ±1/2 m/s.

The existing experiment still accepts initial-state controls. It does not acquire new measurement or inverse-estimation inputs here. The noise-box inversion is a standalone Python extension.

### Execute the observation experiment

This standard-library cell checks 144 states, four error corners per state, interpolation and extrapolation, and the opposite signs that attain the velocity bound. Its printed rows reproduce the comparison table. Floating-point tolerances check the formulas numerically; the sharp bounds follow from the derivation above.

```python
from math import cosh, sinh, sqrt
from itertools import product


def position(rate, x0, v0, time):
    return x0 + v0 * time if rate == 0 else (
        x0 * cosh(rate * time) + v0 * sinh(rate * time) / rate)


def infer_velocity(rate, gap, first, second):
    if gap <= 0 or rate < 0:
        raise ValueError('Use nonnegative rate and strictly positive time gap.')
    return ((second - first) / gap if rate == 0 else
            rate * (second - first * cosh(rate * gap)) / sinh(rate * gap))


def velocity_bound(rate, gap, error):
    return 2 * error / gap if rate == 0 else (
        rate * error * (1 + cosh(rate * gap)) / sinh(rate * gap))


def predict_position(rate, gap, first, second, time):
    if rate == 0:
        return (1 - time / gap) * first + time / gap * second
    return (sinh(rate * (gap - time)) * first + sinh(rate * time) * second) / sinh(rate * gap)


def position_bound(rate, gap, time, error):
    if rate == 0:
        return error * (abs(1 - time / gap) + abs(time / gap))
    return error * (abs(sinh(rate * (gap - time))) + abs(sinh(rate * time))) / sinh(rate * gap)


def observation_checks():
    count = 0
    for rate, gap, x0, v0 in product([0, .5, 1, 2], [.1, .5, 1, 2], [-1, 0, 1], [-1, .5, 1]):
        last = position(rate, x0, v0, gap)
        assert abs(infer_velocity(rate, gap, x0, last) - v0) < 2e-12
        error = .001
        for da, db in product([-error, error], repeat=2):
            estimate = infer_velocity(rate, gap, x0 + da, last + db)
            assert abs(estimate - v0) <= velocity_bound(rate, gap, error) + 2e-12
            for time in [gap / 2, gap, gap + 1]:
                predicted = predict_position(rate, gap, x0 + da, last + db, time)
                actual = position(rate, x0, v0, time)
                bound = position_bound(rate, gap, time, error)
                assert abs(predicted - actual) <= bound + 2e-11
                if time <= gap:
                    assert bound <= error + 1e-15
        count += 1
    # Opposite signs attain the velocity bound; each bound is not merely a loose
    # consequence of the same numerical inverse operation.
    for gap in [.1, 1, 2]:
        v = infer_velocity(1, gap, -.001, .001)
        assert abs(v - velocity_bound(1, gap, .001)) < 1e-15
        print('Gap, velocity bound, one-second extrapolation bound:', gap,
              velocity_bound(1, gap, .001), position_bound(1, gap, gap + 1, .001))
    print(f'{count} observation states: exact recovery and four noise corners passed')


observation_checks()
```

### Research questions with worked answers

1. **Does a second photograph always identify velocity?** Under the known law and known Δ>0 it does in exact arithmetic. At the same time Δ=0 it does not. Unknown force parameters, timing or observation bias change the identification problem.
2. **Does a perfect fit imply a reliable future?** No. Two noisy positions can always define an exact fitting trajectory in this two-dimensional linear state space. The extrapolation error still follows the growing box bound.
3. **Why can midpoint interpolation be better than either observation bound?** Both observations constrain the same trajectory. Their positive interpolation weights sum to less than one for λ>0, whereas extrapolation uses a negative weight and amplifies the worst-case error.

The uniqueness and separation framework is discussed in [Wisconsin's differential-equation notes](https://people.math.wisc.edu/~angenent/519.2016s/notes/uniqueness.html). The observation inverse and unit-specific error bounds above are directly derived for this declared linear force law.

## Exercises with explained solutions

1. **Two futures from one photograph.** Let m=k=1 and x0=1. Compare v0=−1 and v0=1 at t=2. **Solution:** The modal coefficients are respectively (c+, c−)=(0, 1) and (1, 0). Positions are e^(−2) and e²; velocities are −e^(−2) and e². The positions initially match, but the complete initial states do not.

2. **An invariant that misses the branch.** Compute the initial energy of those two states. **Solution:** Both give E=(1²−1²)/2=0. Equal energy and position do not identify the velocity sign. The level E=0 contains a growing and a decaying direction.

3. **Direction-dependent perturbations.** With m=k=T=1, compare perturbations (0.1, 0.1) and (0.1, −0.1) after three seconds. **Solution:** Their initial norms are equal. The first norm is multiplied by e³, the second by e^(−3). Both satisfy the operator upper bound e³ times the initial norm; only the growing direction attains it.

4. **A zero-stiffness bound.** Use m=2, k=0, state A=(1, 0), state B=(1, 1), and t=2. **Solution:** A remains at x=1 while B reaches x=3, v=1. The state-difference norm is sqrt(5), from initial norm one. The shear's largest singular value is 1+sqrt(2), which is larger than sqrt(5), demonstrating a strict bound.

5. **Reverse a rounded observation.** Round the stable m=k=1 state at t=5 to (0.007, −0.007). **Solution:** This rounded state lies on the same decaying direction. Multiplying by the inverse flow gives initial amplitude 0.007*e^5≈1.03889. The approximately 0.0550 m two-component error is consistent with the inverse operator bound; it does not refute uniqueness.

6. **Choose an informative validation.** A learned predictor matches E=0 and x0=1 but outputs the growing trajectory for data from the decaying state. **Solution:** Energy validation alone accepts both. Validate the complete initial state, including v0=−1, and then the Newton state transition. Report numerical energy residuals separately from initial-state mismatch and conditioning.

## Further reading and boundaries

The [Wisconsin uniqueness notes](https://people.math.wisc.edu/~angenent/519.2016s/notes/uniqueness.html) explain Lipschitz uniqueness and bounds on separation. [Sutherland's ODE text](https://www.math.stonybrook.edu/~scott/Book331/Existence_Solutions.html) relates uniqueness to complete-state phase portraits. The destabilizing-spring examples and rounding experiment here are explicit worked derivations. Nonlinear chaos, stochastic dynamics, constrained state reconstruction and model error require further lessons.
