## Can a complete initial state still admit several futures?

For a smooth force law, specifying position and velocity selects a Newtonian trajectory. The word “smooth” carries a mathematical responsibility. Continuity of force alone does not guarantee uniqueness. To see why, we construct several explicit twice continuously differentiable trajectories with the same initial position, the same initial velocity and the same force law.

The example concerns an ideal differential equation. It does not establish that a real object spontaneously moves after an arbitrary wait. Instead, it identifies an assumption needed before interpreting an equation as a deterministic predictor. The distinction matters when a learned dynamics model is validated only through equation residuals, energy conservation or one initial observation.

## State the force on its entire domain

Let mass m>0 and coefficient c≥0. On the whole real position axis define

$$
F(x)=12mc\sqrt{\max(x, 0)}, \qquad
\dot x=v, \qquad\dot v=12c\sqrt{\max(x, 0)},
\qquad x(0)=v(0)=0.
$$

Here c has units square root of length divided by time squared. The product c*sqrt(x) has acceleration units. For negative positions the force is zero. The extension through max(x, 0) makes the law explicit in a neighborhood of the initial point, rather than leaving an unstated boundary rule at x=0. The constructed solutions below stay nonnegative.

Mass appears in the force but cancels from acceleration. Changing mass while keeping c fixed therefore changes both the specified force law and the energy scale, while leaving these trajectories unchanged. This is not an experiment holding the force magnitude fixed across masses.

## First inspect the stationary solution

The function x(t)=0 with v(t)=0 satisfies the initial data and the Newton equation for every nonnegative time. Its force, acceleration, kinetic energy and chosen potential energy are all zero. There is no error in a numerical solver that returns this trajectory.

Keep this true stationary solution as a separate comparison curve. A finite waiting time larger than the current query time produces the same currently observed state, but it does not represent a stationary solution for all future times. A finite plotting window cannot prove that motion never begins.

## Derive a moving branch rather than guessing a solver outcome

Seek a positive solution after a release time tau of the form x=A*(t−tau)^p. Balancing powers in x''=12c*sqrt(x) gives p−2=p/2, hence p=4. Balancing positive coefficients gives 12A=12c*sqrt(A), so A=c² when c>0.

For every tau≥0, define s=max(t−tau, 0). Then

$$
x_\tau(t)=c^2s^4, \qquad
v_\tau(t)=4c^2s^3, \qquad
a_\tau(t)=12c^2s^2.
$$

Before release all three are zero. After release, sqrt(x_tau)=c*s² because c and s are nonnegative. Substitution therefore gives m*a_tau=F(x_tau) exactly. Selecting tau chooses one member of a solution family; tau is not an additional parameter in the force law.

## Check regularity at the release event

At t=tau, position, velocity and acceleration from the moving side all tend to zero. They match the stationary side. Thus the glued trajectory is C² and satisfies Newton's equation at the release time as well as away from it. Velocity does not jump, and no impulse has been inserted.

This check is essential. Joining arbitrary curve pieces can otherwise introduce a velocity jump that would require an impulsive force absent from the model. Here even the acceleration is continuous. The ambiguity arises from the law near the zero state, not from an unmodeled collision.

## Compare two finite waits and the stationary branch

Choose m=1 kg and the numerical coefficient c=1 in the stated units. Compare tau_A=0 s and tau_B=1 s at t=2 s. The first solution has x_A=16 m and v_A=32 m/s; the second has x_B=1 m and v_B=4 m/s. Both began with x=v=0 and obey the same force law. The stationary solution remains at zero.

Before one second, the delayed solution B and the stationary branch coincide even though their later behavior differs. At exactly one second, B still has zero position, velocity and acceleration. A readout of zero at that query does not identify the global member of the family.

If the two selected waits are equal, A and B coincide. This does not restore uniqueness of the equation: other waits and the stationary solution are still available when c>0. Distinguish “these two displayed curves differ” from “the initial-value problem has more than one solution.”

## Energy conservation does not resolve the ambiguity

Choose the continuous potential

$$
U(x)=-8mc\, [\max(x, 0)]^{3/2}.
$$

Its derivative gives −U'(x)=F(x), including U'(0)=0. On a moving family member,

$$
K=\frac{m v_\tau^2}{2}=8mc^4s^6, \qquad
U(x_\tau)=-8mc^4s^6, \qquad E=K+U=0.
$$

All waits and the stationary solution have the same conserved energy. Growth of kinetic energy is balanced by increasingly negative potential energy; it is not evidence of a missing external impulse. The unbounded ideal potential is an explicit part of this mathematical example.

The calculation uses exact rational controls and exact polynomial evaluation on this family. It does not claim that a square root at an arbitrary input position is rational. The potential and force identities are checked using the family relation sqrt(x)=c*s².

## Identify the condition that fails

For c>0 and small epsilon>0,

$$
\frac{|F(\varepsilon)-F(0)|}{|\varepsilon-0|}
=\frac{12mc}{\sqrt\varepsilon}\longrightarrow\infty.
$$

No finite local Lipschitz constant at zero can bound this quotient. The force is continuous there, but its change relative to position can become arbitrarily large. The first-order state vector field inherits this lack of local Lipschitz regularity in x.

Local Lipschitz continuity is a sufficient uniqueness condition, not a necessary one. Failure to satisfy a sufficient theorem alone would not prove nonuniqueness. What proves nonuniqueness in this example is the explicitly verified set of distinct C² solutions sharing one complete initial state. The slope-probe table illustrates the quotient; a finite table by itself is not the proof that it is unbounded.

## Preserve the zero-coefficient limit

When c=0, the force law becomes F=0 everywhere and is smooth. The specified zero initial position and velocity give the unique stationary free-particle solution. Every finite-wait formula collapses to x=v=0, so changing tau no longer changes a trajectory.

The displays must change their interpretation in this limit. A coefficient-zero case cannot be labeled nonunique merely because two waiting-time controls remain visible. Likewise, a positive coefficient with both waits after the query can have identical current states despite being a nonunique initial-value problem over a longer interval.

## Interpret numerical integrations carefully

Starting an ordinary solver at the exact zero state often leaves it there, which is a valid solution. Starting at a strictly positive state on a moving branch tests the unique continuation away from the singular origin. These are different experiments.

Independent post-release ODE checks therefore begin with a positive position and its consistent velocity at a specified positive elapsed time. They validate the moving trajectory where the vector field is locally regular. They do not infer a unique release time from the original zero state. The exact polynomial derivative and potential-integral checks provide the evidence at the release join itself.

Do not use a tiny numerical perturbation and then silently identify its selected branch with the exact zero-data problem. Record the perturbation, solver tolerances and starting state. A regularized or noisy force law is a different model whose limiting behavior needs a separate analysis.

## What this teaches physics-informed learning

A trajectory optimizer may enforce the Newton residual, the initial state and E=0, yet still have several exact minimizers distinguished by waiting time. A low residual does not select the intended physical explanation when the mathematical model itself lacks uniqueness.

To make a prediction, justify additional modeling information: for example an independently motivated regularized force, a measured later state, or a stochastic law with an explicit selection mechanism. Such additions change the problem. Do not present an arbitrary solution-family selector as a consequence of the original Newton equation. Distinguish missing state information, unstable dependence on a complete state, and genuine nonuniqueness of the specified law.

## Research extension: a regularization can select a future rather than reveal one

### Compare two uniquely solvable approximations of the same law

The original law x''=12c√(max(x,0)), c>0, with x(0)=x'(0)=0 admits rest and every delayed branch c²max(t−τ,0)⁴. Adding a regularization makes a different equation. To see what that change can select, introduce ε>0 with units square root of length:

$$
a_\epsilon^A(x)=12c\left(\sqrt{\max(x,0)+\epsilon^2}-\epsilon\right),\qquad
a_\epsilon^B(x)=12c\sqrt{\max(x,0)+\epsilon^2}.
$$

This ε is a regularization scale, unlike the position-measurement error in the preceding lesson. Both accelerations are globally Lipschitz for fixed ε, with an x-slope bound 6c/ε. They can have a derivative corner at zero because of max(x,0); they are not asserted to be smooth everywhere. Their first-order initial-value problems nevertheless have unique global solutions.

Both approximate the same original acceleration uniformly:

$$
\sup_{x\in\mathbb R}|a_\epsilon^{A,B}(x)-12c\sqrt{\max(x,0)}|\le12c\epsilon.
$$

Multiplying by the fixed mass gives a force error bound 12mcε. The Lipschitz constants are not uniform as ε tends to zero. Approximate laws can therefore be uniformly close while their selected futures differ.

Family A has aᴬε(0)=0. Its unique zero-data solution stays at rest. Family B has aᴮε(0)=12cε>0, so it moves to x>0 immediately. The initial state is identical, but for every positive ε the equations themselves differ. Choosing either modification requires a modeling justification; it is not additional information deduced from the original equation.

### Prove what the moving approximation approaches

For B, both position and velocity are positive for t>0. Multiply acceleration by velocity and integrate from the zero initial state:

$$
(x')^2=16c\left[(x+\epsilon^2)^{3/2}-\epsilon^3\right],\qquad
t_\epsilon(x)=\int_0^x\frac{dz}{4\sqrt c\sqrt{(z+\epsilon^2)^{3/2}-\epsilon^3}}.
$$

For the original immediate branch the inverse time is t₀(x)=x^(1/4)/√c. Make the substitution z=r⁴ and factor a³−ε³=(a−ε)(a²+aε+ε²), using a−ε=r⁴/(a+ε). The regularized clock becomes

$$
t_\epsilon(x)=\frac1{\sqrt c}\int_0^{x^{1/4}}g_\epsilon(r)\,dr,\quad
g_\epsilon(r)=r\sqrt{\frac{a+\epsilon}{a^2+a\epsilon+\epsilon^2}},\quad a=\sqrt{r^4+\epsilon^2}.
$$

The factored expression also avoids subtracting nearly equal numbers in a numerical clock calculation. For r>0 set q=ε/r² and b=√(1+q²). Then g²=1/[b+q²/(b+q)]. Since b−1≤q²/2 and b+q≥1, the bracket differs from one by at most 3q²/2. The inequality 1−1/√D≤(D−1)/2 for D≥1 gives

$$
0\le1-g_\epsilon(r)\le\min\left(1,\frac{3\epsilon^2}{4r^4}\right),\qquad
0\le t_0(x)-t_\epsilon(x)\le\frac54\sqrt{\frac\epsilon c}=:\alpha_\epsilon.
$$

For the clock bound, split the r integral at √ε. The first segment contributes at most √ε. Integrating the tail bound to infinity contributes at most √ε/4. This proves the constant 5/4 for every x≥0; it is an analytic bound, not a quadrature residual promoted to a certificate.

Invert the increasing clock by inserting tε(xᴮε(t))=t:

$$
c^2t^4\le x_\epsilon^B(t)\le c^2(t+\alpha_\epsilon)^4,\qquad
\sup_{0\le t\le T}|x_\epsilon^B(t)-c^2t^4|\le
c^2\bigl((T+\alpha_\epsilon)^4-T^4\bigr)\longrightarrow0.
$$

Family B therefore converges uniformly on each fixed finite time interval to the original **immediate** branch. Family A remains on the original stationary branch. Uniform approximation of the force alone has not specified which original future should be selected.

### Check the complete state, not only position

The square-root inequality |√x−√y|≤√|x−y| transfers the position enclosure to an acceleration bound. Integrating the acceleration difference from equal zero initial velocities gives

$$
D_\epsilon(T)=c^2\bigl((T+\alpha_\epsilon)^4-T^4\bigr),\quad
A_\epsilon(T)=12c\bigl(\sqrt{D_\epsilon(T)}+\epsilon\bigr),\qquad
\sup_{0\le t\le T}|(x_\epsilon^B)''(t)-12c^2t^2|\le A_\epsilon(T),\quad
\sup_{0\le t\le T}|(x_\epsilon^B)'(t)-4c^2t^3|\le T A_\epsilon(T).
$$

Both Dε and Aε tend to zero. Position, velocity and acceleration all converge uniformly on each finite interval, so the distinction between the selected limits concerns genuine Newtonian trajectories. At c=0 both original and regularized laws vanish and select only rest; the formulas dividing by c are not used there.

For c=1 in the declared SI units, numerical clock inversion gives the following positions at t=1 s. Upper-bound entries are rounded upward to six decimals; the formula above is the bound itself.

| ε (√m) | A position (m) | B position, approximate (m) | Analytic B upper bound (m) |
|---|---:|---:|---:|
| 0.1 | 0 | 2.325399 | 3.790106 |
| 0.01 | 0 | 1.333797 | 1.601807 |
| 0.001 | 0 | 1.097866 | 1.167739 |
| 0.0001 | 0 | 1.030210 | 1.050946 |

![Two regularizations converge to different original solutions](/assets/learning/classical-initial-selection.en.svg)

The gray curve is A's exact stationary solution for every ε. The blue line is the original immediate solution t⁴. The colored regions are **analytic enclosures** for B at ε=0.01 and 0.0001, not computed B trajectories. At t=0 the upper enclosure can be positive even though the actual initial position is exactly zero.

### Execute an independent inverse-clock calculation

The standard-library cell checks the force error, 48 clock comparisons and the analytic clock bound. It then bisects the clock at c=1,t=1 to reproduce the table, comparing 4,096- and 8,192-panel Simpson evaluations. That convergence comparison is a numerical check, not a rigorous quadrature error enclosure.

```python
from math import sqrt
from itertools import product


def regularized_acceleration(x, epsilon, coefficient, family):
    root = sqrt(max(x, 0) + epsilon * epsilon)
    return 12 * coefficient * (root - epsilon if family == 'A' else root)


def release_clock(x, epsilon, coefficient=1.0, steps=4096):
    """Numerical time integral, after x=r^4 removes the endpoint singularity.

    The algebraic factorization avoids subtracting nearly equal cubic roots.
    Simpson comparison is a numerical check, not a certified quadrature bound.
    """
    if x == 0:
        return 0.0
    if epsilon == 0:
        return x ** .25 / sqrt(coefficient)
    upper = x ** .25

    def integrand(r):
        a = sqrt(r ** 4 + epsilon * epsilon)
        return r * sqrt((a + epsilon) / (a * a + a * epsilon + epsilon * epsilon))

    step = upper / steps
    return step * (integrand(0) + integrand(upper) + sum(
        (4 if j % 2 else 2) * integrand(j * step) for j in range(1, steps))) / (3 * sqrt(coefficient))


def regularization_checks():
    count = 0
    for coefficient, epsilon in product([.5, 1, 2], [.1, .01, .001, .0001]):
        assert regularized_acceleration(0, epsilon, coefficient, 'A') == 0
        for x in [-1, 0, 1e-8, .001, 1, 16]:
            original = 12 * coefficient * sqrt(max(x, 0))
            for family in ['A', 'B']:
                assert abs(regularized_acceleration(x, epsilon, coefficient, family) - original) <= 12 * coefficient * epsilon + 2e-13
        for x in [.01, .1, 1, 16]:
            t0 = release_clock(x, 0, coefficient)
            t1 = release_clock(x, epsilon, coefficient)
            t2 = release_clock(x, epsilon, coefficient, 8192)
            bound = 1.25 * sqrt(epsilon / coefficient)
            assert abs(t1 - t2) < 2e-7
            assert -2e-12 <= t0 - t2 <= bound + 2e-12
            count += 1
    print(f'{count} regularized-clock comparisons passed; force errors and rest family checked')


regularization_checks()
for epsilon in [.1, .01, .001, .0001]:
    alpha = 1.25 * sqrt(epsilon)
    lo, hi = 1.0, (1 + alpha) ** 4
    for _ in range(32):
        mid = (lo + hi) / 2
        if release_clock(mid, epsilon) < 1:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    assert 1 <= x <= (1 + alpha) ** 4
    assert abs(release_clock(x, epsilon, steps=8192) - 1) < 3e-7
    print(epsilon, x, (1 + alpha) ** 4)

```

A separate independent DOP853 calculation in the repository checks A/B trajectories, B's energy first integral and the position/velocity/acceleration bounds. It is a numerical cross-check of the declared regularized equations; it does not identify a physically preferred regularization.

The native experiment continues to display the **original** law. Input `[1,2,0,10,10]` compares immediate and one-second-delayed branches at t=1 s, giving positions 1 and 0 m. Changing the query time to 20 gives 16 and 1 m at t=2 s; true rest stays at zero. These are original solutions. The A/B regularizations above are separate Python models, not meanings assigned to native wait controls.

### Research questions with worked answers

1. **Does uniqueness after regularization prove uniqueness of the original law?** No. Each ε>0 defines a different equation. The explicit A and B limits select different original solutions with the same initial state.
2. **Can a regularization be justified only by a small uniform force error?** That criterion is insufficient for selecting a future here. Both families satisfy it. Additional physical or statistical assumptions must distinguish the modification.
3. **Does the proof cover all delayed branches or all perturbations?** No. It proves rest for this A family and immediate release for this B family. It does not claim that every wait is selected by these two formulas or that arbitrary noisy models share their limit.

For the role of Lipschitz assumptions in uniqueness, see [Wisconsin's notes](https://people.math.wisc.edu/~angenent/519.2016s/notes/uniqueness.html) and [Sutherland's existence and uniqueness discussion](https://www.math.stonybrook.edu/~scott/Book331/Existence_Solutions.html). The two regularizations, clock enclosure and complete-state convergence bounds above are explicit derived examples.

## Exercises with explained solutions

1. **Check a delayed trajectory.** Use c=1 and tau=1. Find x, v and a at t=2. **Solution:** The elapsed time is s=1, so x=1, v=4 and a=12 in their respective units. The force is 12m; hence m*a=F exactly. At t=1 all three kinematic quantities are zero, matching the waiting branch.

2. **Compare equal initial states.** For c=m=1, compare waits zero and one second at t=2. **Solution:** The states are (16, 32) and (1, 4). Both have initial state (0, 0), unlike the smooth-spring example with different initial velocities. Their different futures directly demonstrate nonuniqueness for this force law.

3. **Check the energy budget.** Compute the energies of those two states. **Solution:** K_A=512 and U_A=−512, while K_B=8 and U_B=−8 joules. Both totals vanish. The stationary solution also has zero energy, so this invariant cannot select a wait.

4. **A zero observation before release.** Set tau_A=3, tau_B=5 and query t=2. Can the two solutions be distinguished now? **Solution:** Both current states are zero, and both match the stationary branch. They differ at later times when c>0. Equality at one query is not equality of entire solution functions.

5. **A sufficient condition and a special limit.** Evaluate the force quotient at epsilon=1/j², and then set c=0. **Solution:** For positive c the quotient is 12mc*j and has no finite upper bound as j grows. At c=0 every quotient is zero and the law is smooth. All finite-wait curves coincide with the unique stationary solution; failure of uniqueness must not be carried into this limit.

6. **Assess a learned trajectory.** A model returns the stationary branch and another returns a delayed branch, both with zero Newton and energy residuals. Which is rejected by the stated problem? **Solution:** Neither. Both satisfy the complete zero initial data and the force law. To prefer one, provide justified additional information and state the revised problem. A positive-state ODE check validates continuation of a moving branch but does not choose its original wait.

## References and further depth

The [Wisconsin uniqueness notes](https://people.math.wisc.edu/~angenent/519.2016s/notes/uniqueness.html) state a Lipschitz uniqueness theorem. [Sutherland's ODE text](https://www.math.stonybrook.edu/~scott/Book331/Existence_Solutions.html) discusses regularity assumptions for state evolution. The force law, polynomial construction and energy calculations here are explicit derived examples. Existence theorems under continuity, differential inclusions, physically justified regularization and selection of stochastic limits remain further study.
