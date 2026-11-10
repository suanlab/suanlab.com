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

## Exercises with explained solutions

1. **Check a delayed trajectory.** Use c=1 and tau=1. Find x, v and a at t=2. **Solution:** The elapsed time is s=1, so x=1, v=4 and a=12 in their respective units. The force is 12m; hence m*a=F exactly. At t=1 all three kinematic quantities are zero, matching the waiting branch.

2. **Compare equal initial states.** For c=m=1, compare waits zero and one second at t=2. **Solution:** The states are (16, 32) and (1, 4). Both have initial state (0, 0), unlike the smooth-spring example with different initial velocities. Their different futures directly demonstrate nonuniqueness for this force law.

3. **Check the energy budget.** Compute the energies of those two states. **Solution:** K_A=512 and U_A=−512, while K_B=8 and U_B=−8 joules. Both totals vanish. The stationary solution also has zero energy, so this invariant cannot select a wait.

4. **A zero observation before release.** Set tau_A=3, tau_B=5 and query t=2. Can the two solutions be distinguished now? **Solution:** Both current states are zero, and both match the stationary branch. They differ at later times when c>0. Equality at one query is not equality of entire solution functions.

5. **A sufficient condition and a special limit.** Evaluate the force quotient at epsilon=1/j², and then set c=0. **Solution:** For positive c the quotient is 12mc*j and has no finite upper bound as j grows. At c=0 every quotient is zero and the law is smooth. All finite-wait curves coincide with the unique stationary solution; failure of uniqueness must not be carried into this limit.

6. **Assess a learned trajectory.** A model returns the stationary branch and another returns a delayed branch, both with zero Newton and energy residuals. Which is rejected by the stated problem? **Solution:** Neither. Both satisfy the complete zero initial data and the force law. To prefer one, provide justified additional information and state the revised problem. A positive-state ODE check validates continuation of a moving branch but does not choose its original wait.

## References and further depth

The [Wisconsin uniqueness notes](https://people.math.wisc.edu/~angenent/519.2016s/notes/uniqueness.html) state a Lipschitz uniqueness theorem. [Sutherland's ODE text](https://www.math.stonybrook.edu/~scott/Book331/Existence_Solutions.html) discusses regularity assumptions for state evolution. The force law, polynomial construction and energy calculations here are explicit derived examples. Existence theorems under continuity, differential inclusions, physically justified regularization and selection of stochastic limits remain further study.
