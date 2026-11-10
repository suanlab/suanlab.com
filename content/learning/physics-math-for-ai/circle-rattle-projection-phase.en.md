## A perfect circle is not a complete numerical validation

Suppose every displayed position lies on a circle and every momentum is tangent. It is tempting to declare the simulation correct. But a method can satisfy those geometric constraints while dissipating angular momentum, contracting phase area or advancing around the circle at the wrong rate.

This lesson compares an exactly solvable physical system with two actual numerical maps. RATTLE enforces position and tangent-momentum constraints using constrained kicks and a drift. A simpler alternative normalizes a free drift and projects momentum onto the new tangent.

The example is deliberately free motion on a fixed circle. In this special case RATTLE preserves energy and angular momentum in exact arithmetic, yet it still accumulates a phase error. This exact energy property must not be generalized to arbitrary constrained Hamiltonian systems.

## Specify the constrained phase space

Let $q,p\in\mathbb R^2$, radius $R>0$ and mass $m>0$. The constraints are

$$
\phi_1=\frac{q\cdot q-R^2}{2}=0,\qquad
\phi_2=q\cdot p=0.
$$

The first has units m² and the second Js. Physical states have both the correct radius and tangent momentum. The energy is $E=p\cdot p/(2m)$ and angular momentum is $L=q_xp_y-q_yp_x$.

Use an initial signed tangent momentum $p_t$ and define $L_0=Rp_t$ and $I=mR^2$. An intrinsic angle $\theta$ and angular momentum $L$ form a canonical pair on a local angle chart. Their symplectic area has units Js. A position-plane area measurement is a different geometric object.

## Write the exact reference motion

The physical reduced equations are

$$
\dot\theta=L/I,\qquad \dot L=0.
$$

For one step of length $h$, define the dimensionless signed angle

$$
\nu=\frac{hL_0}{I}=\frac{hp_t}{mR}.
$$

After $n$ exact steps, unwrapped phase advance is $n\nu$. Position and momentum rotate together by that angle. Their magnitudes, energy and angular momentum stay fixed.

Track the unwrapped phase rather than only an angle modulo $2\pi$. A numerical orbit can appear close after completing the wrong number of turns. A circular picture alone cannot reveal that error.

## Derive the free-circle RATTLE step

Write the first constrained momentum as $p_{1/2}=p-\gamma q$ and drift to

$$
Q=q+\frac hm p_{1/2}.
$$

Using the initial constraints gives

$$
\left(1-\frac{h\gamma}{m}\right)^2+\nu^2=1.
$$

Choose the root near the identity:

$$
a=\sqrt{1-\nu^2},\qquad
Q=a q+\frac hm p.
$$

A final radial correction enforces $Q\cdot P=0$. It yields

$$
P=a p-\frac{\nu p_t}{R}q.
$$

The coefficient $\gamma$ has units kg/s; it is not an external force. For nonzero $h$, a cancellation-resistant expression is $\gamma=hp_t^2/[mR^2(1+a)]$. At zero step the algorithm explicitly returns the identity and zero correction rather than evaluating $0/0$.

## Distinguish a regular root, a boundary and no root

A real near-identity position update exists when $|\nu|\le1$. On the open region $|\nu|<1$, the square root is smooth. At $|\nu|=1$, a finite quarter-turn update exists but its derivative with respect to initial angular momentum is unbounded.

When $|\nu|>1$, there is no real RATTLE position root for the requested step. The continuous circular motion still exists; the chosen numerical drift cannot reach a consistent position with that root. The model stops before creating a nonexistent state.

The classification uses the exact integer input ratio for $\nu$ and its discriminant. It does not turn a negative discriminant into zero with a tolerance. Boundary states remain distinct from failed steps, and a finite boundary update is not labeled a regular differentiable map.

## Identify the reduced symplectic map

For the near-identity branch, RATTLE advances by $\delta=\arcsin\nu$ and preserves $L$. On its regular domain,

$$
\begin{pmatrix}\theta'\\L'\end{pmatrix}
=\begin{pmatrix}\theta+\arcsin(hL/I)\\L\end{pmatrix},\qquad
D\Phi_h=\begin{pmatrix}
1&h/(I\sqrt{1-\nu^2})\\0&1
\end{pmatrix}.
$$

The determinant is one. This is the intrinsic canonical derivative, not the determinant of an arbitrary ambient four-dimensional extension of the constrained update.

At the root boundary the derivative is unavailable even though the selected state can be computed. The zero-step map remains the identity. The model also retains the initial, zero-update identity derivative when the first requested nonzero step would be outside the root domain.

## Show why exact invariants do not remove phase error

The numerical phase advance after $N$ RATTLE steps is $N\arcsin\nu$, whereas the physical advance is $N\nu$. Their difference is

$$
\Delta\theta_N=N(\arcsin\nu-\nu).
$$

For small $\nu$, $\arcsin\nu=\nu+\nu^3/6+O(\nu^5)$. Holding $T=Nh$ fixed therefore gives a second-order global phase error.

Take $m=R=p_t=1$, $h=1/2$ s and $N=8$. Exact advance is $4$ rad, while RATTLE advances $8\pi/6$. The error is about $0.1887902$ rad. Both methods can have the same energy $1/2$ J and angular momentum $1$ Js throughout this comparison. An invariant trace alone misses the timing error.

## Derive the simpler normalized-drift projection

The alternative first ignores the constraint reaction during the drift:

$$
\widetilde Q=q+\frac hm p,\qquad
Q=R\frac{\widetilde Q}{\|\widetilde Q\|}.
$$

It then projects the old momentum onto the new tangent:

$$
P=p-\frac{Q\cdot p}{R^2}Q.
$$

Both position and tangent constraints hold in exact arithmetic. However, the reduced update is

$$
\theta'=\theta+\arctan\nu,\qquad
L'=\frac{L}{\sqrt{1+\nu^2}}.
$$

A component of momentum has been removed. Normalizing the position does not supply the physical constraint impulse needed to preserve the intended dynamics. This is a concrete algorithm, not a statement that every possible projection method has the same defect.

## Calculate the dissipation and area contraction

The projected angular momenta satisfy

$$
L_n=\frac{L_0}{\sqrt{1+n\nu_0^2}},\qquad
E_n=\frac{E_0}{1+n\nu_0^2}.
$$

This follows by taking reciprocals of $L_{n+1}^2$ in the recurrence; the zero-momentum case follows directly without dividing by zero. In the preceding eight-step example, energy falls from $1/2$ J to $1/6$ J.

The cumulative reduced Jacobian has determinant

$$
\det D\Phi_h^{\,n}=(1+n\nu_0^2)^{-3/2}.
$$

At eight steps it is $3^{-3/2}$, about $0.19245$. Both geometric constraints can therefore be satisfied while canonical area contracts severely. The upper-right derivative also depends on how changing the initial angular momentum changes every later angle increment; the implementation propagates that dependence.

## Handle stationary states and root boundaries honestly

If $p_t=0$, the selected position and momentum stay fixed under both algorithms. The projected map has determinant one at this selected state, but it is not symplectic over the whole reduced domain for a nonzero step: neighboring nonzero momenta still contract area.

If $\nu=1$, RATTLE makes a finite quarter-turn, retains the free-circle invariants and has no finite generic derivative. If $\nu=1.1$, it cannot take the requested first step. The normalizing method still returns states, but its mere availability does not make those states physically accurate.

These cases require different labels: a stationary selected orbit, a regular map, a finite root boundary and an unreachable update. Null derivatives and missing states must not be converted into zeros or replaced with the first regular sample.

## Refine at fixed physical time

Compare $(N,h)$, $(2N,h/2)$ and $(4N,h/4)$. Each run ends at the same physical time. For RATTLE, the phase-error ratio approaches four in the small-step regime, consistent with second-order accuracy.

For normalized drift with tangent projection, momentum loss behaves differently. At fixed duration, the factor $N\nu_0^2$ is proportional to $h$. Its angular-momentum error is therefore generally first order. The exact constraint equations do not upgrade that state accuracy.

The projected angle is accumulated from all changing momenta, not computed as $N\arctan\nu_0$ with a frozen initial momentum. That shortcut would simulate a different algorithm. Negative initial angular momentum reverses the phase direction while retaining the same squared discriminant and energy-loss factors.

## Compare actual floating-point updates with ideal identities

The product executes Cartesian updates at every displayed step. It does not replace the RATTLE trajectory with a freshly normalized circle at each frame. Position and tangent residuals therefore expose accumulated floating-point roundoff.

The closed formulas for energy, angular momentum and phase provide separate ideal references. The verification also evaluates an independent reduced solution at eighty-digit precision. Numerical agreement is tested with declared tolerances, while the exact branch classification comes from the integer input relation.

Tiny constraint residuals support implementation consistency at the tested precision. They are not exact-arithmetic proofs and do not replace the reduced symplectic derivation. The same caution applies when a determinant near one is measured from a finite-difference or learned tangent model.

## Read controls, cursors and roundtrips

Radius and signed tangent-momentum codes are divided by two in m and kg m/s; step code is divided by twenty seconds. Initial angle uses a quadrant and the rational-circle parameter $u=\mathrm{code}/4$. A separate query selects an actual integer step index.

If the query index exceeds the requested horizon, it is outside the run. If RATTLE stopped before that index, its query is unavailable because the update failed. Neither case silently selects the last point. The analytic physical reference and the alternative projected run remain available for comparison.

The model also runs negative steps from completed terminal states. RATTLE returns to its initial phase state up to roundoff on the selected branch. The dissipative projection removes momentum on the way back as well and generally fails the roundtrip. Position and momentum roundtrip discrepancies retain separate units.

## Apply the experiment to constrained AI models

A learned simulator may enforce normalized coordinates and tangent outputs while changing the dynamics on the constraint manifold. Evaluate the reduced phase map, not just the visual constraint error.

Track invariant behavior, phase accumulation, endpoint state error, local update feasibility and sensitivity to initial data. State whether a conservation result is special to free motion or extends to a larger family with forces. Preserve failures in training and evaluation data rather than silently adapting the requested step.

This experiment does not prove that arbitrary constrained systems conserve energy under RATTLE, nor that a projected neural network is globally symplectic. It gives a fully interpretable counterexample to validation based only on a circle and a tangent arrow.

## Work through six checks with explanations

### Exercise 1

For $m=R=p_t=1$ and $h=1/2$, compute the RATTLE angular increment and compare it with the exact increment.

**Solution:** $\nu=1/2$, so $\delta=\arcsin(1/2)=\pi/6$. The exact increment is $1/2$ rad. The positive difference accumulates with each step despite unchanged energy and angular momentum.

### Exercise 2

Compute the projected energy after eight steps of that example.

**Solution:** $1+N\nu_0^2=1+8/4=3$. Thus $E_8=(1/2)/3=1/6$ J and $L_8=1/\sqrt3$ Js. Both geometric constraint residuals can still vanish in exact arithmetic.

### Exercise 3

Why does $|\nu|=1$ differ from $|\nu|>1$?

**Solution:** At equality the discriminant is zero and the selected state has a finite quarter-turn update. Its derivative contains $1/\sqrt{1-\nu^2}$ and is unavailable. Beyond equality no real constrained drift root exists for this step.

### Exercise 4

Does a zero-momentum projected orbit prove that projection is symplectic?

**Solution:** No. The selected orbit is stationary and its determinant is one at that point. A neighborhood contains nonzero momenta with determinant below one. A global structural property cannot be certified by that special selected trajectory.

### Exercise 5

Why is the projected total phase not $N\arctan\nu_0$?

**Solution:** Each projection reduces angular momentum. The next increment uses $\nu_j=\nu_0/\sqrt{1+j\nu_0^2}$. Sum those changing increments. Keeping the initial value fixed would produce a different map.

### Exercise 6

What does a requested query after a failed RATTLE step contain?

**Solution:** No computed state exists there. Return an unavailable query and the step-failure reason, while retaining earlier actual states and the still-valid continuous reference. Copying the last valid point would fabricate a numerical continuation.

## Sources and the scope of the derivation

[Hairer, Lubich and Wanner, sections 6.2–6.3](https://courses.physics.ucsd.edu/2019/Winter/physics141/Assignments/stoermer-verlet.pdf) describe RATTLE constraints, symmetry and symplecticity on the constraint manifold. [Hairer's symplectic-integrator lectures](https://unige.ch/~hairer/poly_geoint/week2.pdf) provide the splitting and variational background. The explicit free-circle root, reduced maps, projection decay, phase error and feasibility classifications are derived here. The general theory is not used to assert exact physical-energy conservation for every constrained problem.
