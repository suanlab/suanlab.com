## Follow a symmetry that changes the action value

A Galilean boost adds a constant velocity to a motion. Even a free particle's kinetic Lagrangian changes under that transformation. The change is an endpoint derivative, so the equations of motion still transform correctly. This is a direct reason to retain boundary terms in Noether's theorem.

This unit compares boosts with time translation in a spatially uniform, time-dependent force. Boosts remain quasi-symmetries of this model, whereas pure time translation generally fails when the force ramps with time. We also deliberately introduce nonphysical trial paths. An exact action identity can hold on such paths without making them solutions.

All paths, actions and charges are rational polynomials. Physical time runs from zero to two seconds. The code integrates these polynomials exactly; it does not numerically solve an ODE or infer conservation from finitely many samples.

## Specify the drive and a representative Lagrangian

Use one position $q$ in m, mass $m>0$ in kg and a force

$$
F(t)=F_0+F_1t,\qquad
L_0=\frac m2\dot q^2+F(t)q.
$$

Here $F_0$ has units N and $F_1$ has units N/s. The potential is $-F(t)q$. Its explicit time dependence matters: mechanical energy is $E=m\dot q^2/2-F(t)q$, whose derivative need not vanish.

Add $dG/dt$ with $G=\gamma t^2q/2$ and $\gamma$ in N/s:

$$
L_\gamma=L_0+\gamma tq+\frac{\gamma t^2}{2}\dot q,
\quad p_\gamma=m\dot q+\frac{\gamma t^2}{2},
\quad H_\gamma=E-\gamma tq.
$$

Force, force slope and gauge coefficient controls use input/2. Initial position and velocity also use input/2 in their respective SI units. The boost parameter is input/2 and dimensionless. The time code is input/20 s. Changing the gauge coefficient changes the representative, not the physical drive.

## Separate the point variation from the fixed-time variation

Let a dimensionless parameter act through $\delta t=\epsilon\tau$ and $\delta q=\epsilon\eta$. The change of a path evaluated at the same clock time is governed by the characteristic

$$
\chi=\eta-\dot q\tau.
$$

A boost uses $\tau=0$ and $\eta=u_*t$, with $u_*=1$ m/s. A pure time translation uses $\tau=\tau_*=1$ s and $\eta=0$, so its fixed-time characteristic is $-\dot q\tau_*$. Although its point-position variation is zero, its fixed-time path variation is generally nonzero.

For these constant clock generators, define the prolonged Lagrangian variation and its boundary defect as

$$
D=\tau\partial_tL+\eta\partial_qL+\dot\eta\partial_{\dot q}L-\dot B.
$$

The corrected charge is $Q=p\eta-H\tau-B$. Both generators are normalized so Q has units Js. The clock charge for positive time shifts has a minus sign relative to energy; reversing the parameter reverses the charge without changing conservation.

## Keep the off-shell identity visible

Define $R=m\ddot q-F(t)$ in N. With the above conventions,

$$
\boxed{\dot Q=D+R\chi.}
$$

The two terms on the right have units J. This is a product-rule identity before the equation of motion is imposed. Noether conservation follows only when the transformation has zero global defect and the selected path has zero residual for the entire interval.

For boosts, $\chi=u_*t$. For clock translation, $\chi=-\dot q\tau_*$. A residual can have a large effect on one charge and no instantaneous effect on another if its characteristic vanishes at that instant. At $t=0$, the boost characteristic is zero, so a zero boost-charge rate there says very little about the force law.

## Derive the boost boundary rather than dropping it

For an infinitesimal boost, $\delta q=u_*t$ and $\delta\dot q=u_*$. The base Lagrangian varies as $m\dot q u_*+F(t)u_*t$. Its boundary primitive is

$$
B_{b,0}=m u_*q+u_*\left(\frac{F_0t^2}{2}+\frac{F_1t^3}{3}\right).
$$

The extra representative contributes $\delta G=\gamma u_*t^3/2$. Therefore

$$
B_b=B_{b,0}+\frac{\gamma u_*t^3}{2},
$$

$$
Q_b=u_*\left[m(t\dot q-q)-\frac{F_0t^2}{2}-\frac{F_1t^3}{3}\right].
$$

The gauge coefficient cancels exactly. The global boost defect is zero for every permitted $F_0,F_1,\gamma$. Spatial uniformity of the force is important; this conclusion is not a statement about arbitrary position-dependent forces.

On a general trial, $\dot Q_b=u_*tR$. The explicit clock dependence of a conserved boost charge is completely compatible with Noether's theorem. A charge need not be independent of time as an expression in coordinates.

## Derive the clock charge and its genuine defect

For pure time translation, the gauge boundary is $B_t=\tau_*\partial_tG=\tau_*\gamma tq$. It follows that

$$
Q_t=-\tau_*H_\gamma-B_t=-\tau_*E,
\qquad D_t=\tau_*F_1q.
$$

The full balance is

$$
\dot Q_t=\tau_*F_1q-\tau_*R\dot q.
$$

When $F_1=0$, time translation is a global quasi-symmetry even if the displayed $L_\gamma$ depends explicitly on time. The improved charge still represents the original mechanical energy. Strict time independence of this representative requires both $F_1=0$ and $\gamma=0$.

When $F_1\ne0$, the drive has genuine clock dependence. The term $\tau_*F_1q$ is not removed by correcting the chosen gauge representative. Its vanishing at the origin does not make it the zero function. Other time-dependent transformations may exist; the classification here concerns this specified pure time-translation generator.

## Construct physical and deliberately nonphysical candidates

The physical particular solution is

$$
q_{\mathrm{phys}}(t)=q_0+v_0t+\frac{F_0t^2}{2m}+\frac{F_1t^3}{6m}.
$$

Direct differentiation gives $m\ddot q=F_0+F_1t$. Trial mode zero uses this solution. Mode one adds $a_*t^2/2$ with $a_*=1$ m/s². Mode two adds $j_*t^3/6$ with $j_*=1$ m/s³. Their residuals are respectively zero, $ma_*$, and $mj_*t$.

Mode two has zero residual at the initial instant and a nonzero residual thereafter. Mode one may even create a constant path when its added acceleration cancels the particular solution's acceleration. The code tests the entire residual polynomial rather than trusting the mode name or the initial state.

For a physical candidate, the boost charge simplifies to $Q_b=-m u_*q_0$ for all time. It is independent of the initial velocity, the two force coefficients and the representative. This statement concerns the normalized boost charge, not every mechanical observable.

## Check a finite boost including the quadratic term

The finite transformation is $q_\epsilon=q+\epsilon u_*t$ and $\dot q_\epsilon=\dot q+\epsilon u_*$. The exact finite boundary is

$$
\mathcal B_\epsilon=\epsilon B_{b,0}
 +\frac{\epsilon^2m u_*^2t}{2}
 +G(q_\epsilon,t)-G(q,t).
$$

Its derivative equals $L_\gamma(q_\epsilon,\dot q_\epsilon,t)-L_\gamma(q,\dot q,t)$. The term quadratic in epsilon comes from the kinetic energy. Omitting it is permissible only in a first-order calculation, not in the finite action comparison.

If $S(t)$ and $S_\epsilon(t)$ start at zero at the initial time, then

$$
S_\epsilon(t)-S(t)=\mathcal B_\epsilon(t)-\mathcal B_\epsilon(0).
$$

Subtract the initial boundary value. It can be nonzero when $q_0\ne0$. This equality is exact on nonphysical candidates too. It verifies a transformation of the Lagrangian, not the selected curve's equation of motion. A boost preserves the candidate's residual; it maps a nonphysical curve to another nonphysical curve with the same residual.

## Work through the free-particle boundary example

Choose $m=2$ kg, $q_0=1$ m, $v_0=2$ m/s, zero drive, zero gauge and $\epsilon=1$. The physical path is $q=1+2t$, and the boosted velocity is 3 m/s.

At two seconds, the naive boost charge $p u_*t$ is 8 Js and its boundary $m u_*q$ is 10 Js. The corrected boost charge is -2 Js. Original kinetic energy is 4 J; boosted kinetic energy is 9 J. The equations retain their form although energy is frame dependent.

The two-second action difference is 10 Js. The finite boundary equals 12 Js at the final endpoint and 2 Js initially, yielding the same difference. Reporting the final boundary alone would be wrong by 2 Js.

The clock charge is -4 Js. This sign reflects the selected direction of time translation. The familiar conserved energy is $-Q_t/\tau_*=4$ J, rather than a claim that negative energy has appeared.

## Compare a ramped drive with a gauge correction

Take $m=1$ kg, $F_0=0$, $F_1=1$ N/s, $q_0=v_0=0$, $\gamma=1$ N/s and $\epsilon=1$. The physical path is $q=t^3/6$. At two seconds, position is $4/3$ m, velocity is 2 m/s and force is 2 N.

The boost canonical charge and its full boundary both equal 8 Js, so $Q_b=0$. The corrected clock charge is $2/3$ Js because mechanical energy is $-2/3$ J. The displayed Hamiltonian is $-10/3$ J. The naive clock charge $10/3$ Js loses the gauge boundary $8/3$ Js to become $2/3$ Js.

Here $D_t=4/3$ J and $R=0$ at two seconds. The clock charge changes for a physical reason: the force schedule breaks pure time translation. In contrast, the boost remains a global quasi-symmetry.

The finite two-second action difference is 9 Js. If gamma is reset to zero, it becomes 5 Js, while the corrected boost charge remains zero. Actions and charges respond differently to endpoint changes. At the initial origin, the clock defect happens to vanish; the global clock flag must still remain false.

## Find the trap in an initially correct trial

Choose unit mass, zero drive and zero initial position and velocity. Select trial mode two, giving $q=t^3/6$ even though the physical free solution would remain at rest.

At the origin, residual and both charge rates are zero. Nevertheless $R=t$ N along the trial. The boost charge becomes $Q_b=t^3/3$ Js and its rate is $t^2$ J. At two seconds these are $8/3$ Js and 4 J. The clock charge is $-t^4/8$ Js, whose rate at two seconds is -4 J.

Both global transformations are valid quasi-symmetries of the model. The boost and clock identities hold exactly, and the finite action-boundary identity also holds exactly. The trial still violates the dynamics. A green algebraic identity is not a substitute for the residual test.

## Interpret the three different invariances

Keep three ideas separate. A symmetry of equations maps solutions into solutions. A quasi-symmetry of the Lagrangian gives an endpoint relation and a Noether charge. Invariance of a particular observable under a finite transformation is another condition. The boost example has the first two properties, while kinetic energy changes.

Changing epsilon changes the comparison path, not the infinitesimal generator or its charge. At epsilon zero, the two action curves coincide and their difference vanishes, but the boost charge can still be $-m u_*q_0\ne0$. Changing gamma changes canonical quantities and action endpoints but not the corrected charges or Newton residual.

For learned dynamical systems, these distinctions guide useful tests. Check covariance of the force law, the complete residual, the appropriate boundary correction and the physical units separately. A loss based only on an action identity can be satisfied by off-shell candidates. A loss based only on a conserved scalar can miss the evolution law. To extend these ideas to discrete integrators, one must derive a discrete variational principle; a sampled continuous formula is not automatically a discrete Noether theorem.

## Exercises with explained solutions

1. **Finite boundary endpoints.** In the free example, why is the action difference 10 Js rather than 12 Js?

   **Solution:** The finite boundary is $\mathcal B=2q+t$ Js with the stated reference units. It equals 12 Js at two seconds and 2 Js initially. The action difference is the final-minus-initial boundary, not just its final value.

2. **A finite second-order term.** What error results from keeping only terms linear in epsilon in the free example at epsilon one?

   **Solution:** The missing endpoint change is $\epsilon^2m u_*^2T/2=2$ Js. The exact action difference is 10 Js; the linear calculation gives 8 Js. Infinitesimal charge calculations and finite action comparisons use different truncation requirements.

3. **Drive versus representative.** In the ramp example, explain why the corrected clock charge is $2/3$ Js while the naive one is $10/3$ Js.

   **Solution:** $H_\gamma=-10/3$ J, so $-\tau_*H_\gamma=10/3$ Js. Subtract $B_t=\tau_*\gamma tq=8/3$ Js. The result is $2/3$ Js. Its nonzero rate is due to $F_1$, not failure to correct the gauge.

4. **An initially invisible residual.** Evaluate the boost balance at two seconds on the free cubic trial.

   **Solution:** Residual is 2 N and boost characteristic is 2 m. Their product is 4 J. The global boost defect is zero and the charge rate is 4 J, matching the off-shell identity. The initial zero did not certify the path.

5. **Zero comparison parameter.** Set epsilon to zero in a physical free case with $m=2$ kg and $q_0=1$ m. Must the boost charge vanish?

   **Solution:** No. The comparison path is unchanged, so its action difference is zero. The generator is still normalized by $u_*=1$ m/s, and its charge remains $-m u_*q_0=-2$ Js. Epsilon is not a coefficient which rescales the definition of the generator in this experiment.

6. **Energy changes under a symmetry.** Find the kinetic-energy change of the free example and explain why it is compatible with boost symmetry.

   **Solution:** The energies are 4 J and 9 J, so the difference is 5 J. A boost changes velocity and therefore frame-dependent kinetic energy. It preserves the form of the free equation and gives a corrected boost charge; neither statement requires kinetic energy to be invariant under the boost.

## Reading and limits

The standard finite-dimensional symmetry argument is developed in David Tong's [The Lagrangian Formalism, section 2.4](https://davidtong.org/teaching/classical-dynamics/dynhtml/S2). Paul K. Townsend's [Noether theorems and higher derivatives, section 2](https://arxiv.org/html/1605.07128v2) makes spacetime characteristics and boundary contributions explicit. The ramped-force, finite-boost and residual examples here are original worked constructions. Local gauge symmetries, field currents, higher derivatives and the classification of all possible dynamical symmetries remain additional topics.
