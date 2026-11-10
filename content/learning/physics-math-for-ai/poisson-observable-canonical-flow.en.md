## Ask how an observable changes

Hamilton equations tell us how position and momentum evolve. A researcher often needs a different quantity: a product of coordinates, a fitted invariant, a quadratic feature, or a measurement whose definition changes with time. Differentiating each quantity by starting again from the trajectory is possible, but a common operator makes the structure easier to inspect.

This lesson builds that operator from the chain rule. Our examples use a single physical free particle or harmonic oscillator. The new objects are observables and their brackets; the physical trajectory remains the previously verified analytic Cartesian flow. No new differential-equation integrator is introduced.

## Fix the canonical convention and units

Use position q in meters and momentum p in kg m/s, with m>0 and k≥0. The Hamiltonian is

$$
H(q,p)=\frac{p^2}{2m}+\frac{kq^2}{2},\qquad
\dot q=\frac{p}{m},\quad \dot p=-kq.
$$

Define the bracket with the order (q,p):

$$
\{F,G\}=F_qG_p-F_pG_q.
$$

The sign convention matters: {q,p}=1, {p,q}=−1, and {q,q}={p,p}=0. Exchanging arguments changes the sign. The bracket is not an ordinary product and is not a covariance or a probability distribution. Its name is unrelated to the Poisson count model from probability.

Units provide an immediate check. Since q times p has units of action Js, the bracket has units [F][G]/Js. Consequently {q,H} has units m/s and {p,H} has units N. Adding those two rates into an unweighted Euclidean norm would mix physical units.

## Recover the chain rule along Hamilton flow

Let A(q,p,t) be a smooth observable with a possible explicit clock dependence. Along a solution,

$$
\frac{dA}{dt}=A_q\dot q+A_p\dot p+A_t
=A_qH_p-A_pH_q+A_t
=\{A,H\}+\partial_tA.
$$

A partial time derivative holds q and p fixed. A total time derivative follows their actual motion. The two are different operations even if an experiment displays a single time cursor.

For a time-independent observable, a zero bracket with H at every relevant phase point gives conservation. A zero value at one phase point does not establish that identity. For an explicitly time-dependent observable, conservation instead requires the sum of the bracket and partial time derivative to vanish along the required trajectories.

## Choose four action-valued observables

To compare observables on one vertical axis, give all four the same action units. Set nu=Omega for a nonzero oscillator frequency, and nu=1 per second in the free limit. The selectable base quantities are

$$
C=qp,\qquad F=\frac{m\nu q^2}{2},\qquad
G=\frac{p^2}{2m\nu},\qquad B=\frac{H}{\nu}.
$$

Nu is a fixed reference frequency for the selected experiment, not an extra function of phase position. Their time rates from the bracket are

$$
\{C,H\}=\frac{p^2}{m}-kq^2,\quad
\{F,H\}=\nu qp,\quad
\{G,H\}=-\frac{kqp}{m\nu},\quad
\{B,H\}=0.
$$

For a free particle, G and B coincide. That degeneracy is expected because the potential energy vanishes; the interface must not pretend that the four choices are always independent functions.

The clock-dependent observable is A=A_base+beta t H, with dimensionless beta. A has units Js, while its bracket rate, explicit rate and total rate all have units J. These rates are not themselves physical energy loss; they are rates of an action-valued observable.

## Work through a rational free example

Choose m=1 kg, k=0, q0=1 m and v0=1 m/s. At t=1 s, q=2 m, p=1 kg m/s and H=0.5 J. With A=C and beta=0, A=2 Js and its bracket rate is p²/m=1 J. Physical force is zero, but the position-momentum product changes because position changes.

Now keep the same trajectory and choose beta=−2:

$$
A=qp-2tH=qp-tp^2/m=q_0p.
$$

This observable is constant for every free trajectory. At the selected time its base contribution is 2 Js and its explicit-time contribution is −1 Js, leaving A=1 Js. Its bracket with H is still 1 J. The explicit partial derivative is −1 J, so the total derivative is zero.

The lesson is precise: an explicitly time-dependent conserved observable need not commute with H by itself. Ignoring its clock term would falsely reject a valid invariant. Conversely, adding a fitted clock term without deriving the physical convention could hide a poor invariant model.

## Find a zero derivative without conservation

Consider m=1 kg, Omega=pi/2 per second, q0=1 m and v0=0. The actual trajectory is q=cos(Omega t) and p=−Omega sin(Omega t). Its energy is pi²/8 J.

For A=qp and beta=0, the observable is zero at t=0 and t=1 s, but equals −pi/4 Js at t=0.5 s. Its rate is zero at that middle instant because kinetic and potential energies are equal there. It is certainly not constant on the full interval.

By comparison, B=H/nu is pi/4 Js throughout the same trajectory. Moving the query cursor should select different values without changing the already computed trajectory. Whole-function conservation and a selected orbit's accidental constancy also need distinct labels: for example, a resting free particle can make F constant even though {F,H} is not identically zero on free-particle phase space.

## Test a canonical coordinate scaling

Let a>0 be dimensionless and define

$$
Q=aq,\qquad P=p/a.
$$

The original bracket gives {Q,P}=1. Written using the new independent coordinates, the same Hamiltonian is

$$
\widetilde H(Q,P)=\frac{a^2P^2}{2m}+\frac{kQ^2}{2a^2}.
$$

Apply the chain rule to any observable: A_Q=A_q/a and A_P=a A_p. Inserting those derivatives in the standard (Q,P) bracket cancels the two scale factors. Thus the new-coordinate computation gives exactly the same physical bracket value, apart from floating-point evaluation differences.

Q still has length units and P momentum units. Their numerical values and phase portrait differ from (q,p); this is a representation change, not a second spatial trajectory. A value preserved by a coordinate transformation need not have the same numerical coordinate components.

## Diagnose the tempting incorrect scaling

Suppose both coordinates are multiplied by a: Q=aq and Pbad=ap. In the original canonical structure,

$$
\{Q,P_{\mathrm{bad}}\}=a^2.
$$

If we incorrectly assume a unit fundamental bracket in these new coordinates, the resulting observable bracket is smaller by a². In the rational free example with a=2, the physical Q rate is 2 m/s, whereas the naive rate from the incorrectly canonical Hamiltonian is 0.5 m/s.

The coordinate map is invertible. Its failure here is the unjustified use of the standard canonical bracket after the transformation. It can describe the same motion if its induced Poisson matrix is used:

$$
J_{\mathrm{bad}}=a^2\begin{pmatrix}0&1\\-1&0\end{pmatrix}.
$$

Multiplying the naive bracket by a² restores the physical rate. At a=1 the distinction disappears. This is an introductory constant linear example; it does not prove a general theorem about higher-dimensional volume preservation or arbitrary nonlinear coordinate maps.

## Read the bracket as an algebra

Bilinearity and the derivative product rule give

$$
\{F,GH\}=\{F,G\}H+G\{F,H\}.
$$

The action-valued quadratic basis provides a checkable example:

$$
\{F,G\}=C,\qquad \{G,C\}=-2G,\qquad \{C,F\}=-2F.
$$

The cyclic nested brackets for this triple are −2C, +2C and zero; their sum vanishes. This illustrates the Jacobi identity. A successful numerical example is not a proof for arbitrary functions. For smooth functions the general identity follows by expanding derivatives, grouping second-derivative terms and using their symmetry against the bracket's antisymmetry.

The identities constrain a learned bracket operator differently from trajectory fit. A model may match a few observed time derivatives while violating antisymmetry, the product rule or the Jacobi identity elsewhere. State the domain and differentiability assumptions before making a global structural claim.

## Interpret the numerical output honestly

The model evaluates 201 times over two seconds. Frequencies use Omega=pi f/4 and k=m Omega², so changing mass at fixed f also changes stiffness. The free limit uses an exact linear flow, and cardinal trigonometric states reuse the previous analytic-flow implementation.

Action histories, rates and phase portraits have separate axes. Identity residuals remain raw floating-point differences. Reported maximum residuals refer to sampled output times; they are neither continuous-time bounds nor integration errors from a newly introduced solver. Conservation flags come from the selected formulas and exact parameter conditions, rather than from accepting a small graph variation as a proof.

## Exercises with explanations

1. Compute {q,H} and {p,H}. **Solution:** Only q_q=1 and p_p=1 survive, yielding p/m and −kq. The signs recover Hamilton equations and check the argument order.
2. Why is {qp,H}=0 at equal kinetic and potential energy insufficient for conservation? **Solution:** The expression is 2K−2U. Equal energies can occur at isolated times while qp varies before and afterward, as the half-second oscillator example demonstrates.
3. Verify the free explicitly time-dependent invariant. **Solution:** For k=0, {qp,H}=2H and partial_t(−2tH)=−2H. Their sum vanishes for all phase points and times, even when the bracket alone is nonzero.
4. Calculate the scaling error at a=2. **Solution:** The bad coordinates have fundamental bracket four. Treating them as standard canonical coordinates divides bracket-generated rates by four. Using their induced Poisson matrix corrects the result.
5. Evaluate the quadratic Jacobi triple at q=2 m, p=1 kg m/s. **Solution:** C=2 Js, so the three terms are −4 Js, +4 Js and zero. The cancellation checks this example, not every possible function.
6. Does beta=1 guarantee that the displayed observable changes on every orbit? **Solution:** No. On a zero-energy resting free trajectory, H=0 and the clock contribution vanishes. The observable function remains explicitly time dependent away from that selected orbit.

## Research connection and further reading

When learning physical invariants, include explicit time arguments and coordinate conventions in the hypothesis class. Compare bracket identities and independent time derivatives in addition to trajectory loss. A conserved observable, a canonical coordinate choice and a small sampled residual are three different claims.

For the standard bracket convention and canonical-transformation framework, see David Tong's [Hamiltonian Formalism, sections 4.3 and 4.4](https://davidtong.org/teaching/classical-dynamics/dynhtml/S4). The action-valued choices, bad-scaling comparison and rational clock-dependent example here are derived directly and checked independently with symbolic differentiation, 60-digit trajectory derivatives and exact fractions.
