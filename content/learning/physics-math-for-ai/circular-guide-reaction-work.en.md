## A guide reaction is not automatically the total force

A particle on a fixed circular guide can accelerate along the circle while a radial reaction keeps it on the allowed path. The reaction and the tangential drive play different roles. Combining them into a single unexplained “centripetal force” hides both the tangential acceleration and the energy input.

We use one positive constant mass m and a fixed positive radius R. An ideal smooth bilateral guide enforces the circle, and an external drive has constant signed tangential magnitude f. There is no friction on this guide. This is a declared constraint model, not a general string-tension or contact-release law.

## Choose compatible initial data

Let er=(cosθ,sinθ) and eθ=(−sinθ,cosθ). The allowed position and velocity are

$$
q=R e_r,\qquad v=R\omega e_\theta,\qquad \omega=\dot\theta.
$$

Thus |q|²=R² and q·v=0. Initial angle and angular velocity are meaningful independent inputs. An arbitrary initial normal velocity would violate the differentiated constraint. The experiment does not silently project such a velocity onto the tangent; doing that would require an additional impulse/capture model.

The guide center and radius stay fixed in the laboratory. Moving constraints can exchange energy differently and are outside this lesson.

## Differentiate the geometry before choosing forces

The moving basis satisfies erdot=ωeθ and eθdot=−ωer. Therefore

$$
a=R\alpha e_\theta-R\omega^2e_r,\qquad \alpha=\dot\omega.
$$

This is a kinematic identity for fixed R. It describes tangential and normal acceleration; it does not add a new physical interaction called “centripetal force.” Newton's law must identify which actual forces supply those components.

The applied drive is f eθ. Its magnitude is constant, but its laboratory direction changes with position. It is not a constant Cartesian force vector.

## Solve the tangential equation and reaction

The ideal guide contributes only a radial reaction. With no other radial force,

$$
mR\alpha=f,\qquad
F_{\mathrm{guide}}=-mR\omega^2e_r.
$$

We report the signed outward-radial scalar Nrad=−mRω², so an inward reaction has a negative value. Its magnitude is mRω². The total force is f eθ+Nrad er, not either component alone.

These two directions are perpendicular. Adding their scalar values does not give the total force magnitude. A model with gravity or another radial applied force would require a different reaction calculation. A unilateral string or a breakable contact would also need an admissibility or release law beyond this ideal guide.

## Integrate the actual angular motion

Constant m,R and f give constant angular acceleration:

$$
\alpha=\frac{f}{mR},\qquad
\omega(t)=\omega_0+\alpha t,
$$

$$
\theta(t)=\theta_0+\omega_0t+\frac12\alpha t^2.
$$

The angle is unwrapped: it retains rotation count and orientation. If initial angular velocity opposes acceleration, the turning time is−ω0/α when positive. At that instant velocity and radial reaction vanish, but tangential drive and acceleration can remain nonzero. The particle subsequently reverses rather than sticking, because no static-friction rule was introduced.

A zero drive gives uniform circular motion or a stationary point, depending on the initial angular velocity.

## Constraint reaction does no work in this model

Since er·eθ=0,

$$
P_{\mathrm{guide}}=F_{\mathrm{guide}}\cdot v=0,\qquad
W_{\mathrm{guide}}=0.
$$

The reaction can be large and nonzero without changing kinetic energy. Its role is to change the direction of velocity while maintaining the constraint. The zero-work conclusion uses a fixed smooth guide and compatible velocity; it is not a statement that all constraint forces in every mechanism do zero work.

Numerical vector dot products may retain small roundoff, while the scalar reaction-work result follows analytically from orthogonality. An independent Cartesian integration checks the actual vector motion and the reaction-power integral.

## Tangential drive supplies signed work

The drive power is fRω. Integrating gives

$$
W_{\mathrm{drive}}=fR[\theta(t)-\theta_0],\qquad
K=\frac12mR^2\omega^2,
$$

$$
K(t)-K(0)=W_{\mathrm{drive}}.
$$

Drive work is negative while drive opposes motion and positive after motion reverses to follow it. It is not generally proportional to the straight-line chord between endpoints. Reducing angle modulo2π would discard entire loops of real work.

For a full positive loop, the constant tangential drive performs2πRf work. With f nonzero there can be no globally single-valued position-only potential on the circle producing this drive: a return to the same position has nonzero loop work. A local angular expression cannot remove that global distinction.

## Check the angular impulse separately

About the guide center, radial reaction torque is zero and drive torque is Rf. Hence

$$
L=mR^2\omega,\qquad
L(t)-L(0)=Rf\,t.
$$

Angular momentum is therefore conserved when drive vanishes, not simply because a circular guide exists. The energy and angular balances test different integrals. A turning instant can have L=0 and still have nonzero Ldot.

All angular increments, reaction scalars, work and angular quantities are exact rational calculations for the control codes. Absolute orientation contains the selected multiple of π, and spatial vectors use numerical sine and cosine. The table distinguishes these calculation types.

## Read actual path and reference geometry correctly

The spatial panel draws the actual path from time zero to the query time in blue. The orange dashed full circle is the guide's geometry, including points the particle may not yet have visited. The green query dot shows its current position. Axes use equal spatial scales.

High-turn paths use enough samples to bound angular changes between neighboring points; a rapid orbit is not drawn as a few misleading long chords. Time panels span ten seconds and separately show radial/tangential force components in N, kinetic-energy/work budget in J and angular momentum in kg m²/s.

## Six exercises with worked solutions

1. **Exercise: driven from rest.** Use m=R=f=1,θ0=ω0=0 and t=1. **Solution:** α=1,ω=1,Δθ=1/2. Reaction magnitude is1N, drive work and K are1/2J, and L=1kg m²/s. Reaction work remains zero.
2. **Exercise: no drive.** Set f=0,ω0=1 with m=R=1. **Solution:** reaction magnitude stays1N and K=1/2J. The nonzero reaction does no work; L=1 is conserved. Here reaction happens to be the entire force because drive is absent.
3. **Exercise: stop without sticking.** Use f=−1,ω0=1,m=R=1 at1s. **Solution:** ω=0,Δθ=1/2,K=0 and Wdrive=−1/2J. Reaction is zero, but drive−1N and angular acceleration−1rad/s² remain. Motion reverses afterward.
4. **Exercise: one complete driven loop.** Start from rest with m=R=f=1. **Solution:** the first positive loop has Δθ=2π, so work and final K are2πJ. The position returns to its start while speed changes. The required time is√(4π)s.
5. **Exercise: retain winding count.** In exercise1 query10s. **Solution:** ω=10,Δθ=50rad,K=Wdrive=50J. The principal angular position is insufficient for computing that work. Reaction magnitude is100N, still with zero reaction work.
6. **Exercise: incompatible velocity.** At R=1,q0=(1,0), can v0=(1,0) be used as smooth initial guide motion? **Solution:** no, q0·v0=1 violates tangency. A compatible example is v0=(0,1). Capturing or projecting the original velocity needs a separate impulse model and energy budget; the circle constraint alone does not specify it.

## Connection to constrained learning

A learned position sequence can lie on the circle while its predicted velocity is not tangent or its forces violate Newton's law. Check radius, tangency, reaction power, force balance, work and angular impulse together. Conversely, forcing positions back onto a manifold after each step can introduce unmodeled impulses. Record the actual constraint and integration assumptions rather than treating projection as automatically physical.

## Background and scope

For constrained coordinates, see [MIT Press, Structure and Interpretation of Classical Mechanics, Chapter1](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/9579/sicm_edition_2.zip/chapter001.html). The constant tangential-drive example is derived here. Frictional guides, changing geometry, contact loss and rigid-body reactions need additional models.
