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

## Research extension: distinguish virtual work from power on a moving guide

Extend the fixed-guide calculation as a separate problem: its center stays fixed, but an external mechanism prescribes a positive twice-differentiable radius R(t). The only in-plane particle force is an ideal bilateral radial guide reaction. Set the tangential drive to zero. The native fixed-radius controls retain their original meaning; this new physical constraint is reproduced by independent code below.

A changing physical guide differs from relabelling an observer or moving an angular reference marker. Its normal motion can transfer energy to the particle even without friction. We derive the compatibility, Newton and work equations before choosing a contraction protocol.

### Differentiate the time-dependent constraint

With er,e_theta the same orthonormal moving basis as before, write:

$$
g(q,t)=\tfrac12(\lVert q\rVert^2-R(t)^2)=0,\quad
q\cdot v=R\dot R,\qquad
v=\dot R e_r+R\omega e_\theta.
$$

The lab velocity now has a normal component. Importing the fixed-guide condition q dot v=0 would reject compatible moving-guide data. At a frozen time, however, a permitted virtual displacement still satisfies q dot delta q=0. A virtual displacement is a variation within the instantaneous constraint, not the particle's actual displacement over time.

### Find the reaction and the conserved angular momentum

Differentiate the basis and velocity, then separate the radial and tangential equations of m a=N_rad er:

$$
a=(\ddot R-R\omega^2)e_r+(2\dot R\omega+R\dot\omega)e_\theta,\qquad
N_{\mathrm{rad}}=m(\ddot R-R\omega^2),\quad
\ell=mR^2\omega=\mathrm{constant}.
$$

The tangential component gives the derivative of m R² omega equal to zero. About the fixed center, radial motion contributes no cross product, so ell is the full mechanical angular momentum. N_rad is signed outward-positive; a bilateral guide may supply either sign. The fixed-radius formula for angular acceleration cannot simply be reused when R changes.

### Separate zero virtual work from actual power

The gradient of g is q. Represent the normal reaction by a multiplier lambda and use the frozen-time virtual constraint. For the actual motion, differentiate g(q(t),t)=0 instead:

$$
F_{\mathrm{guide}}=\lambda\nabla_qg,\quad \lambda=\frac{N_{\mathrm{rad}}}R,\quad
q\cdot\delta q=0\ \Longrightarrow\ F_{\mathrm{guide}}\cdot\delta q=0,\qquad
P_{\mathrm{guide}}=-\lambda\partial_tg=N_{\mathrm{rad}}\dot R.
$$

Because partial_t g=−R R_dot, the lab power is N_rad R_dot. Equivalently, the reaction is perpendicular to the relative velocity v−u_g for normal guide velocity u_g=R_dot er, while its dot product with lab velocity need not vanish. The external guide mechanism can supply or extract work. This energy transfer is not frictional heat.

### Verify the kinetic-energy derivative directly

Using omega=ell/(m R²), kinetic energy includes both radial and tangential motion:

$$
K=\frac m2\dot R^2+\frac{\ell^2}{2mR^2},\qquad
\dot K=m\dot R\ddot R-\frac{\ell^2\dot R}{mR^3}
=N_{\mathrm{rad}}\dot R.
$$

Substitution of the reaction formula gives the same power without defining work as an unexplained energy residual. The fixed-radius limit R_dot=0 recovers zero guide power. The general derivation applies only while R remains positive and the stated compatible smooth motion exists; no impact or capture law has been introduced.

### Solve a linearly changing radius without dividing by its rate

Let R=R0−b t on a declared interval with positive R, where b has units m/s. Integration of omega gives:

$$
R(t)=R_0-bt>0,\qquad
\theta(t)-\theta_0=\frac{\ell t}{mR_0R(t)},\qquad
\dot\theta(t)=\frac\ell{mR(t)^2}.
$$

Differentiating this rational angle expression gives ell/(m R²), because R−t R_dot=R0. It remains well defined at b=0, avoiding the removable division by b in the equivalent reciprocal-difference formula. Since R_double_dot=0, the reaction and its directly integrated work are:

$$
N_{\mathrm{rad}}=-\frac{\ell^2}{mR(t)^3},\qquad
W_{\mathrm{guide}}(0,t)=\int_0^tN_{\mathrm{rad}}(s)\dot R(s)\,ds
=\frac{\ell^2}{2m}\left(\frac1{R(t)^2}-\frac1{R_0^2}\right).
$$

The radial kinetic term is constant in this linear protocol; the changing tangential term accounts for its energy transfer. Contracting with nonzero ell gives inward force and inward normal velocity, hence positive power. Expansion gives negative guide power even though no friction is present.

### Work a finite contraction and its two other limits

For 0≤t≤1 s choose m=1 kg,ell=2 kg m²/s,R0=2 m,b=1 m/s and theta0=0. The initial radial velocity is already −1 m/s. We are specifying compatible motion on this interval, not silently capturing a particle or starting and stopping the apparatus with instantaneous velocity jumps.

| t (s) | R (m) | K (J) | supplied guide work (J) |
|---|---:|---:|---:|
| 0 | 2 | 1 | 0 |
| 1/2 | 3/2 | 25/18 | 7/18 |
| 1 | 1 | 5/2 | 3/2 |

![A shrinking circular guide gives the particle positive radial-reaction work while preserving its angular momentum; energy and work are shown separately from the spatial path.](/assets/learning/classical-moving-guide-work.en.svg)

The spatial axes use metres with equal unit lengths. Dashed circles are the initial and final guide geometry; the blue line is the particle's actual one-second path. The lower panel shows kinetic energy and accumulated supplied work in joules. It does not place force or velocity arrows on a distance axis with an undeclared scale.

For b=−1 m/s, the guide expands to R=3 m at 1 s and extracts 5/18 J. For b=0, radial velocity vanishes and the original zero-work situation returns. That fixed limit can be checked in the native lab with codes (1,4,0,0,5,10): m=1,R=2,zero tangential drive,theta0=0,omega0=1/2,t=1. Its reaction is −1/2 N,K=1/2 J,ell=2 kg m²/s and guide work zero.

### Reproduce the scalar budgets with exact fractions

The code uses numerical SI values and checks four signed contraction rates at five times, including expansion, a small contraction and the fixed limit. The angular equation, guide power, multiplier power and energy budget are checked independently as algebraic expressions. A zero radius is rejected. It does not implement a new native moving-guide solver or a general contact-release rule.

```python
from fractions import Fraction as F

def state(b, t):
    m, R0, ell = F(1), F(2), F(2)
    R, rd = R0-b*t, -b
    if not (0 <= t <= 1) or R <= 0:
        raise ValueError("Use the stated interval with positive radius")
    omega = ell/(m*R*R)
    alpha = 2*b*ell/(m*R**3)
    reaction = -ell*ell/(m*R**3)
    power = reaction*rd
    theta = ell*t/(m*R0*R)
    K = m*rd*rd/2+ell*ell/(2*m*R*R)
    K0 = m*b*b/2+ell*ell/(2*m*R0*R0)
    W = ell*ell/(2*m)*(1/(R*R)-1/(R0*R0))
    assert m*R*R*omega == ell
    assert 2*rd*omega+R*alpha == 0
    assert power == b*ell*ell/(m*R**3)
    assert K-K0 == W
    # g_t = -R*R_dot and lambda = reaction/R.
    assert -(reaction/R)*(-R*rd) == power
    return R, theta, omega, reaction, K, W

for b in map(F, [-1, 0, "1/10", 1]):
    for t in map(F, [0, "1/4", "1/2", "3/4", 1]):
        result = state(b, t)
    print("contraction rate b, final R, theta, omega, reaction, K, work:", b, *result)
assert state(F(1), F(1))[-2:] == (F(5, 2), F(3, 2))
assert state(F(-1), F(1))[-1] == F(-5, 18)
assert state(F(0), F(1)) == (F(2), F(1, 2), F(1, 2), F(-1, 2), F(1, 2), F(0))
try:
    state(F(2), F(1))
except ValueError:
    pass
else:
    raise AssertionError("A zero radius must be rejected")
```

Changing the native radius input between runs is not this physical protocol: it does not specify its normal velocity or actuator work. Keep the original fixed-guide experiment separate from this standalone extension. The smooth equations above supply the general reasoning; twenty rational preparations check their evaluation, rather than replacing a continuous proof or an independent Cartesian integration.

### Worked questions for learned constraints

1. **Why is zero virtual work insufficient for zero actual power?** The permitted fixed-time variation is tangent, while lab velocity includes R_dot er. The time derivative of the constraint supplies the missing term.
2. **Can energy change while center angular momentum stays constant?** Yes. The reaction is radial and has zero torque, but its power is N_rad R_dot. In the example ell stays two while kinetic energy rises from one to 5/2 J.
3. **What metadata must a constrained surrogate retain?** Record g(q,t), compatible initial velocity, physical guide motion and its work convention. A small radius residual alone cannot validate tangent velocity, reaction force or the energy budget.

For time-dependent constraint coordinates see [Sussman and Wisdom, SICM 1.6.2 and Exercise 1.32](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/9579/sicm_edition_2.zip/chapter001.html). The moving-circle protocol, formulas and worked values are derived here for the declared ideal guide.

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
