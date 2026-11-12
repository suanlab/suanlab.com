## Acceleration can depend on the observer

An unforced particle is stationary in an inertial laboratory. A camera accelerating away from it records a moving particle with a nonzero coordinate acceleration. The camera did not create a new physical interaction with the particle. Its coordinate description requires an additional inertial term.

This lesson extends the earlier Galilean observer change to a nonrotating observer with constant translational acceleration. We derive the coordinate law, reconstruct the same physical event and examine what goes wrong when the inertial term is omitted. Rotating frames, contact forces and apparent-weight measurements require further models and are not silently inferred here.

## Specify the moving origin before changing the law

Let the laboratory be inertial. The other observer's origin has laboratory position

$$
R(t)=b+Ut+\frac12At^2,
\qquad x'=x-R(t).
$$

The parameters b,U,A have units meters,m/s and m/s squared. The clocks share Newtonian time. A prime labels coordinates assigned by the other observer, not a time derivative.

The displacement convention determines all signs. The moving origin itself has x'=0, and a laboratory event is reconstructed as x=x'+R(t). Stating that convention is more reliable than memorizing a force sign without specifying which frame moves.

## Differentiate the coordinate definition

Differentiating once and twice gives

$$
v'=v-U-At,\qquad a'=a-A.
$$

When A=0 this reduces to the Galilean case: velocities differ by a constant while accelerations agree. A nonzero A changes acceleration components as well.

The observer is assumed not to rotate. Rotation produces additional terms involving angular velocity and relative motion. Applying only minus A to a rotating coordinate system would omit part of its kinematics.

## Derive the inertial coordinate term

For a constant positive particle mass and prescribed constant physical force, the laboratory law is m*a=F. In the accelerating coordinates,

$$
m a'=F-mA.
$$

The term Finertial=-mA is introduced so that the coordinate acceleration satisfies the modified equation. Its effective force is Feffective=F+Finertial.

The inertial term is not another physical interaction with a new partner body. One should not invent a separate action-reaction partner for it inside this one-particle model. Physical force and inertial coordinate term are reported separately even though they appear together in the primed equation.

## Transform initial data and integrate consistently

At t=0, the appropriate initial data are

$$
x'_0=x_0-b,\qquad v'_0=v_0-U.
$$

The acceleration A of the origin does not add another initial-velocity shift at t=0. With constant F,m,A, the primed solution is

$$
x'(t)=x'_0+v'_0t+\frac12\left(\frac{F}{m}-A\right)t^2.
$$

Adding R(t) reproduces the laboratory trajectory x0+v0*t+F*t squared/(2m). Velocity reconstruction adds U+At. The consistency is exact for the declared model, including transformed initial conditions.

Keeping the physical law but forgetting the initial-data transformation would define another error. The interface separates a consistent solution from a counterfactual that uses the correct primed initial data but deliberately omits the inertial acceleration term.

## Work an unforced-particle example

Take m=1 kg,F=0,x0=0,v0=0,b=0,U=0 and A=1 m/s squared. The particle remains at laboratory position zero with laboratory acceleration zero.

The accelerating observer assigns x'=-t squared/2, v'=-t and a'=-1 m/s squared. At2 s, its position and velocity are-2 m and-2 m/s. The inertial coordinate term is-1 N and the modified equation gives m*a'=-1 N.

Reconstruction adds the origin's position2 m, returning the same laboratory position zero. The nonzero coordinate acceleration does not require an additional physical force on the particle.

## Zero primed acceleration does not eliminate physical force

Now use m=2 kg,F=2 N and A=1 m/s squared, with initial data and origin offsets zero. Laboratory acceleration is1 m/s squared. The observer accelerates at the same rate, giving a'=0.

At2 s the laboratory position is2 m, while the primed position remains zero. Physical force2 N and inertial term-2 N cancel in the primed equation. The physical force itself has not vanished.

A statement about apparent weight would additionally require a contact or measuring-device model. This free-particle coordinate calculation does not determine a support force, scale reading or elevator-wall interaction merely because an effective term is zero.

## Quantify the error from omitting the inertial term

Suppose a program uses the correct primed initial data but integrates xwrong'=x0'+v0'*t+F*t squared/(2m), as if the observer were inertial. Subtracting the correct primed trajectory gives

$$
x'_{\mathrm{wrong}}-x'=\frac12At^2.
$$

The error grows quadratically with time for fixed A. At t=0 the positions agree, so an initial-position check alone will miss the mistake. If A=0 the error vanishes for all time, recovering the inertial limit.

A plot can show the laboratory path, the correct physical reconstruction and the incorrect reconstruction. Their difference is a modeling error in coordinate dynamics, not an observed change of the physical laboratory experiment.

## Separate coordinate work from extra physical energy

For the constant effective acceleration, the primed kinetic-work identity is

$$
K'(t)-K'_0=(F-mA)\,[x'(t)-x'_0].
$$

Its right-hand side can be split into the physical-force term F*DeltaXprime and the inertial coordinate term(-mA)*DeltaXprime. Both are needed to match the primed kinetic-energy change in this representation.

In the unforced example at2 s, primed kinetic energy increases from0 to2 J. Physical force work in the primed coordinates is0 and inertial coordinate work is2 J. This does not insert a physical2 J source into the stationary laboratory particle. Kinetic energy is observer-dependent, and the inertial term belongs to the accelerating description.

Under the stated uniform terms this algebraic identity is exact. It is not a universal assertion that every non-inertial force model has a simple time-independent potential or that every energy interpretation transfers unchanged to rotating frames.

## Keep parameter changes and observer changes distinct

Changing A changes the observer's description while the laboratory force, mass and initial conditions can remain fixed. Changing F changes the physical laboratory model. Changing m at fixed F changes laboratory acceleration as well as the magnitude of the inertial term.

These controls should not be treated as interchangeable ways to make a plotted curve move. Inspecting one change at a time reveals which quantities are physical laboratory inputs and which describe the observer.

A negative A simply means acceleration opposite the selected positive axis. The inertial term reverses accordingly. A signed component is not inherently an invalid physical input.

## Interpret exact checks and sampled curves

Supported inputs are rational: mass uses kilograms directly, while other codes are divided by10 in their stated units. Reconstruction, modified-force balance, omission error and primed kinetic-work residuals are checked with exact reduced fractions.

Rendered curves use sampled decimal coordinates. They illustrate the analytic trajectory but are not the proof that every time satisfies the equation. At the selected time, exact labels keep the declared mathematical identities separate from plotting precision.

The figures compare actual time-dependent trajectories in this uniform-acceleration model. A force value or an inertial term should not be plotted as a spatial displacement without specifying an additional conversion rule. Position graphs use meters and time graphs use seconds.

## Implications for physical models learned by AI

A trajectory predictor trained on accelerating cameras can confuse coordinate acceleration with a new physical force if observer motion is absent from its inputs. A model may fit the observations while assigning an incorrect physical interpretation to its coefficients.

A useful evaluation transforms initial conditions and the observer trajectory consistently, reconstructs the same laboratory event and checks the modified equation. It also reports the scope: nonrotating constant translational acceleration is a tractable example, not a certificate for arbitrary camera motion or all non-inertial mechanics.

## Research extension: derive rotating-frame terms before interpreting energy

A camera can turn while its origin stays still. Then even a fixed laboratory point traces a curve in the camera's coordinates. Subtracting the camera origin's acceleration is no longer enough: changing axes also changes how relative velocity contributes to acceleration. We now derive that extension in a plane. The existing interactive lab remains the one-dimensional, nonrotating, translating observer; the two-dimensional examples below are standalone calculations.

### Define the rotation and differentiate twice

Let q be inertial laboratory position and O(t) a twice differentiable moving origin. The columns of R are the observer's orthonormal axes in laboratory coordinates. Choose constant angular rate Ω, positive counterclockwise, and let r be rotating coordinates. J rotates a vector counterclockwise by 90 degrees:

$$
R(t)=\begin{pmatrix}\cos\Omega t&-\sin\Omega t\\
\sin\Omega t&\cos\Omega t\end{pmatrix},\qquad
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad
\mathbf q=\mathbf O+R\mathbf r,\quad \dot R=R\Omega J.
$$

Because J²=−I, product differentiation gives two cross terms involving relative velocity:

$$
\dot{\mathbf q}=\dot{\mathbf O}+R(\dot{\mathbf r}+\Omega J\mathbf r),
\qquad
\ddot{\mathbf q}=\ddot{\mathbf O}
+R(\ddot{\mathbf r}+2\Omega J\dot{\mathbf r}-\Omega^2\mathbf r).
$$

The factor 2 is essential: one term differentiates R multiplying relative velocity and the other differentiates RΩJr. Multiply Newton's law by R transpose and move the extra terms to the force side:

$$
m\ddot{\mathbf r}
=R^T\mathbf F-mR^T\ddot{\mathbf O}
-2m\Omega J\dot{\mathbf r}+m\Omega^2\mathbf r.
$$

Read these terms in order: physical force in the new axes; translating-origin inertial term; Coriolis term; centrifugal term. All have units N. If Ω=0, R=I and the earlier translating-frame law is recovered. If the angular rate Ω(t) varies, use the rotation angle given by its time integral rather than Ωt. Differentiation then additionally gives the Euler force −m·dotΩ·Jr; the displayed constant-rate formula does not include it.

### A force can bend a trajectory while doing no work

A vector is perpendicular to its 90-degree rotation. More formally, the scalar v transpose Jv equals its transpose v transpose J transpose v, which is its own negative. Therefore:

$$
P_{\mathrm{Cor}}=-2m\Omega\dot{\mathbf r}^{\,T}J\dot{\mathbf r}=0,
\qquad J^T=-J.
$$

Zero instantaneous power constrains the speed contribution, not the full acceleration vector. A perpendicular acceleration changes direction. Dropping Coriolis from an equation because it does zero work would erase that directional effect.

### Work the stationary laboratory point example

Fix O=0, m=1 kg, Ω=1 rad/s and laboratory q=(1,0) m, with no physical force. Numerical t is in seconds. The correct rotating trajectory and its derivatives are:

$$
\mathbf r=(\cos t,-\sin t),\quad
\dot{\mathbf r}=(-\sin t,-\cos t),\quad
\ddot{\mathbf r}=-\mathbf r,\qquad
\mathbf F_{\mathrm{Cor}}=-2\mathbf r,\quad
\mathbf F_{\mathrm{cf}}=\mathbf r.
$$

The Coriolis and centrifugal vectors sum to −r N, matching m times the rotating acceleration. At t=0 they are (−2,0) N and (1,0) N respectively. There is no additional laboratory interaction: adding the coordinate terms is part of describing the same fixed point.

If Coriolis is omitted, the equation instead becomes r''=r. With the **same transformed initial position and velocity**, r(0)=(1,0) and r'(0)=(0,−1), its exact solution is:

$$
\mathbf r_{\mathrm{wrong}}(t)=(\cosh t,-\sinh t),
\qquad
\ddot{\mathbf r}_{\mathrm{wrong}}=\mathbf r_{\mathrm{wrong}}.
$$

At 1 s the position discrepancy is approximately 1.056854 m. Both predictions start with identical position and velocity, so checking only the initial state misses this error. Multiplication by R preserves the error norm, making the same discrepancy visible after reconstructing laboratory coordinates.

![Correct rotating coordinates of a fixed laboratory point remain on a unit circle; omission of Coriolis produces a hyperbolic path with the same initial position and velocity.](/assets/learning/classical-rotating-frame.en.svg)

The plot shows rotating **position** coordinates in metres with equal horizontal and vertical scales, for 0≤t≤2 s. Blue and dashed orange are separate model predictions, not two paths of the same correct physical experiment. The laboratory reference point stays fixed at (1,0) m.

### State the assumptions for a conserved rotating quantity

Set O=0 and keep Ω constant. Suppose physical force is −∇V for a time-independent potential invariant under these rotations, so V(Rr)=V(r). This includes a central spring and the free case. Then the centrifugal term has potential −mΩ²||r||²/2. Differentiating the following quantity and using the force equation leaves only Coriolis power, which is zero:

$$
E_{\mathrm{rot}}=\frac m2\|\dot{\mathbf r}\|^2+V(\mathbf r)
-\frac m2\Omega^2\|\mathbf r\|^2,
\qquad
\dot E_{\mathrm{rot}}=\dot{\mathbf r}\cdot
(m\ddot{\mathbf r}+\nabla V-m\Omega^2\mathbf r)=0.
$$

This is not the laboratory mechanical energy. Expand the laboratory velocity and use [r×Jr]_z=||r||² to obtain:

$$
E_{\mathrm{lab}}=\frac m2\|\dot{\mathbf r}+\Omega J\mathbf r\|^2+V,
\quad L=m\,[\mathbf r\times(\dot{\mathbf r}+\Omega J\mathbf r)]_z,
\qquad E_{\mathrm{rot}}=E_{\mathrm{lab}}-\Omega L.
$$

Both E_lab and L are conserved under the stated time-independent rotational symmetry, so the identity also verifies constancy of E_rot. For a nonrotational potential, its rotating representation generally depends on time; the simple conservation argument above no longer applies. A translating accelerating origin also requires a fresh energy balance.

The wrong free-point trajectory above also conserves E_rot=0 at every time: its velocity is (sinh t,−cosh t), whose squared norm equals the squared norm of (cosh t,−sinh t). The positive kinetic and negative centrifugal contributions cancel. Thus even an **exact energy-conservation test can accept the model that omitted Coriolis**. The full vector equation and physical reconstruction expose its error.

For a concrete circular laboratory solution of radius ρ with an isotropic spring, use:

$$
\mathbf q(t)=\rho(\cos\omega t,\sin\omega t),\quad
V=\frac12m\omega^2\|\mathbf q\|^2,\qquad
E_{\mathrm{lab}}=m\omega^2\rho^2,\quad
L=m\omega\rho^2,\quad
E_{\mathrm{rot}}=m\omega(\omega-\Omega)\rho^2.
$$

The free case ω=0 uses the same stationary laboratory point. With m=1 kg and ρ=1 m:

| ω (rad/s) | Ω (rad/s) | E_lab (J) | L (kg m²/s) | E_rot (J) |
|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 |
| 2 | 1 | 4 | 2 | 2 |
| 2 | 3 | 4 | 2 | −2 |

Changing Ω in the last two rows changes the observer, while the laboratory orbit and physical energy stay the same. Negative E_rot is not negative laboratory kinetic energy or evidence of dissipation. It contains the centrifugal potential and the −ΩL term.

### Reproduce the two-dimensional checks

This standard-library code checks force components, Coriolis power, physical reconstruction and the energy identity at 24 states. These floating-point residual checks illustrate the derived identities; they do not replace their algebraic proofs or validate a real camera.

```python
from math import sin, cos, cosh, sinh, hypot

def J(x):
    return (-x[1], x[0])

def rotate(x, angle):
    c, s = cos(angle), sin(angle)
    return (c*x[0]-s*x[1], s*x[0]+c*x[1])

max_defect = 0.0
for mass in [1.0, 2.0]:  # kg; radius = 1 m
    for omega, Omega in [(0.0,1.0), (2.0,1.0), (2.0,3.0)]:
        # Isotropic spring stiffness k=m*omega^2; omega=0 is free.
        nu = omega-Omega
        lab_energy, lab_L = mass*omega**2, mass*omega
        for t in [0.0, 0.3, 1.0, 2.0]:
            r = (cos(nu*t), sin(nu*t))
            v = tuple(nu*x for x in J(r))
            acc = tuple(-nu**2*x for x in r)
            physical = tuple(-mass*omega**2*x for x in r)
            coriolis = tuple(-2*mass*Omega*x for x in J(v))
            centrifugal = tuple(mass*Omega**2*x for x in r)
            defect = max(abs(mass*acc[j]-physical[j]
                             -coriolis[j]-centrifugal[j])
                         for j in range(2))
            max_defect = max(max_defect, defect)
            assert defect < 1e-12
            assert abs(sum(a*b for a,b in zip(v,coriolis))) < 1e-12
            jacobi = (mass*sum(x*x for x in v)/2
                      +mass*omega**2/2-mass*Omega**2/2)
            assert abs(jacobi-(lab_energy-Omega*lab_L)) < 1e-12
            lab_q = rotate(r, Omega*t)
            assert hypot(lab_q[0]-cos(omega*t),
                         lab_q[1]-sin(omega*t)) < 1e-12
        print("m, omega, Omega, E_lab, L, E_rot:",
              mass, omega, Omega, lab_energy, lab_L, jacobi)

# Free lab point, Omega=1: correct r=(cos t,-sin t).
# Omitting Coriolis gives r_wrong=(cosh t,-sinh t).
t = 1.0
error = hypot(cosh(t)-cos(t), sinh(t)-sin(t))
assert 1.05 < error < 1.06
print("position error after 1 s, metres:", round(error,6))
# The incorrect path still passes this rotating-energy conservation test.
for t in [0.0, 0.3, 1.0, 2.0]:
    wrong_r = (cosh(t), -sinh(t))
    wrong_v = (sinh(t), -cosh(t))
    wrong_energy = (sum(x*x for x in wrong_v)
                    -sum(x*x for x in wrong_r))/2
    assert wrong_energy == 0.0
print("incorrect path E_rot:", wrong_energy)
print("maximum force residual:", max_defect)
```

### Worked research questions

1. **At t=0, what acceleration would the free-point model predict if only the centrifugal term were kept?** It predicts (1,0) m/s²; the correct rotating acceleration is (−1,0) m/s². The missing Coriolis term is (−2,0) N for the 1 kg mass.
2. **Can equal energy values certify a correct direction of motion?** No. Work controls a scalar projection of acceleration onto velocity. Test the full transformed vector equation and reconstruct laboratory positions and velocities as well.
3. **What must a simulator trained on rotating observations include?** The observer rotation, transformed initial state and the velocity-dependent Coriolis term, with origin acceleration and Euler terms when applicable. Conserving E_rot under the stated symmetry is useful evidence, but it cannot substitute for those kinematic conditions.

All rotating formulas and examples in this extension are derived directly from q=O+Rr and Newton's law. They add a specified two-dimensional analysis without changing the existing translating-observer implementation.

## Exercises with worked solutions

1. **Exercise:** An unforced1 kg particle is at rest in the lab while A=1 m/s squared. What is a'? **Solution:** a'=0-1=-1 m/s squared and the inertial term is-1 N.
2. **Exercise:** At2 s in that example, reconstruct the laboratory position. **Solution:** x'=-2 m and R=2 m, so x=x'+R=0.
3. **Exercise:** If F=2 N,m=2 kg,A=1 m/s squared, has physical force vanished because a'=0? **Solution:** No. The physical2 N force is balanced by a-2 N inertial term in the coordinate equation.
4. **Exercise:** What position error follows from omitting the inertial term with otherwise correct primed initial data? **Solution:** A*t squared/2.
5. **Exercise:** Does nonzero inertial coordinate work imply an extra physical laboratory energy source? **Solution:** No. It contributes to the accelerating-coordinate kinetic-work identity; its interpretation differs from a new laboratory interaction.
6. **Exercise:** What changes when A=0? **Solution:** The inertial term and omission error vanish, and the Galilean inertial-observer transformation is recovered.

## Sources and further study

[MIT8.01, Chapter31, section31.2](https://live.ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/mit8_01scs22_chapter31.pdf) gives background on translating non-inertial frames and the inertial term. The constant-acceleration origin, exact initial-value comparisons, omission error and coordinate-work examples here are direct derivations under the stated assumptions.
