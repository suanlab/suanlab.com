## The net force changes velocity, not just position

A particle can keep moving while its forces balance. It can also have zero velocity at one instant while a nonzero force continues to accelerate it. These observations are compatible because Newton's second law relates net force to acceleration, the time derivative of velocity.

This lesson uses a constant positive point-particle mass in a Newtonian inertial laboratory. Two prescribed constant external forces act in a plane. We solve the resulting initial-value problem exactly and compare force addition, spatial motion, momentum balance and work. There is no contact, rigid-body rotation or changing-mass process hidden inside the example.

## Choose the particle and its external interactions

The selected system is one particle of mass m>0. Let the two external force vectors be F1 and F2. Their net force is

$$
\mathbf F_{\mathrm{net}}=\mathbf F_1+\mathbf F_2,
\qquad m\mathbf a=\mathbf F_{\mathrm{net}}.
$$

Every component has its own signed coordinate. Adding magnitudes without directions generally gives the wrong result. For example, a2 N rightward force and a2 N upward force produce the vector(2,2) N, not a4 N rightward force.

Both forces in this sum act on the selected particle. An action-reaction partner acting on a different body is not another term in this one-particle force balance. Equal and opposite forces on the same particle can balance, but that is a different statement from identifying a Newton-third-law pair across two bodies.

## Keep force geometry separate from position geometry

A force-plane drawing has horizontal and vertical axes measured in newtons. Arrows from its origin represent force components. A head-to-tail translated copy is a vector-addition construction; its apparent location is not the physical attachment point of another force.

The spatial trajectory uses meters on both axes. Its shape depends on the initial velocity and acceleration. We should not draw a force arrow on a position axis without declaring an additional scale that converts the arrow into a displacement illustration.

Maintaining equal scales within each two-dimensional plane also matters. Stretching one spatial axis changes apparent angles and curvature. The numerical model retains actual components; the plot should not quietly distort their geometry.

## Solve the constant-force initial-value problem

The illustration fixes initial position to the origin and allows an arbitrary initial velocity v0. Since the forces and mass are constant, acceleration is constant and integration gives

$$
\mathbf a=\frac{\mathbf F_{\mathrm{net}}}{m},\qquad
\mathbf v(t)=\mathbf v_0+\frac{\mathbf F_{\mathrm{net}}}{m}t,
\qquad
\mathbf q(t)=\mathbf v_0t+\frac{\mathbf F_{\mathrm{net}}}{2m}t^2.
$$

Velocity has units m/s and position has units m. Differentiate q to recover v, then differentiate v to recover a. These checks confirm that the displayed solution satisfies both the differential law and the declared initial data.

The law alone does not select a unique trajectory without initial conditions. Two particles under the same mass and forces can follow different paths if their initial velocities differ. A force-space diagram cannot replace the initial-value specification.

## Work through a genuinely two-dimensional example

Choose m=2 kg,F1=(2,0) N,F2=(0,2) N and v0=(1,0) m/s. The net force is(2,2) N, so acceleration is(1,1) m/s squared.

At t=2 s, velocity is(3,2) m/s and position is(4,2) m. The spatial path is curved because the initial velocity is not parallel to acceleration. The velocity evolves along a straight line in velocity space, while position contains a quadratic time term.

Initial kinetic energy is1 J and query kinetic energy is13 J. Force1 does8 J of work and force2 does4 J, totaling12 J. The calculation connects the vector motion to energy without replacing the force law with an unrelated scalar shortcut.

## Balanced forces allow uniform motion

Now take F1=(2,0) N,F2=(-2,0) N with the same mass and initial velocity. Net force and acceleration are zero, but the velocity remains(1,0) m/s. At2 s the particle has moved to(2,0) m.

It is in force balance, but it is not stationary. A stationary motion would additionally require initial velocity zero. A numerical flag for balanced forces should therefore remain separate from a flag for rest throughout the motion.

The forces can still do individual signed work: after that displacement, their works are4 J and-4 J. Total work is zero and kinetic energy stays constant. Zero net work does not require every individual work term to vanish.

## A turning instant is not a static equilibrium

Consider m=1 kg,F1=(-1,0) N,F2=0 and v0=(1,0) m/s. At1 s, velocity is zero and position is(0.5,0) m. Acceleration remains(-1,0) m/s squared.

Immediately afterward the velocity becomes negative in the x direction. The particle was momentarily at rest, not at a stationary equilibrium. Reading one zero velocity value cannot certify zero net force or a constant future position.

This distinction also applies at the initial instant. Starting at rest under a nonzero force does not make the force disappear. A model should report the instantaneous velocity and the entire-motion property separately.

## Verify the vector momentum balance

With constant mass, momentum is P=m*v. Subtracting its initial value gives

$$
\mathbf P(t)-\mathbf P_0=\mathbf F_{\mathrm{net}}t.
$$

Both vector components must satisfy the balance. At t=0 the momentum change is zero even if the force is nonzero; that single equality does not establish momentum conservation for all times.

The constant-mass assumption is explicit. For an open system gaining or losing material, momentum carried across the system boundary must be accounted for. Substituting a changing mass into this point-particle formula without specifying mass transfer is not an authorized extension of this model.

## Derive the kinetic-work identity

For kinetic energy K=m*v dot v/2, substitution of the velocity solution gives

$$
K(t)-K_0
=\mathbf F_{\mathrm{net}}\cdot\mathbf v_0t
+\frac{\|\mathbf F_{\mathrm{net}}\|^2}{2m}t^2
=\mathbf F_{\mathrm{net}}\cdot\mathbf q(t).
$$

Because the individual forces are constant,

$$
W_1+W_2=\mathbf F_1\cdot\mathbf q+\mathbf F_2\cdot\mathbf q
=K(t)-K_0.
$$

Work is signed. A force opposing displacement can contribute negative work. The turning example loses0.5 J of kinetic energy by the query time. Nothing in this calculation identifies every possible thermal or internal-energy transfer; kinetic energy is the quantity being balanced under the declared particle model.

## Understand the controls and exact checks

Mass uses kilograms directly. Force and initial-velocity codes are divided by10, and the time code is divided by10 seconds. The initial position is fixed to zero. All supported query values and balance flags use reduced rational arithmetic.

The sampled spatial path uses decimal coordinates for rendering. Its geometric appearance illustrates the analytic solution rather than proving a balance by sampled agreement. The query time moves a selected state along the declared solution; it does not change the constant-force law.

Changing mass while holding physical forces fixed changes acceleration. Changing initial velocity can change the whole spatial path while preserving the force diagram. These are different experimental changes and should be inspected one at a time.

## Implications for learned dynamics

A learned acceleration model should respond to vector superposition and mass in the stated way if that is its intended physical domain. Checking only one component can hide a missing force in another direction. Checking only velocity at a turning instant can also hide persistent acceleration.

An informative evaluation reports the selected system, force inputs, mass, initial data and observer frame. It distinguishes exact model identities from empirical correctness of the model in a real apparatus. A successful constant-force prediction does not by itself validate contact laws, rotations or changing-mass behavior outside the lesson's assumptions.

## Research extension: turn a force residual into a trajectory bound

Suppose a learned force makes a small error. Does that mean its predicted position is accurate? A force error changes velocity, and the changed velocity then accumulates into position. Earlier errors have longer to act. This gives a quantitative connection between a force loss and a trajectory evaluation, provided the loss actually controls the force over the full interval.

### Define what is being compared

Use two exact Newtonian trajectories with the same constant positive mass m and the same inertial frame. Their initial states may differ. Define the signed vector residual and errors as follows:

$$
\mathbf e_q=\widehat{\mathbf q}-\mathbf q,\quad
\mathbf e_v=\widehat{\mathbf v}-\mathbf v,\quad
\mathbf r(s)=\widehat{\mathbf F}(s)-\mathbf F(s),\qquad
m\dot{\mathbf e}_v=\mathbf r.
$$

For prescribed time-dependent forces, the residual is simply the difference of those functions. For state-dependent forces, evaluate each force **along its own actual trajectory**: the hatted force at the hatted state and the reference force at the reference state. This extension is a standalone analysis of continuous motion; the existing interactive controls still use two constant forces.

Integrate the velocity-error equation, then integrate position error. Interchanging the integrals over 0≤s≤u≤t gives the kernel t−s: a force error at time s acts for the remaining time. Integrable residuals suffice for these identities, with absolutely continuous velocities:

$$
\mathbf e_v(t)=\mathbf e_{v0}+\frac1m\int_0^t\mathbf r(s)\,ds,
\qquad
\mathbf e_q(t)=\mathbf e_{q0}+t\mathbf e_{v0}
+\frac1m\int_0^t(t-s)\mathbf r(s)\,ds.
$$

These equations retain initial-state errors. An experiment cannot attribute all position error to its learned force while quietly changing initial velocity.

### Uniform and integral budgets answer different questions

If the Euclidean force norm satisfies ||r(s)||≤ε throughout [0,t], apply the triangle inequality and integrate the nonnegative kernel:

$$
\|\mathbf e_v(t)\|\le\|\mathbf e_{v0}\|+\frac{\epsilon t}{m},\qquad
\|\mathbf e_q(t)\|\le\|\mathbf e_{q0}\|+t\|\mathbf e_{v0}\|
+\frac{\epsilon t^2}{2m}.
$$

The force contribution is linear in time for velocity and quadratic for position. With identical initial states and a constant residual εu along any unit vector u, both bounds are attained. A mass of 2 kg and residual magnitude 1 N give the following exact values:

| t (s) | velocity error norm (m/s) | position error norm (m) |
|---|---|---|
| 0 | 0 | 0 |
| 1/2 | 1/4 | 1/16 |
| 1 | 1/2 | 1/4 |
| 2 | 1 | 1 |

If an integrated squared force error is justified instead, Cauchy–Schwarz supplies another bound:

$$
\|\mathbf e_v(t)\|\le\|\mathbf e_{v0}\|+\frac{\sqrt{t}}{m}B_t,
\qquad
\|\mathbf e_q(t)\|\le\|\mathbf e_{q0}\|+t\|\mathbf e_{v0}\|
+\frac{\sqrt{t^3/3}}{m}B_t,
\quad B_t^2=\int_0^t\|\mathbf r(s)\|^2\,ds.
$$

To obtain the position constant, square the scalar kernel and integrate: integral of (t−s)² is t³/3. The velocity kernel is 1 and its squared integral is t. B_t has units N√s, so both right-hand sides have the required velocity or position units. The different powers are not interchangeable error metrics.

At a fixed horizon T, a residual c(T−s)u attains the position L2 bound. A constant aligned residual attains the velocity L2 bound. They are different extremizers. For T=2 s, m=2 kg and c=1 N/s, the position error is 4/3 m; B_T²=8/3 N²s and the squared upper bound is exactly 16/9 m². This proves the constants can be sharp without a random simulation.

### A zero training loss at three times can still miss motion

Take identical initial states, A>0 in newtons, τ>0 in seconds and the following residual:

$$
\mathbf r(t)=A\,p(t/\tau)\mathbf u,\quad
p(u)=u^2(u-1)^2(u-2)^2,\quad
\mathbf u=(3/5,4/5),\quad 0\le t\le2\tau.
$$

All three samples t=0,τ,2τ give zero force error, independent of A. Yet p is positive between its zeros. Expanding gives p(u)=u⁶−6u⁵+13u⁴−12u³+4u². Integrating this polynomial exactly yields:

$$
\int_0^2p(u)\,du=\int_0^2(2-u)p(u)\,du=\frac{16}{105},
\quad
\|\mathbf e_v(2\tau)\|=\frac{16A\tau}{105m},
\quad
\|\mathbf e_q(2\tau)\|=\frac{16A\tau^2}{105m}.
$$

For A=1 N, τ=1 s and m=2 kg, both numerical error values are 8/105, with their respective units m/s and m. Increasing A makes them arbitrarily large while the three sampled errors remain zero. The uniform and L2 bounds remain valid; those three zeros simply do not supply either required continuous budget.

![A force residual vanishes at the three observation times but remains positive between them; the resulting position error accumulates to 8/105 metres.](/assets/learning/classical-force-residual.en.svg)

The upper panel shows force in N and the lower panel position-error norm in m, for the stated A=1, τ=1, m=2 example. The horizontal axes are physical seconds. The markers in the upper panel are the three exact zero samples, not an assertion that the intervening curve was observed.

### Reproduce the exact examples

The code uses rational arithmetic and numerical values in the declared SI units. The polynomial argument is dimensionless; it equals numerical time only because τ=1 s. It does not fit a polynomial from the three zeros.

```python
from fractions import Fraction as F

mass = F(2)  # kg; numerical times below are in seconds
for t in [F(0), F(1,2), F(1), F(2)]:
    # Constant 1 N residual along the unit vector (3/5, 4/5).
    dv, dq = t/mass, t*t/(2*mass)
    print("constant: t, velocity error, position error:", t, dv, dq)

# At T=2 s, residual (2-t) N saturates the position L2 bound.
T = F(2)
B_squared = T**3/3  # integral of (T-t)^2; units N^2 s
position = T**3/(3*mass)
assert position**2 == T**3/(3*mass**2) * B_squared
assert position == F(4,3)
print("L2 extremizer: position error =", position)

# A=1 N, tau=1 s; p(t/tau) in dimensionless time u=t/tau.
p = [F(0), F(0), F(4), F(-12), F(13), F(-6), F(1)]
value = lambda u: sum(c*u**i for i,c in enumerate(p))
assert all(value(F(u)) == 0 for u in [0,1,2])
impulse = sum(c*T**(i+1)/F(i+1) for i,c in enumerate(p))
position = sum(c*T**(i+2)/F((i+1)*(i+2))
               for i,c in enumerate(p))/mass
assert impulse == F(16,105)
assert impulse/mass == position == F(8,105)
print("zero sampled force error; velocity and position errors:",
      impulse/mass, position)
```

### Worked research checks

1. **How long does a uniform 0.02 N budget keep the force-only position error below 0.01 m for m=2 kg?** With matching initial states, 0.02t²/(2·2)≤0.01 gives t≤√2 s. This is a sufficient exact-flow guarantee if the continuous budget is valid throughout that interval.
2. **Does a small force error evaluated only on the reference trajectory establish the same bound for a feedback model?** No. It does not yet control the force at the predicted states. A justified state-regularity bound, such as a Lipschitz estimate and the resulting stability argument, is additional evidence.
3. **Does this include the error of a discrete numerical solver?** No. Compare exact flows here. For computed trajectories, separately control discretization or construct an interpolant with a justified differential residual; a list of time-step outputs alone does not provide it.

The kernel formulas, inequalities and polynomial counterexample are direct derivations under the assumptions above. They connect a declared continuous force budget to motion; they do not certify an empirical force law from sparse data.

## Exercises with worked solutions

1. **Exercise:** Add(2,0) N and(0,2) N for a2 kg particle. **Solution:** Net force is(2,2) N and acceleration is(1,1) m/s squared.
2. **Exercise:** With v0=(1,0) m/s, find the state after2 s. **Solution:** Velocity is(3,2) m/s and position is(4,2) m.
3. **Exercise:** Do opposite forces require the particle to stop? **Solution:** No. Zero net force preserves the initial velocity, which can be nonzero.
4. **Exercise:** Is zero velocity at the turning instant a force-balance certificate? **Solution:** No. The example at1 s has zero velocity and acceleration(-1,0) m/s squared.
5. **Exercise:** What are the two works in the two-dimensional example? **Solution:** F1 dot q=8 J and F2 dot q=4 J, matching the12 J kinetic increase.
6. **Exercise:** Should a reaction force on another body be added to this particle's free-body sum? **Solution:** No. The sum contains forces acting on the selected particle.

## Sources and further study

[NASA Glenn's Newton's Second Law](https://www.grc.nasa.gov/WWW/K-12/BGP/newton2.html) gives background on constant-mass vector components. The initial-value solution, balanced/turning examples and exact momentum/work checks above are direct derivations for this declared two-force particle model.
