## Three conserved totals can coexist with complicated motion

Two interacting particles can accelerate, exchange momentum and trade kinetic energy for potential energy while their total momentum, angular momentum and energy remain constant. A conservation law constrains a total. It does not freeze each component or determine the current phase of the motion.

This lesson selects both particles as the system. Their masses are positive constants, no external force acts, and a declared central harmonic interaction couples them. We solve their actual planar motion and inspect three distinct conservation mechanisms. The particles are mathematical points; crossing paths do not trigger an undeclared contact or collision rule.

## Specify the system and count the interaction once

Let z=q1−q2 and let stiffness k>0. The forces and pair potential are

$$
F_1=-kz,\qquad F_2=+kz,\qquad V=\frac{k}{2}|z|^2.
$$

The potential belongs to the pair and is counted once, not once per particle. Its minimum is at zero separation. This is a declared zero-rest-separation model, not a rod or a finite-rest-length spring between solid bodies.

Selecting only particle1 would put particle2 outside the system boundary and make F1 external to that subsystem. The closed-system statements below apply to the selected two-particle system, while individual momenta generally change.

## Force cancellation conserves total linear momentum

Add the two Newton equations:

$$
P=m_1v_1+m_2v_2,\qquad \dot P=F_1+F_2=0.
$$

The equality-and-opposition of the pair is sufficient for this total force cancellation. It does not require either force to vanish. A nonzero initial P persists, and zero initial P allows equal-and-opposite momentum exchange.

In the momentum panel the total x component is compared with both individual x components. Both y-component balances are also checked numerically in the readout. A constant blue total and varying individual curves are compatible, rather than contradictory.

## Central direction also cancels internal torque

About a fixed laboratory origin,

$$
L=q_1\times m_1v_1+q_2\times m_2v_2,
$$

$$
\dot L=q_1\times F_1+q_2\times F_2
=(q_1-q_2)\times F_1=z\times(-kz)=0.
$$

Planar cross products denote their signed z components. Equal and opposite forces alone would leave the term z×F1; its vanishing here also uses central direction. Internal torque is therefore a separate calculation from total force.

The conserved angular momentum can be nonzero. Internal rotation does not require external torque to sustain its already specified angular momentum. The interaction changes individual velocities while preserving the aggregate.

## A potential-work relation conserves total energy

Define kinetic energy as the sum over both particles. Their total kinetic power is

$$
\dot K=F_1\cdot v_1+F_2\cdot v_2=-kz\cdot(v_1-v_2),
$$

$$
\dot V=kz\cdot\dot z=kz\cdot(v_1-v_2).
$$

Thus E=K+V is constant. The internal forces can do nonzero total work because the two particles have different displacements. Their equal-and-opposite vector sum does not cancel their work automatically.

Central direction by itself does not establish energy conservation for every possible interaction. Here we additionally supplied a time-independent potential with the correct force gradients. A velocity-dependent dissipative central interaction would require another energy budget.

## Separate center motion from relative motion

Let M=m1+m2 and define the reduced mass μ=m1m2/M. The center coordinate and relative coordinate satisfy

$$
R=\frac{m_1q_1+m_2q_2}{M},\qquad
\ddot R=0,\qquad \mu\ddot z=-kz.
$$

The center moves uniformly with velocity P/M. The relative oscillator has frequency ω=√(k/μ). Starting from the actual particle initial data, compute R0,z0 and w0=v10−v20. Then

$$
R(t)=R_0+\frac{P}{M}t,\qquad
z(t)=z_0\cos(\omega t)+\frac{w_0}{\omega}\sin(\omega t).
$$

Reconstruct the two particles using q1=R+(m2/M)z and q2=R−(m1/M)z; differentiate for velocities. Positive stiffness is part of this experiment. No division by a zero oscillator frequency is used.

## Decompose kinetic and angular quantities

Substitution and cancellation of cross terms give

$$
K=\frac{|P|^2}{2M}+\frac{\mu}{2}|\dot z|^2,
$$

$$
L=R\times P+\mu z\times\dot z.
$$

The first terms describe center motion; the second terms describe relative motion. Zero total momentum removes center kinetic energy but leaves internal kinetic energy. A stationary center does not imply stationary particles.

For this closed pair, center angular momentum is constant because R changes parallel to P. Relative angular momentum is constant because the relative force is central. The interface checks the decompositions as well as the totals; these are useful diagnostics against a wrong mass weighting.

## Change a fixed reference point without changing the system

For another fixed point D=d,

$$
L_D=L-d\times P.
$$

Both L and P are constant, so LD is constant too, although its value can differ. If total P=0, every fixed translated point reports the same total L. This is distinct from selecting one particle as a different subsystem, which changes the forces classified as external.

The comparison-point controls affect angular reporting only. Particle paths, forces and stored energy remain unchanged.

## Read the four panels with their own units

The spatial panel shows both actual paths and center motion in meters over ten seconds. Query dots match the particle/center line colors. The momentum panel uses kg m/s, the angular panel kg m²/s and the energy panel joules. Different units are not superimposed on a shared numerical axis.

Initial P,L,E are exact rational calculations from the integer control codes. Trigonometric trajectories and query residuals are floating-point evaluations of the analytic solution. Independent numerical integration checks both particle motion and accumulated internal work; heat or work is not invented from a residual to make a test pass.

## Research extension: budget externally modulated stiffness before testing energy conservation

Can a pair retain total momentum and angular momentum while its mechanical energy grows? Yes, if an external controller changes its interaction. This extension develops exercise six into an exactly reproducible example. It separates force cancellation, torque cancellation and energy supplied by a parameter change.

The native experiment above holds k fixed throughout a run and evaluates its analytic trajectory. The time-dependent protocol below is a standalone thought experiment and Python calculation. Keep constant positive particle masses, fixed inertial lab axes and a central force pair F1=−k(t)z,F2=−F1. The controller stays outside the selected particle system and supplies parameter work without a net force or torque in this idealization.

### Differentiate the energy that actually depends on time

Use the center/relative decomposition already derived, with P the total momentum:

$$
M=m_1+m_2,\quad \mu=\frac{m_1m_2}{M},\quad z=q_1-q_2,\quad w=\dot z,\qquad
E=\frac{\lVert P\rVert^2}{2M}+\frac\mu2\lVert w\rVert^2+\frac{k(t)}2\lVert z\rVert^2.
$$

The center kinetic term is constant because total force cancels. Differentiating the relative kinetic term uses μw'=−kz. Differentiating V requires both the changing separation and the changing parameter:

$$
\dot K=-kz\cdot w,\qquad
\dot V=kz\cdot w+\frac{\dot k}2\lVert z\rVert^2,\qquad
\dot E=\frac{\dot k}2\lVert z\rVert^2.
$$

The two k z dot w terms cancel; the explicit time derivative survives. A model that preserves P and L has not thereby established constant E. Units help interpret the remaining term: k' has units N/(m s), so k' times squared separation is power in watts. Its time integral is work in joules, rather than impulse.

### Account for a finite stiffness switch

For an ideal finite jump of k, position and velocity are continuous. A finite force jump does not give the particles a Dirac impulse. The controlled potential changes at the same position, producing:

$$
\Delta W=\frac{k_{\mathrm{after}}-k_{\mathrm{before}}}{2}\lVert z\rVert^2,
\qquad E(t)-E(0)=W_{\mathrm{control}}(0,t).
$$

Sum this contribution at every switch and include the smooth parameter-work integral on any smooth segment. Do not invent controller work from an unexplained trajectory residual. Here it is independently specified by the changes in k and the switch positions. A real finite-speed controller instead follows the smooth power law and may require its own dynamical state.

### Compose two exact quarter periods

Choose m1=m2=2 kg, so μ=1 kg. A constant-frequency segment of duration pi/(2 omega) maps each Cartesian component exactly by:

$$
\begin{pmatrix}z_j\\w_j\end{pmatrix}_{\!\mathrm{after}}
=Q_\omega\begin{pmatrix}z_j\\w_j\end{pmatrix}_{\!\mathrm{before}},\qquad
Q_\omega=\begin{pmatrix}0&1/\omega\\-\omega&0\end{pmatrix},\qquad
\Delta t=\frac{\pi}{2\omega}.
$$

The off-diagonal entries carry seconds and inverse seconds; position and velocity are different physical quantities. Now hold k_hi=4 N/m,omega_hi=2 s^-1 for pi/4 s, then k_lo=1 N/m,omega_lo=1 s^-1 for pi/2 s, and return to k_hi. With numerical SI values for the two frequencies:

$$
Q_1Q_2=\begin{pmatrix}-2&0\\0&-1/2\end{pmatrix},\qquad
(z,w)\longmapsto(w/2,-2z)\longmapsto(-2z,-w/2),\qquad
T=\frac{3\pi}{4}\ \mathrm{s}.
$$

The period matrix has determinant one and eigenvalues −2 and −1/2. It preserves the relative cross product while amplifying the initial-position direction and contracting the initial-velocity direction. Positive stiffness in every segment does not supply a constant-energy or uniform-amplitude guarantee under this externally powered protocol. Not every initial state grows: the z0=0 direction contracts. This linear parametric instability is not a demonstration of chaos.

### Track the actual pair and the supplied work

Set the stationary center R=0 and P=0. The particles have positions ±z/2 and velocities ±w/2. At n complete cycles, just after returning to the high stiffness:

$$
z_n=(-2)^nz_0,\quad w_n=(-1/2)^nw_0,\quad
L_n=\mu(z_n\times w_n)=L_0,\qquad
E_n=\frac{k_{\mathrm{hi}}}{2}4^n\lVert z_0\rVert^2
+\frac\mu2\,4^{-n}\lVert w_0\rVert^2.
$$

Here L is also the total lab angular momentum because the center contribution vanishes. For z0=(1,0) m,w0=0, the exact work budget is:

| cycle n | z_x (m) | E_n (J) | accumulated controller work (J) |
|---|---:|---:|---:|
| 0 | 1 | 2 | 0 |
| 1 | −2 | 8 | 6 |
| 2 | 4 | 32 | 30 |
| 3 | −8 | 128 | 126 |

![Energy rises from 2 to 8, 32 and 128 joules as stiffness is switched between positive values; the energy and stiffness panels share time but use different units.](/assets/learning/classical-parametric-pair-work.en.svg)

The upper energy axis uses base-two logarithmic spacing with ticks labelled in joules; the lower axis is stiffness in N/m. At zero separation the high-to-low switch does no work. At maximum separation the low-to-high switch adds energy. Energy stays constant between switches. The plotted scalar example has L=0; it does not imply that growth requires zero angular momentum.

Indeed z0=(1,0) m,w0=(0,1) m/s has L=1 kg m²/s. Its energy changes from 5/2 J to 65/8 J in the first cycle while L stays one. The first switch can extract energy and the second supply more: account for both, rather than treating each switch as necessarily positive work.

### Reproduce with exact quarter-period arithmetic

The code uses fractions and exact quarter-period maps, with μ=1 and numerical SI values. It checks two initial velocities over cycles zero through five, every constant-segment energy and both switch works. The matrix composition and formulas above establish the general identities; the finite examples check their implementation.

```python
from fractions import Fraction as F

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]

def energy(z, w, k):
    return (dot(w, w)+k*dot(z, z))/2  # reduced mass = 1

def quarter(z, w, omega):
    return tuple(x/omega for x in w), tuple(-omega*x for x in z)

for initial_w in [(F(0), F(0)), (F(0), F(1))]:
    z, w = (F(1), F(0)), initial_w
    E0, L0, work = energy(z, w, F(4)), cross(z, w), F(0)
    print("initial relative velocity:", initial_w)
    for n in range(6):
        assert z == (F((-2)**n), F(0))
        assert w == tuple(x*F(-1, 2)**n for x in initial_w)
        assert cross(z, w) == L0
        assert energy(z, w, F(4))-E0 == work
        assert energy(z, w, F(4)) == 2*F(4)**n+dot(initial_w, initial_w)*F(1, 4)**n/2
        print(n, "E:", energy(z, w, F(4)), "L:", L0, "work:", work)
        if n == 5:
            break
        before = energy(z, w, F(4))
        z, w = quarter(z, w, F(2))
        assert energy(z, w, F(4)) == before
        work += (F(1)-4)*dot(z, z)/2
        before = energy(z, w, F(1))
        z, w = quarter(z, w, F(1))
        assert energy(z, w, F(1)) == before
        work += (F(4)-1)*dot(z, z)/2
```

The quarter-period times pi/4 and pi/2 do not lie exactly on the native 0.1 s query grid. Growing states can also leave its bounded position/velocity input ranges. Reproduce the protocol with this independent code, rather than describing a sequence of native runs as an exact implementation. The unchanged constant-k widget remains useful for its original closed-pair conservation experiment.

### Worked questions for learned conservation constraints

1. **Does zero total force imply zero supplied energy?** No. Pair-force cancellation constrains P; the controller changes the interaction potential through k(t). Use the independently declared work term.
2. **Does determinant one prevent growth?** No. This period matrix expands one direction and contracts another. Preservation of a cross product or phase-area element does not bound each state component.
3. **What should a learned surrogate report?** State whether k is a fixed coefficient or an input protocol, record the controller-work convention and compare time-dependent states. P/L penalties alone cannot identify that protocol or validate its energy budget.

The switched protocol and values are derived here. The system-boundary and center/relative reasoning follows the conservation-law background cited below.

## Six exercises with worked solutions

1. **Exercise: internal oscillation.** Set equal1kg masses at(−1,0),(1,0), both initially at rest, with k=1. **Solution:** ω=√2 and q1x=−cos(√2t),q2x=cos(√2t). Total P=L=0 but particles move. K=2sin²(√2t),V=2cos²(√2t),so E=2J.
2. **Exercise: avoid double counting.** What is the initial potential in exercise1? **Solution:** separation has squared length4, so V=4/2=2J. Counting that same pair twice would incorrectly give4J.
3. **Exercise: internal rotation.** Give particle1 velocity(0,−1) and particle2(0,1) in exercise1. **Solution:** each initial angular contribution is1, so L=2. Initial K=1J and V=2J give E=3J. The relative motion is elliptical and does not require an external torque.
4. **Exercise: drifting unequal masses.** Use m1=1,m2=2, the same initial positions, v1=(1,1),v2=(1,−1/2). **Solution:** P=(3,0),R=(1/3+t,0),L=−2 and E=17/4J. Center kinetic energy is3/2J; the remaining11/4J is internal kinetic plus pair potential.
5. **Exercise: translated angular reference.** In exercise4 choose D=(0,1). **Solution:** d×P=−3,so LD=−2−(−3)=1. Both angular quantities are conserved although their values differ. Particle trajectories remain identical.
6. **Exercise: a thought variation beyond the controls.** If k were externally changed with time, would all three totals still be conserved? **Solution:** equal-and-opposite central forces still cancel total force and torque. But dE/dt=(dk/dt)|z|²/2,so energy generally changes through the external modulation. The current widget keeps k constant.

## Connection to learning physical systems

Conservation constraints do not determine the force law or trajectory phase. A learned simulator may preserve total P while getting internal oscillation wrong, or preserve energy while using incorrect mass weighting. Compare trajectories, force directions, decompositions and independent work integrals together. The algebra proves identities for the declared model; measurements are needed to determine whether the model fits a physical apparatus.

## Background and scope

For system-level conservation assumptions, see [MIT OCW, Conservation Laws for Systems of Particles](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/5b68bc0342763cf099422af5a075940a_MIT16_07F09_Lec11.pdf). The harmonic pair and worked values above are derived here. Distributed fields, rigid-body spin, contact and dissipation require additional state or interactions.
