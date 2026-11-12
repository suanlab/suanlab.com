## Two moving particles can have zero total momentum

Imagine two carts initially at rest, with an internal actuator pushing them in opposite directions. The center of mass can stay fixed while both carts acquire speed. Adding their momenta gives zero; adding their kinetic energies gives a positive number. Cancellation of signed momentum is different from cancellation of motion or energy.

This lesson replaces the actuator with a deliberately simple force law: a prescribed constant pair +f on particle1 and −f on particle2. Independent external forces E1 and E2 may also act. Both masses are constant and positive. We solve one-dimensional Newtonian motion in an inertial frame. The pair is a declared mathematical interaction, rather than an automatic spring, rope or contact model.

## Start with a separate law for each particle

Choose signed positions q1,q2 and velocities v1,v2. The force laws are

$$
m_1a_1=f+E_1,\qquad m_2a_2=-f+E_2.
$$

Only forces acting on the particle belong in its own equation. The partner force belongs on the other particle's equation. Constant accelerations give

$$
v_i(t)=v_{i0}+a_it,\qquad
q_i(t)=q_{i0}+v_{i0}t+\frac12a_it^2.
$$

Initial positions need not coincide, and initial velocities need not vanish. A negative interaction reverses the directions of the internal pair. The two particle paths may cross. Crossing is allowed for this point-particle model and does not trigger an impact, exchange of velocities or change in the prescribed forces.

## Add the equations to obtain total momentum balance

Define total momentum P=m1v1+m2v2. Adding the equations cancels f against −f:

$$
\dot P=E_1+E_2,\qquad
P(t)-P(0)=(E_1+E_2)t.
$$

The cancellation happens because both members of the interaction pair lie inside the selected two-particle system. It does not say that either particle has zero force. When the total external force is zero, total momentum remains its initial value. That value can be zero or nonzero. Conservation constrains the sum; individual momenta can still change by equal and opposite amounts.

Opposing external forces also deserve care. If E1=−E2, total momentum is conserved even though each external force can perform work. Zero net external force is a statement about a sum of forces, not a statement that the surroundings have no effect on relative motion or energy.

## Center of mass is a mass-weighted position

Let M=m1+m2 and define

$$
R=\frac{m_1q_1+m_2q_2}{M},\qquad
\dot R=\frac{P}{M},\qquad
\ddot R=\frac{E_1+E_2}{M}.
$$

The fixed masses allow differentiation without extra mass-transfer terms. The center behaves like a representative point carrying the total mass under the total external force. It is a derived coordinate, not a third particle with its own extra interaction.

For unequal masses, R lies closer to the heavier particle. The arithmetic midpoint is correct only for equal masses. A constant center position requires zero total external force and zero initial total momentum in this constant-force model. Zero center acceleration alone permits uniform center motion.

## Changing the boundary changes which force is external

The interactive selection code0 includes both particles. Code1 selects only particle1; code2 selects only particle2. Changing this code does not change physical motion. It changes the momentum balance being inspected:

$$
\dot P_{\{1\}}=f+E_1,\qquad
\dot P_{\{2\}}=-f+E_2.
$$

For a one-particle system the partner lies outside the boundary, so its force is external to that subsystem. Calling f an internal force once and then discarding it from every possible subsystem balance would be an error. Always name the system before classifying a force.

The exact table reports selected momentum, selected external force and selected impulse. The full-system center and total momentum remain visible as separate reference quantities. In particular, total momentum may be conserved while selected one-particle momentum is not.

## Relative motion stores an independent kinetic contribution

Let relative velocity be v1−v2 and reduced mass μ=m1m2/M. Expanding the squares gives

$$
K=\frac12m_1v_1^2+\frac12m_2v_2^2
=\frac{P^2}{2M}+\frac12\mu(v_1-v_2)^2.
$$

To see the algebra, write v1=P/M+(m2/M)(v1−v2) and v2=P/M−(m1/M)(v1−v2). The cross terms cancel after mass weighting. The first contribution describes center motion; the second describes relative motion. Both are nonnegative for positive masses.

If P=0, only the first contribution vanishes. The particles may still have relative speed and positive kinetic energy. Consequently, momentum conservation alone cannot certify kinetic-energy conservation, a collision type or the absence of an internal energy exchange.

## Internal forces can do nonzero total work

Let Δqi=qi(t)−qi(0). Since all prescribed forces are constant,

$$
W_{\mathrm{int}}=f(\Delta q_1-\Delta q_2),\qquad
W_{\mathrm{ext}}=E_1\Delta q_1+E_2\Delta q_2,
$$

$$
K(t)-K(0)=W_{\mathrm{int}}+W_{\mathrm{ext}}.
$$

Equal and opposite internal forces do not imply zero internal work: they act through generally different displacements. Adding forces is sufficient for momentum balance; calculating work additionally needs the motion at each point of application. This is why a fixed center can coexist with increasing kinetic energy.

## Declare the potential before claiming mechanical conservation

For this constant pair, choose the linear potential

$$
V(q_1,q_2)=-f(q_1-q_2).
$$

Its negative gradients give +f and −f. Thus ΔV=−Wint and

$$
\Delta(K+V)=W_{\mathrm{ext}}.
$$

With zero external work, the declared mechanical energy is conserved even if K changes. A different potential zero adds a constant without changing this balance. The linear potential is unbounded over unrestricted positions; it is a mathematical example, not a universal material law. An actuator with a finite reservoir or a spring with a displacement-dependent force would need additional state or another interaction law.

## Interpret the plots and boundary readouts

The position panel shows q1 in blue, q2 in orange and R in green against time. These curves share meters on the vertical axis. The momentum panel compares the actual total momentum with P0+(E1+E2)t in kg m/s. Agreement is expected because the same declared dynamics obey the balance; overlapping curves are not evidence from an independent experiment.

Use the query control for exact current values, and then change just the boundary code. Positions, velocities and total momentum should stay fixed while selected force and selected momentum may change. Reverse the internal pair, try unequal masses, and compare an opposing external pair with both external forces zero. These changes distinguish aggregate motion from relative dynamics.

## Research extension: separate observable motion from an unknown force assignment

The simulator declares its internal and external forces as known inputs. In an inverse problem, an observer may instead measure positions and accelerations while the external forces are uncalibrated. Those are different information conditions. We will show why even exact trajectories can leave the internal force unknown, and what additional measurement resolves the ambiguity.

### Project external forces onto center and relative motion

Keep the same positive constant masses and one-dimensional inertial model. The internal pair is +f,−f, and all three forces f,E1,E2 are constant. Change coordinates from q1,q2 to center R and relative position r:

$$
M=m_1+m_2,\quad \mu=\frac{m_1m_2}{M},\quad
q_1=R+\frac{m_2}{M}r,\quad q_2=R-\frac{m_1}{M}r,
\qquad r=q_1-q_2.
$$

Add and subtract the particle equations with their mass weights:

$$
M\ddot R=E_1+E_2,\qquad
\mu\ddot r=f+g,\quad
 g=\frac{m_2E_1-m_1E_2}{M}.
$$

A zero external sum fixes center acceleration but leaves the external relative force g unspecified. For example, E1=m1A,E2=m2A produces center acceleration A and g=0. The pair E1=G,E2=−G produces no center acceleration but contributes g=G to relative dynamics. Center observations alone cannot distinguish these effects on the internal motion.

The same separation appears in work. Substitute the coordinate differentials into the two external work contributions:

$$
E_1\,dq_1+E_2\,dq_2=(E_1+E_2)\,dR+g\,dr,\qquad
f\,dq_1-f\,dq_2=f\,dr.
$$

The net external force multiplies center displacement; its mass-weighted difference multiplies relative displacement. This explains why zero net external force can coexist with nonzero external work. Equal and opposite internal forces contribute through relative motion as well.

### Find the missing direction in the observation map

Assume masses and complete noiseless trajectories are known, so both accelerations can be differentiated exactly. This is deliberately stronger than a noisy finite-data experiment. If the separate force components are unknown, acceleration observes only:

$$
\underbrace{\begin{pmatrix}m_1a_1\\m_2a_2\end{pmatrix}}_{\mathbf y}
=\underbrace{\begin{pmatrix}1&1&0\\-1&0&1\end{pmatrix}}_{H}
\underbrace{\begin{pmatrix}f\\E_1\\E_2\end{pmatrix}}_{\boldsymbol\theta},
\qquad H\begin{pmatrix}1\\-1\\1\end{pmatrix}=\mathbf0.
$$

H has rank two: its second and third columns are independent. Its null space is one-dimensional. Adding any constant c times the displayed null vector preserves both observed accelerations:

$$
(f,E_1,E_2)\longmapsto(f+c,E_1-c,E_2+c),\qquad
E_1+E_2\longmapsto E_1+E_2,\quad g\longmapsto g-c.
$$

The change in g cancels the change in f, preserving relative acceleration; the external sum preserves center acceleration. With the same initial conditions, every particle trajectory, momentum and kinetic energy is identical for the whole observation interval. More samples of those same trajectories cannot remove a null direction that is present at every time.

The force assignment and the declared interaction potential −fr do change. This inverse problem does not invalidate the native lab: that lab supplies f,E1,E2 explicitly rather than estimating them from motion alone.

### Work three indistinguishable assignments

Use m1=1 kg,m2=2 kg, initial positions −1 m and 1 m, and zero velocities. Start from f=1 N,E1=E2=0. Apply c=−2,0,2 N. All three resulting input sets fit the native force ranges and give these same trajectories in SI numerical units:

$$
q_1(t)=-1+\frac12t^2,\quad q_2(t)=1-\frac14t^2,\quad
v_1=t,\quad v_2=-\frac12t,\quad R=\frac13,\quad P=0.
$$

At 1 s, the displacements are 1/2 m and −1/4 m, so relative displacement is 3/4 m and kinetic energy has increased by 3/4 J. The work assignment varies:

| f (N) | E1 (N) | E2 (N) | internal work (J) | external work (J) |
|---|---|---|---|---|
| −1 | 2 | −2 | −3/4 | 3/2 |
| 1 | 0 | 0 | 3/4 | 0 |
| 3 | −2 | 2 | 9/4 | −3/2 |

Each row sums to 3/4 J and satisfies the mechanical balance with **its own declared potential**. Observed kinetic motion alone cannot decide which internal/external split generated the data. A separately calibrated interaction-energy or external-force measurement would provide information missing from those trajectories.

![Three different internal and external force assignments produce the same particle trajectories and center, while their internal and external work contributions differ and sum to the same kinetic change.](/assets/learning/classical-center-force-identification.en.svg)

The upper panel shows the shared trajectories in metres, not three separate colored force assignments. The lower panel compares signed work in joules for the three table rows at 1 s. Its paired bars always sum to the same 3/4 J. Position and work have distinct axes and units.

### A unique regularized answer can still miss the generating force

A learner might select the smallest squared-force penalty R=f²+E1²+E2² among exact fits. With observed force components y=(1,−1) N, write all forces relative to F_ref=1 N to keep the arithmetic dimensionless:

$$
\phi=\frac{f}{F_{\mathrm{ref}}},\quad F_{\mathrm{ref}}=1\ \mathrm N,
\quad \frac{\mathcal R}{F_{\mathrm{ref}}^2}
=\phi^2+(1-\phi)^2+(-1+\phi)^2
=\frac23+3\left(\phi-\frac23\right)^2.
$$

The unique minimum is f=2/3 N,E1=1/3 N,E2=−1/3 N. It fits every observed trajectory exactly, yet differs from the generating assignment f=1 N,E1=E2=0. The prior selects one point on the indistinguishable line; it does not add a calibration observation. Other penalty weights can select a different point.

The continuous minimum is a standalone exact-fraction calculation. Native force codes have 0.1 N resolution: codes 7,3,−3 give forces 0.7,0.3,−0.3 N and the same accelerations. That is the minimum penalty on the supported exact-fit grid, 67/100 N². It still differs from the generating force. The code checks the grid explicitly.

### Add information that breaks the null direction

If E1 is independently calibrated, the first particle equation determines f. With exactly known mass and deterministic error bounds, the triangle inequality also quantifies uncertainty:

$$
f=m_1a_1-E_1,\qquad
|\widehat f-f|\le m_1\epsilon_a+\epsilon_E
\quad\text{if}\quad
|\widehat a_1-a_1|\le\epsilon_a,\quad |\widehat E_1-E_1|\le\epsilon_E.
$$

For m1=1 kg, acceleration error at most 0.02 m/s² and external-force error at most 0.03 N imply force error at most 0.05 N. This is a deterministic bound under those premises, not a confidence interval inferred without a noise model. Known mass and calibrated external force are substantive additional information.

### Reproduce with exact arithmetic

The rational example checks trajectories at four times for each of the three assignments, both work balances, center/relative equations, the continuous penalty minimum and the native grid minimum. The general all-time equivalence follows from the null-space proof above, rather than from agreement at sampled times.

```python
from fractions import Fraction as F

m1,m2 = F(1),F(2)  # kg; q10=-1 m, q20=1 m, v10=v20=0
M = m1+m2
mu = m1*m2/M

def state(f,e1,e2,t):
    a1,a2 = (f+e1)/m1,(-f+e2)/m2
    q1,q2 = -1+a1*t*t/2,1+a2*t*t/2
    v1,v2 = a1*t,a2*t
    R = (m1*q1+m2*q2)/M
    K = (m1*v1*v1+m2*v2*v2)/2
    wi = f*((q1+1)-(q2-1))
    we = e1*(q1+1)+e2*(q2-1)
    potential,initial_potential = -f*(q1-q2),2*f
    assert K == wi+we
    assert K+potential-initial_potential == we
    assert M*((m1*a1+m2*a2)/M) == e1+e2
    assert mu*(a1-a2) == f+(m2*e1-m1*e2)/M
    return (q1,q2,v1,v2,R,K),wi,we

for c in [F(-2),F(0),F(2)]:
    f,e1,e2 = 1+c,-c,c
    for t in [F(0),F(1,2),F(1),F(2)]:
        observed,wi,we = state(f,e1,e2,t)
        reference,_,_ = state(F(1),F(0),F(0),t)
        assert observed == reference
    _,wi,we = state(f,e1,e2,F(1))
    print("f, E1, E2, internal work, external work:",f,e1,e2,wi,we)

# Forces divided by F_ref=1 N, so phi and the penalty below are dimensionless.
phi = F(2,3)
e1,e2 = 1-phi,-1+phi
assert (phi,e1,e2) == (F(2,3),F(1,3),F(-1,3))
for f in [F(-1),F(0),phi,F(1),F(3)]:
    penalty = f*f+(1-f)**2+(-1+f)**2
    assert penalty == F(2,3)+3*(f-phi)**2
observed,_,_ = state(phi,e1,e2,F(1))
assert observed == state(F(1),F(0),F(0),F(1))[0]
print("continuous minimum-norm forces:",phi,e1,e2)

# Actual lab controls use integer codes divided by10 N.
feasible = []
for code in range(-30,31):
    f = F(code,10)
    e1,e2 = 1-f,-1+f
    if abs(e1)<=3 and abs(e2)<=3:
        feasible.append((f*f+e1*e1+e2*e2,f,e1,e2))
assert min(feasible) == (F(67,100),F(7,10),F(3,10),F(-3,10))
print("lab-grid minimum: penalty and forces:",*min(feasible))
# A calibrated E1 with deterministic error bounds breaks the null direction.
assert m1*F(1,50)+F(3,100) == F(1,20)  # force error <=0.05 N
```

### Worked questions for physical inference

1. **Will selecting only particle1 in the lab identify f from its motion?** The selected net external force is f+E1=1 N in every table row. Boundary selection correctly changes the balance being inspected, but it does not separate two unknown contributions to the same net force.
2. **Will arbitrarily dense noiseless trajectories identify the three constants here?** They give the same two constant acceleration components; H retains its one-dimensional null space. A calibrated force, known intervention or another independent physical observation is needed.
3. **Why does exact work balance not validate the learned interaction potential?** Each candidate declares a different −fr and compensating external work. Its algebraic balance can be exact while its physical force assignment remains unverified by independent observations.

All coordinate projections, null-space and penalty calculations follow directly from this two-particle model. The analysis concerns missing force information, while the existing simulation continues to evaluate explicitly declared inputs.

## Six exercises with worked solutions

1. **Exercise: internal motion from rest.** Set equal1kg masses at−1m and1m, initial velocities zero, f=1N and E1=E2=0. At1s? **Solution:** accelerations are1 and−1m/s²; positions−1/2 and1/2m; velocities1 and−1m/s. P=0 and R=0, while K=1J.
2. **Exercise: energy exchange.** Find potential and mechanical energy for the same motion. **Solution:** initial V=2J and final V=1J. K rises from0 to1J. Therefore K+V remains2J, and internal work is1J.
3. **Exercise: add external force.** Set E1=1N in that example. **Solution:** a1=2 and a2=−1m/s². At1s, q1=0,q2=1/2m and P=1kg m/s. R=1/4m; its acceleration is1/2m/s².
4. **Exercise: external work.** Evaluate energy for exercise3. **Solution:** K=5/2J and V=1/2J, giving total3J instead of initial2J. E1 moves its particle by1m, so external work is1J, matching the increase.
5. **Exercise: select one particle.** Return to exercise1 and select particle1. **Solution:** selected momentum changes from0 to1kg m/s. The partner force is now external, supplying1N s. Total two-particle momentum is still0; there is no contradiction.
6. **Exercise: unequal masses.** Use m1=1kg,m2=2kg, the initial positions of exercise1 and f=1N. **Solution:** at1s, velocities1 and−1/2m/s give P=0. Positions−1/2 and3/4m give R=1/3m, equal to its initial value. K=3/4J and the arithmetic midpoint1/8m is not the center of mass.

## What a learned simulator should preserve

A network can predict the correct center trajectory while producing incorrect relative motion. Checking only total momentum will miss such errors. Conversely, a small kinetic residual does not identify a physically correct force law. Evaluate particle trajectories, the selected-system balance, center dynamics and the declared energy model together, with their assumptions recorded. Additional observations are needed to decide whether this constant-pair model fits an actual apparatus.

## Background and limits

The standard center-of-mass framework is presented in [MIT OpenCourseWare, Velocity and Acceleration of the Center of Mass](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/pages/week-5-momentum-and-impulse/17-6-velocity-and-acceleration-of-the-center-of-mass/). The constant-pair example and all numerical solutions here follow from the stated equations. Rigid-body rotation, variable mass and collision mechanics require further assumptions and are developed separately.
