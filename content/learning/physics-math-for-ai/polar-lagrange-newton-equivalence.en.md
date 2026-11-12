## One motion, two descriptions

A camera reports Cartesian position, while a range-and-bearing sensor reports distance and angle. Neither sensor changes the physical motion. Nevertheless, a Cartesian acceleration component and the second derivative of a polar coordinate can be very different numbers. Lagrange equations provide a systematic way to obtain the correct coordinate equations from energy.

This lesson moves beyond instantaneous coordinate conversion. We derive the differential equations, solve the corresponding central-force motion in Cartesian coordinates, and then examine the polar description along that actual trajectory. A coordinate singularity will appear in one experiment even though position, velocity and force remain finite. This is a limitation of the description rather than a failure of Newtonian motion.

## Recall what stationary action requires

Choose fixed endpoint times and positions. For a smooth path q and a variation η that vanishes at both endpoints, vary the action S=∫L(q,qdot)dt. The product rule and integration by parts give

$$
\delta S=\left[\frac{\partial L}{\partial\dot q_i}\eta_i\right]_0^T
-\int_0^T\left[\frac{d}{dt}\frac{\partial L}{\partial\dot q_i}-\frac{\partial L}{\partial q_i}\right]\eta_i\,dt.
$$

Sum the displayed expression over coordinates i. The boundary term vanishes for these admissible variations. If the first variation vanishes for every sufficiently regular interior variation, the fundamental lemma of variational calculus sets each bracket to zero. This yields the Euler–Lagrange equations. Stationarity does not by itself establish a minimum; the previous lesson supplied examples of flat directions and saddles.

For the general mechanics formulation, see Russ Tedrake's MIT [Multi-Body Dynamics](https://underactuated.mit.edu/multibody.html). The equations below are derived directly for our polar model, including its domain restriction and units.

## Specify the system before differentiating

Let m>0 and choose the central potential U=kr²/2 with k≥0. The force in Cartesian coordinates is −kx. In a regular polar chart, r>0 and

$$
\boldsymbol x=r\begin{pmatrix}\cos\theta\\\sin\theta\end{pmatrix},\qquad
L=\frac m2(\dot r^2+r^2\dot\theta^2)-\frac k2r^2.
$$

Radius is a length, angle is in radians, and L has units J. The coordinate domain excludes the origin. An angular branch can also require a local restriction; repeated angles describe the same position. Do not interpret a principal-angle jump across a branch cut as an impulsive physical motion.

A partial derivative of L holds the other coordinate and velocity arguments independent. A total time derivative follows their joint evolution. Confusing those two operations is a common source of missing inertial terms.

## Derive the radial equation

The radial momentum is ∂L/∂rdot=m rdot, so its total derivative is m rddot. The coordinate derivative is ∂L/∂r=m r thetadot²−kr. Subtracting gives

$$
m\ddot r-mr\dot\theta^2+kr=0,
\qquad \ddot r=r\dot\theta^2-\frac{k}{m}r.
$$

Every term in the first equation has force units N. The positive r thetadot² term in the coordinate equation does not mean the spring pushes outward. It accounts for how the radial coordinate changes while the direction basis rotates. We will reconstruct the physical acceleration to identify the actual force direction.

## Derive the angular equation

The Lagrangian has no explicit theta dependence, but its angular momentum coefficient depends on radius:

$$
\frac{\partial L}{\partial\dot\theta}=mr^2\dot\theta,\qquad
\frac{d}{dt}(mr^2\dot\theta)
=m(r^2\ddot\theta+2r\dot r\dot\theta).
$$

Thus, for r>0,

$$
\ddot\theta=-\frac{2\dot r\dot\theta}{r}.
$$

The product rule produces the term with factor two. Dropping it would make the angular momentum mr²thetadot change without an applied torque. The angular Euler–Lagrange equation has torque units N m, distinct from the radial equation's N. Here we verify this conservation identity directly; the full symmetry-to-conservation theorem is a later topic.

## Recover Newton acceleration

The basis vectors satisfy erdot=thetadot etheta and ethetadot=−thetadot er. Differentiating position twice gives

$$
\boldsymbol a=(\ddot r-r\dot\theta^2)\boldsymbol e_r
+(r\ddot\theta+2\dot r\dot\theta)\boldsymbol e_\theta.
$$

Substitute both Lagrange equations. The physical radial component is −kr/m and the tangential component is zero. Therefore a=−(k/m)x, precisely the Cartesian Newton equation. Coordinate acceleration rddot is not the projection a·er; the latter includes the basis-rotation correction.

The velocity Hessian of L is diagonal with entries m and mr². It is invertible for positive radius, allowing the coordinate equations to determine accelerations. At r=0 that regular-coordinate argument fails. This does not make the Cartesian mass m vanish.

## Use an actual analytic trajectory

Write the natural frequency as Ω=√(k/m). Given initial Cartesian position x₀ and velocity v₀, the solution is

$$
\boldsymbol x(t)=\boldsymbol x_0\cos(\Omega t)
+\frac{\boldsymbol v_0}{\Omega}\sin(\Omega t),\qquad
\boldsymbol v(t)=-\Omega\boldsymbol x_0\sin(\Omega t)+\boldsymbol v_0\cos(\Omega t).
$$

When k=0 the separate free branch is x=x₀+t v₀ and v=v₀. It avoids division by zero. These formulas are analytic solutions; evaluating sine and cosine on a computer still uses floating-point arithmetic. The model does not claim exact decimal energy conservation.

For each regular position, recover r=|x|, rdot=x·v/r and thetadot=(x×v)/r². The plotted angle, when shown, uses a principal branch. Its rate is recovered from physical vectors and is not computed by subtracting wrapped angles.

## Explain the controls and time axis

The lab sets Ω=πf/4 s⁻¹ with integer f from zero to four and k=mΩ². Changing mass at fixed f also changes stiffness; it retains the natural frequency. Initial radius is the radius code divided by two meters, and initial radial velocity is its code divided by two m/s. Initial directions use multiples of π/2.

The angular-rate control W gives initial thetadot=W Ωref/2, where Ωref=Ω for f>0 and Ωref=1 s⁻¹ in the free branch. This permits an exact circular-orbit relation W=±2 without asking a decimal input to equal π. Query time ranges from zero to two seconds. It selects a point on the same physical trajectory rather than changing a solver step size or perturbing a path with epsilon.

## Worked example: straight motion with changing polar rates

Choose m=1, k=0, r₀=1, initial angle zero, rdot₀=1 and thetadot₀=1. Cartesian position is (1,0) and velocity is (1,1). Newton acceleration is zero. Yet initially rddot=1 and thetaddot=−2. The two rotating-basis corrections exactly cancel those coordinate accelerations in the physical vector.

After one second x=(2,1), r=√5, rdot=3/√5 and thetadot=1/5. Cartesian velocity remains (1,1), kinetic energy remains 1 J and angular momentum remains 1 kg m²/s. A changing angular rate does not imply a torque when radius changes too.

## Worked example: circular motion

Choose m=1, f=2, r₀=1, rdot₀=0 and W=2. Then Ω=π/2, initial angular rate equals Ω, and the trajectory is a circle. Radius stays constant, so radial coordinate acceleration is zero. Physical radial acceleration is nevertheless −π²/4 m/s².

Kinetic and potential energies are each π²/8 J, total energy is π²/4 J, and angular momentum is π/2 kg m²/s. At one second the point is (0,1) and velocity is (−π/2,0). The constant radius does not make the motion force-free.

## Worked example: cross the origin without breaking motion

Keep m=1 and f=2, but set both initial rates to zero and start at (1,0). The path is x=(cos(πt/2),0). At t=1 it passes through the origin with velocity (−π/2,0). Force and acceleration are zero at that instant, and the total energy is π²/8 J.

Radius and inverse polar angle/rates cannot form a regular coordinate description at that point. The implementation returns an undefined polar state and splits polar-rate curves around it. It retains the finite Cartesian trajectory. Analytic radial crossing times are inserted even when they fall between the regular output samples, so a polar curve never bridges an undefined origin. Known cardinal phase identities identify represented query-time origins exactly; other event times are floating-point evaluations of analytic roots, with zero position supplied by the root identity. No artificial small positive radius is inserted. Near an origin the chart remains defined but can be poorly conditioned, which is a different issue.

## Research extension: a trajectory does not identify its Lagrangian

A free cart moving at constant velocity supplies position and time data, but it does not directly report the function whose variation generated that motion. Different candidate Lagrangians can fit every such trajectory, even when their velocity Hessians are positive. We will make the hypothesis class explicit before calling a fit physically identified.

### Construct a broader kinetic hypothesis class

Restrict the training example to the native free branch $f=0$, so the potential vanishes. In Cartesian coordinates consider:

$$
s=\boldsymbol v\cdot\boldsymbol v,\qquad
L_\lambda(\boldsymbol v)=\frac m2s+\frac\lambda4s^2,\qquad m>0,\quad\lambda\ge0.
$$

Here $m$ is in kg, $s$ in m²/s² and $\lambda$ in kg s²/m², so both terms in $L_\lambda$ have energy units. For nonzero $\lambda$ this is a nonquadratic candidate outside the native natural kinetic model. The extra coefficient is introduced in this supplementary Python experiment. It is not another native slider or a claim of equivalence for the nonzero-frequency harmonic branch.

### Positive curvature still leaves different models

Differentiate with respect to physical velocity:

$$
\boldsymbol p_\lambda=(m+\lambda s)\boldsymbol v,\qquad
H_\lambda=\frac{\partial^2L_\lambda}{\partial\boldsymbol v\,\partial\boldsymbol v}
=(m+\lambda s)I+2\lambda\boldsymbol v\boldsymbol v^T.
$$

For any nonzero vector $u$, $u^TH_\lambda u=(m+\lambda s)|u|^2+2\lambda(v\cdot u)^2>0$. At nonzero velocity the transverse and longitudinal eigenvalues are $m+\lambda s$ and $m+3\lambda s$; at zero velocity both equal $m$. Thus every candidate has a regular velocity-to-momentum map. This response matrix is velocity dependent when $\lambda>0$; it is not the constant Newtonian mass matrix.

The candidate has no position dependence, so its Euler–Lagrange equation and its regular polar pullback are:

$$
\mathcal E_x=H_\lambda\ddot{\boldsymbol x}=0,\qquad
\mathcal E_q=J^TH_\lambda(J\ddot q+\dot J\dot q)=0,\qquad
\frac{\partial^2\widetilde L_\lambda}{\partial\dot q\,\partial\dot q}=J^TH_\lambda J,\quad r>0.
$$

With $r>0$, invertibility of $J$ and $H_\lambda$ makes every free solution satisfy $\ddot x=0$. The polar equations are the same rotating-basis equations derived earlier. The pulled-back velocity Hessian is positive definite too. At the origin, the chart remains invalid; changing the kinetic candidate does not repair its rank failure.

### Compare observables and the complete fixed-endpoint optimum

Although the positions agree, the momentum and the time-independent energy need not agree:

$$
E_\lambda=\boldsymbol p_\lambda\cdot\boldsymbol v-L_\lambda
=\frac m2s+\frac{3\lambda}{4}s^2.
$$

A conserved candidate energy is not yet a calibrated energy measurement. Moreover, strict convexity and Jensen’s inequality give, for smooth fixed-endpoint paths over $T>0$:

$$
S_\lambda[\boldsymbol x]\ge
T L_\lambda\!\left(\frac{\boldsymbol x(T)-\boldsymbol x(0)}{T}\right),\qquad
\text{equality iff }\boldsymbol v(t)\text{ is constant a.e.}
$$

Thus the same straight path is the unique action minimizer for every candidate, not merely a shared differential-equation solution. Matching the path optimum or verifying conservation of each candidate's own energy still does not choose $\lambda$. The action values themselves differ.

### Use a calibrated force at a stated velocity

Keep the physical force convention fixed and impose $\boldsymbol F$ through the forced Euler–Lagrange equation. Its initial response is:

$$
H_\lambda\boldsymbol a=\boldsymbol F,\qquad
H_1\big|_{m=1,\,v=(1,1)}=\begin{pmatrix}5&2\\2&5\end{pmatrix},\qquad
\boldsymbol a\big|_{F=(1,0)}=\frac1{21}\begin{pmatrix}5\\-2\end{pmatrix}.
$$

The numerical arrays in the example use SI units: $H$ is in kg and acceleration in m/s². For $m=1$ kg, $v=(1,1)$ m/s and $F=(1,0)$ N:

| $\lambda$ / (kg s²/m²) | $E_\lambda$ / J | Initial $a_x$ / (m/s²) | Initial $a_y$ / (m/s²) |
|---|---|---|---|
| 0 | 1 | 1 | 0 |
| 1/2 | 5/2 | 3/8 | -1/8 |
| 1 | 4 | 5/21 | -2/21 |

![Identical free trajectories and different calibrated initial force responses](/assets/learning/analytical-lagrange-kinetic.en.svg)

For these candidates, $F_y=0$ holds $p_y$ constant. The speed-dependent coefficient in $p_y=(m+\lambda s)v_y$ couples the two velocity components, producing the nonzero y acceleration in the table.

These are initial accelerations at the same nonzero velocity, not integrated forced trajectories. A single probe at zero velocity would give $H_\lambda=mI$ for all candidates and would not identify the quartic term. Excitation matters as well as measurement accuracy.

### Distinguish equation symmetry from action quasi-symmetry

Adding $dF(x,t)/dt$ changes a Lagrangian by an expression affine in velocity, whose velocity Hessian is zero. Our nonzero Hessian change is therefore not a total-derivative gauge. A physical symmetry assumption can also discriminate candidates. Under a constant Galilean boost $u$ along x:

$$
L_0(\boldsymbol v+u\boldsymbol e_x)-L_0(\boldsymbol v)
=\frac d{dt}\left(mu x+\frac12mu^2t\right),\qquad
H_\lambda(u\boldsymbol e_x)-H_\lambda(0)
=\lambda u^2\begin{pmatrix}3&0\\0&1\end{pmatrix}.
$$

For $\lambda>0$ and $u\ne0$, the latter difference prevents the boost change from being a total derivative of a function of position and time. Free equations remain boost invariant, but these actions do not all have the usual Galilean quasi-invariance. A learner that enforces the natural quadratic class or this action principle is using an additional physical assumption. State that assumption alongside the trajectory data.

### Execute and test the inference

This standard-library cell reproduces the table and a boost-Hessian check.

```python
from fractions import Fraction as Q

def candidate(lam, v):
    m = Q(1)
    s = sum(x*x for x in v)
    p = [(m+lam*s)*x for x in v]
    H = [[(m+lam*s)*int(i == j)+2*lam*v[i]*v[j]
          for j in range(2)] for i in range(2)]
    energy = m*s/2+3*lam*s*s/4
    return p, H, energy

# SI: m=1 kg, v=(1,1) m/s, F=(1,0) N. Responses are initial accelerations.
for lam in map(Q, ['0', '1/2', '1']):
    p, H, energy = candidate(lam, [Q(1), Q(1)])
    det = H[0][0]*H[1][1]-H[0][1]*H[1][0]
    acceleration = [H[1][1]/det, -H[1][0]/det]
    assert H[0][0] > 0 and det > 0
    assert [sum(H[i][j]*acceleration[j] for j in range(2))
            for i in range(2)] == [1, 0]
    print(lam, energy, *acceleration)
    zero_H = candidate(lam, [Q(0), Q(0)])[1]
    boost_H = candidate(lam, [Q(1), Q(0)])[1]
    assert boost_H[0][0]-zero_H[0][0] == 3*lam
# All candidates produce free straight paths; a calibrated forced response
# and the usual Galilean action principle supply additional information.
```

**Research questions with answers**

- Do a positive velocity Hessian and matching free trajectories identify the natural kinetic energy? **Answer:** No. Every member of this broader family passes both tests.
- Is this only a boundary-term ambiguity? **Answer:** No. A total derivative cannot produce the velocity-dependent Hessian change.
- Does one force measurement at rest identify $\lambda$? **Answer:** No. Its instantaneous Hessian is $mI$ there; use informative nonzero-speed probes or an explicit physical model-class assumption.

## Exercises with explanations

1. Why does the angular equation contain a factor two? **Solution:** Differentiate r²thetadot using the product rule. The derivative of r² is 2r rdot, so the full coefficient is r²thetaddot+2r rdot thetadot.
2. In the free example, calculate angular momentum after one second. **Solution:** m(x vy−y vx)=1(2−1)=1 kg m²/s. Equivalently mr²thetadot=1·5·1/5=1. The angular rate alone is not conserved.
3. A circular orbit has rddot=0. Is its physical radial acceleration zero? **Solution:** No. It equals rddot−r thetadot²=−rΩ² and requires the inward central force.
4. At the origin crossing, should the program divide by a small chosen radius? **Solution:** No. That would invent a coordinate state. Keep Cartesian position and velocity and report the inverse polar state as undefined.
5. Double mass while retaining f and the same initial coordinate data. **Solution:** Stiffness doubles, the physical trajectory stays the same, and energy and angular momentum double. This experiment fixes frequency rather than stiffness.
6. Does deriving Euler–Lagrange equations prove that the trajectory minimizes action over every horizon? **Solution:** No. The equations give stationarity under the stated boundary variations. The previous oscillator example showed how curvature can become flat or have both signs.

## Research connection and verification scope

A learned dynamics model should agree across regular coordinate descriptions. Compare physical accelerations, energy and angular momentum rather than equating coordinate second derivatives. Retain coordinate-domain information in an AI state representation so a singular inverse does not become a fabricated finite state.

The implementation is checked against independent 60-digit Cartesian flow and radius derivatives. Its force and energy diagnostics describe an analytic solution evaluated numerically. The next unit uses Lagrange equations for a coupled system whose trajectory needs numerical integration, so it must also expose integration and refinement errors.
