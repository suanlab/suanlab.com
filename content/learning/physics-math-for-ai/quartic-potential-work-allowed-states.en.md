## Work answers a question about change

A force can vary strongly with position. Multiplying its value at the destination by the total displacement will generally give the wrong work. Imagine moving a small bead through a shaped potential landscape: the slope changes as the bead moves, so each short displacement must be paired with the force acting there.

This lesson uses one constant positive mass in one spatial dimension. Its conservative potential is a quartic polynomial. We calculate endpoint work, all energetically allowed positions and both possible velocity signs. These are state-space statements. The diagrams do not solve a time trajectory, choose an initial condition or certify that every allowed point can be reached in finite time.

## Specify the potential and its units

Let a>0 and let b have either sign:

$$
V(x)=\frac{a x^4}{4}+\frac{b x^2}{2},\qquad
F(x)=-\frac{dV}{dx}=-a x^3-bx.
$$

If x is in meters and V in joules, a has units J/m⁴ and b has units J/m². The resulting force is in newtons. The positive quartic coefficient makes V grow at large absolute position. Positive b produces one well around the origin; negative b produces two wells separated by a local maximum at the origin.

The force points down the potential gradient. It need not point toward the origin everywhere: in a double well, a particle just to the right of the origin is initially pushed farther right toward the right-hand minimum.

## Derive endpoint work by integration

For motion from position A to position B, conservative work is

$$
W_{A\to B}=\int_{x_A}^{x_B}F(x)\,dx
=\left[-\frac{a x^4}{4}-\frac{b x^2}{2}\right]_{x_A}^{x_B}
=V(x_A)-V(x_B).
$$

Reversing the endpoints changes the sign. Splitting an integral at an intermediate position adds the two works; any repeated outward-and-return segment cancels. A different time schedule changes instantaneous power but not the work of this conservative force along the same spatial traversal.

Computing this integral does not establish that an unassisted particle with a particular energy can make that traversal. For example, the integral from a forbidden position is still a mathematically defined function of two endpoints. Treat work evaluation and dynamical admissibility as separate questions.

## Apply the work–kinetic energy theorem

For a trajectory driven only by this force, Newton's law multiplied by velocity gives

$$
\frac{d}{dt}\left(\frac12mv^2\right)=F(x)v,
\qquad
K+V=E.
$$

Consequently a state at position x requires

$$
K(x)=E-V(x)\ge0,\qquad
v=\pm\sqrt{\frac{2(E-V(x))}{m}}.
$$

The square root supplies speed magnitude. For positive available kinetic energy there are two velocity signs, rather than a single automatically chosen direction. At a boundary both signs meet at zero. Negative available kinetic energy is not a negative kinetic energy of a real moving particle: it means the requested state is forbidden and its speed is undefined.

## Solve the complete allowed position set

The condition V(x)≤E can be solved without sampling a picture. Put z=x², so z≥0:

$$
a z^2+2bz-4E\le0,\qquad D=b^2+4aE.
$$

If D<0, the positive-leading quadratic never reaches zero and the allowed set is empty. Otherwise its roots are

$$
z_- =\frac{-b-\sqrt D}{a},\qquad
z_+ =\frac{-b+\sqrt D}{a}.
$$

Intersect the interval between the roots with z≥0. If the intersection is empty, there are no allowed positions. If its lower endpoint is zero, positions form one interval from −√z+ to √z+. A positive lower endpoint instead yields two intervals, one on each side of the origin. Equality cases must be retained: an interval can collapse to a single point.

This algebraic construction finds all components. It avoids overlooking a narrow well or inventing a bridge across a forbidden gap because of a coarse plotting grid.

## Understand the double-well thresholds

For b<0, the stationary positions satisfy x=0 or x=±√(−b/a). The origin is unstable because V''(0)=b<0. The two nonzero points are minima, with

$$
V_{\min}=-\frac{b^2}{4a}.
$$

Below this minimum no states exist. At the minimum, only two stationary equilibrium positions remain. Between the minimum and zero, the allowed positions split into two separated wells. Above zero, a nonzero-speed trajectory can pass through the origin.

At E=0, the allowed position set is connected, but its interpretation requires additional care. The origin itself has zero force and zero speed, and therefore is an equilibrium state. Being at rest there does not imply stability against a perturbation.

## A connected state set is not a finite-time path

Near the origin in the negative-b, zero-energy case,

$$
v^2=\frac{-b}{m}x^2-\frac{a}{2m}x^4.
$$

Thus speed is proportional to |x| near zero. The travel-time integral contains a term proportional to ∫dx/|x|, which diverges. Nonstationary zero-energy branches approach the unstable origin asymptotically instead of crossing it in finite time. The stationary origin is a separate solution selected by its own initial data.

The allowed-position panel therefore says “state set,” not “reachable trajectory.” Even a connected energy-level picture cannot replace the equation of motion and initial conditions. This is an elementary example of a broader distinction between feasibility and reachability.

## Changing the energy zero changes no force or work

The offset control adds the same constant C to potential and total energy:

$$
V_C=V+C,\qquad E_C=E+C,\qquad
E_C-V_C=E-V.
$$

Potential differences, force, endpoint work and allowed positions stay unchanged. Shifting V alone while holding the numerical value of total energy fixed would describe a different physical energy level. The interface deliberately shifts both together so the experiment tests a change of reference.

Mass affects speed at an allowed position but does not change which positions satisfy V≤E at fixed energy. Doubling mass divides speed by √2, while the endpoint work remains the same.

## Read the three spatial diagrams

The first panel compares potential and the selected total energy in joules. The second shows the associated force in newtons. The third displays every allowed position interval, including zero-width points and an empty result. Horizontal axes are position in meters throughout; none is a time axis.

The query endpoint B supplies available kinetic energy, force and speed magnitude. An orange point appears only for an allowed query. Hollow circles mark interval boundaries. Exact rational endpoint quantities are distinguished from numerical radicals used for speed and boundaries. A thin plotted curve is an illustration of the polynomial, not the mechanism by which the intervals were certified.

## Research extension: solve the separatrix and measure the approach clock

The allowed interval at the barrier energy is connected, yet a particle does not cross its unstable origin in finite time. We can make that distinction quantitative by constructing an exact orbit. The following time-dependent analysis is standalone: the native quartic lab still displays potential, force and allowed states, rather than integrating trajectories.

### Choose the double well and its physical scales

Restrict the earlier family to a>0 and b<0, with constant m>0, no drive and no damping. Choose the potential zero at the origin:

$$
\beta=-b>0,\quad V(x)=\frac a4x^4-\frac\beta2x^2,\quad
A=\sqrt{\frac{2\beta}{a}},\quad \omega=\sqrt{\frac\beta m},\qquad E=0.
$$

A is a length and omega an inverse time. The zero-energy turning positions are ±A, while the unstable equilibrium is (x,v)=(0,0). The allowed position projection is [−A,A]. It includes states belonging to distinct motions.

### Construct a nonstationary exact orbit

The hyperbolic secant is sech z=1/cosh z. Consider the positive branch with its outer turning time t0:

$$
x(t)=A\operatorname{sech}(\omega(t-t_0)),\qquad
v(t)=-A\omega\operatorname{sech}(\omega(t-t_0))\tanh(\omega(t-t_0)).
$$

At t=t0, position is A and velocity zero. Differentiate once more, using (sech z)'=−sech z tanh z and tanh²z+sech²z=1:

$$
\ddot x=\omega^2x\left(1-2\operatorname{sech}^2(\omega(t-t_0))\right)
=\frac\beta m x-\frac a m x^3,\qquad m\ddot x=-V'(x).
$$

The equality uses A²=2beta/a and omega²=beta/m. It verifies the differential law, not just an energy curve. The kinetic and potential contributions also cancel identically:

$$
\frac m2v^2=\frac{\beta^2}{a}\operatorname{sech}^2z\tanh^2z,
\qquad
V(x)=-\frac{\beta^2}{a}\operatorname{sech}^2z\tanh^2z,
\quad z=\omega(t-t_0).
$$

For every finite t, cosh is finite and positive, hence x(t)>0. As t tends to either infinity, x and v approach zero. The full orbit leaves the saddle asymptotically in the distant past, turns at A and returns asymptotically; such an orbit is called homoclinic. Its sign-reversed copy stays on the negative side. Initial data exactly at the origin give the separate stationary solution. The polynomial Newton vector field is locally Lipschitz, so one cannot attach a departure after waiting at that equilibrium while retaining this same law.

Zero velocity at the outer turning point does not imply equilibrium: force there is −beta A, which is nonzero. This agrees with the native lab's distinction between an energy boundary and a stationary state.

### Compute an exact finite-resolution arrival time

On the returning branch t≥t0, ask when position first reaches a positive cutoff delta<A. Inverting sech gives:

$$
t_\delta-t_0=\frac1\omega\operatorname{arcosh}\frac A\delta,
\qquad 0<\delta<A,\qquad
\operatorname{arcosh}u=\log\left(u+\sqrt{u^2-1}\right).
$$

The argument A/delta is dimensionless. For small delta, the clock grows like log(2A/delta)/omega and diverges as delta approaches zero. Reaching a measurement threshold is therefore a finite event, while reaching the origin exactly is not. A simulator reporting “arrived” should declare that threshold.

For a=1 J/m⁴, beta=2 J/m², A=2 m and t0=0:

| delta (m) | approach time, m=1 kg (s) | approach time, m=4 kg (s) |
|---|---|---|
| 0.1 | 2.607989 | 5.215979 |
| 0.01 | 4.236601 | 8.473202 |
| 0.001 | 5.864779 | 11.729557 |

The mass change leaves the allowed positions and turning length fixed, but multiplies this clock by two. The table is not the time taken to cross the origin.

![Finite segments of two zero-energy phase-space branches remain separate from the stationary origin; the time to a positive position cutoff increases logarithmically as the cutoff decreases.](/assets/learning/classical-quartic-separatrix.en.svg)

The upper panel is a phase portrait with position in metres and velocity in m/s, using m=1 kg and only −2≤t≤2 s. Open endpoints mark the finite plotting window, not new physical turning points. Arrows show increasing time. The black origin is the stationary state. The lower panel uses a logarithmic cutoff axis and compares the two masses' exact clocks.

### A small energy margin can change long-time crossing behavior

Write epsilon=E−V(0), the energy above the barrier. This difference is unchanged by adding the same constant to E and V. Negative epsilon forbids the origin; zero gives the separatrix; positive epsilon permits passage with nonzero speed.

Choose a fixed window [−d,d] with 0<d²≤beta/a, and enter at −d with positive velocity consistent with epsilon>0. The crossing time is:

$$
T_{\mathrm{cross}}(\epsilon)
=\sqrt{\frac m2}\int_{-d}^{d}
\frac{dx}{\sqrt{\epsilon+\beta x^2/2-a x^4/4}},\qquad \epsilon>0.
$$

Inside the window the quartic term is at most beta x²/4, giving two simple denominator bounds. Taking reciprocal square roots reverses their order; integrating the comparison quadratics gives:

$$
\epsilon+\frac\beta4x^2
\le\epsilon+\frac\beta2x^2-\frac a4x^4
\le\epsilon+\frac\beta2x^2,\quad |x|\le d,
$$

$$
2\sqrt{\frac m\beta}\operatorname{arsinh}\left(d\sqrt{\frac\beta{2\epsilon}}\right)
\le T_{\mathrm{cross}}(\epsilon)\le
2\sqrt{\frac{2m}\beta}\operatorname{arsinh}\left(d\sqrt{\frac\beta{4\epsilon}}\right).
$$

Both sides grow logarithmically as epsilon tends to zero from above. The dimensionless combinations under arsinh have the required units because epsilon is energy. Thus arbitrarily small positive margins can permit crossing, but the required time grows without bound.

At the same initial position −d, the rightward velocities for epsilon=0 and a small positive epsilon approach each other continuously. Their different eventual crossing behavior is a long-time issue; it does not establish finite-time chaos in this integrable one-dimensional system. An energy tolerance must be compared with the barrier margin before using it to certify a side-of-barrier classification.

### Reproduce the orbit, clocks and comparison bounds

The code directly checks the closed orbit against force and energy. For positive margins it also evaluates a smooth change-of-variable quadrature at two resolutions and compares with the analytic bounds. Numerical agreement illustrates the result; the inequalities above supply its continuous-time proof.

```python
from math import sqrt, cosh, tanh, acosh, asinh, sinh

a, beta = 1.0, 2.0  # SI: V=a*x^4/4-beta*x^2/2
A = sqrt(2*beta/a)
for mass in [1.0,4.0]:
    omega = sqrt(beta/mass)
    for t in [-2.0,-1.0,0.0,1.0,2.0]:
        s = 1/cosh(omega*t)
        x, v = A*s, -A*omega*s*tanh(omega*t)
        acc = omega**2*x*(1-2*s*s)
        energy = mass*v*v/2+a*x**4/4-beta*x*x/2
        assert abs(energy) < 1e-12
        assert abs(mass*acc-(beta*x-a*x**3)) < 1e-12
        assert x > 0
    for delta in [0.1,0.01,0.001]:
        time = acosh(A/delta)/omega
        assert abs(A/cosh(omega*time)-delta) < 1e-12
        print("mass, cutoff, approach time:",mass,delta,round(time,6))

def simpson(f,end,n):
    h = end/n
    return h/3*(f(0)+f(end)
                +4*sum(f(i*h) for i in range(1,n,2))
                +2*sum(f(i*h) for i in range(2,n,2)))

# Positive barrier margins; m=1 kg and window [-d,d], d=1 m.
mass,d = 1.0,1.0
assert a*d*d <= beta
for margin in [0.1,0.001,0.000001]:
    lower = 2*sqrt(mass/beta)*asinh(d*sqrt(beta/(2*margin)))
    upper = 2*sqrt(2*mass/beta)*asinh(d*sqrt(beta/(4*margin)))
    u_max = asinh(d*sqrt(beta/(2*margin)))
    # x=sqrt(2*margin/beta)*sinh(u) removes the narrow central peak.
    def f(u):
        z = a*margin/beta**2 * sinh(u)**2*tanh(u)**2
        assert z < 1
        return 1/sqrt(1-z)
    coarse = 2*sqrt(mass/beta)*simpson(f,u_max,4096)
    fine = 2*sqrt(mass/beta)*simpson(f,u_max,8192)
    assert abs(fine-coarse) < 1e-8
    assert lower <= fine <= upper
    print("margin, lower, quadrature, upper:",margin,
          round(lower,6),round(fine,6),round(upper,6))
```

### Worked questions for learned dynamics

1. **What is the force at A=2 m for a=1,beta=2?** It is −4 N. The particle is momentarily at rest there but accelerates back toward the well. At x=v=0 the force is zero instead.
2. **If the cutoff decreases by a factor ten near the origin, how much extra time is needed for m=1?** Asymptotically log(10)/sqrt(2), about 1.62817 s. It is a logarithmic clock, not a constant-speed approach.
3. **Does a tiny conserved-energy residual prove that a predicted crossing is correct?** Compare its energy margin, initial data, differential equation and time horizon with the reference. Near a zero barrier margin, an absolute energy tolerance alone does not settle eventual crossing.

All orbit and comparison calculations are direct derivations for the declared double well. They add an explicit clock while preserving the existing distinction between allowed states and a time trajectory.

## Six exercises with worked solutions

1. **Exercise: two wells.** Set a=1,b=−2,E=−1/2 and m=1. Is x=1 allowed? **Solution:** V(1)=−3/4, so K=1/4 and speed=1/√2. The origin has V=0>E and is forbidden. Two allowed intervals remain.
2. **Exercise: all boundaries.** Find those intervals. **Solution:** z=2±√2. Hence positions lie in [−√(2+√2),−√(2−√2)] and [√(2−√2),√(2+√2)]. Do not connect across the central forbidden gap.
3. **Exercise: minimum energy.** Change E to−1. **Solution:** D=0 and z=2. Only x=±√2 remain. Their force and speed are zero. Query x=1 instead has K=−1/4, so it supplies no real speed.
4. **Exercise: endpoint work.** For a=1,b=−2, move from0 to1. **Solution:** W=0−(−3/4)=3/4J. The reverse work is−3/4J. This calculation alone does not assert that a particle at E=−1/2 can start at the forbidden origin.
5. **Exercise: offset and mass.** Add C=2J and change mass from1 to4kg in exercise1. **Solution:** displayed total energy becomes3/2J and displayed query potential5/4J. K stays1/4J; speed halves; the intervals and work stay unchanged.
6. **Exercise: connected separatrix.** Set E=0. Does [−2,2] guarantee crossing the origin? **Solution:** no. At the origin velocity and force vanish. The nonstationary branches require infinite time to approach it; initial data at the origin give the stationary solution.

## Connection to constrained learning and research

An energy penalty in a learned model can enforce E−V≥0 while leaving direction, timing and reachable orbit unspecified. Training on a feasible-state set alone cannot identify a vector field. Evaluate conservative work, force gradients and actual trajectory predictions separately. An exact symbolic identity verifies the declared polynomial calculation; empirical validation still requires data establishing that this potential describes the intended physical system.

## Background and scope

For the standard potential-gradient and work framework, see [MIT OCW, Potential Energy and Conservation of Energy](https://www.ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/mit8_01scs22_chapter14.pdf). The quartic family, allowed-set derivation and examples above are derived within this lesson. Higher-dimensional topology, tunneling and time integration are separate topics.
