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
