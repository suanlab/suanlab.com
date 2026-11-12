## Being on a constraint is only the first requirement

A point can lie on a circle while its assigned velocity points straight out of it. A velocity can be tangent while the assigned acceleration fails to bend the path. Position, velocity and acceleration describe different levels of local consistency. Checking only one level can make an apparently constrained prediction physically incomplete.

This lesson will use a fixed bilateral circle centered at the origin with radius R>0. Position q has units meters, velocity v has units m/s and acceleration a has units m/s squared. A constant unit mass is used when illustrating kinetic energy. We inspect data at one instant and geometric correction operations; no complete external-force law or time integration is silently inferred.

## Differentiate the fixed-position constraint

The circle is described by

$$
g(q)=q\cdot q-R^2=0.
$$

For a sufficiently differentiable path on that fixed circle, differentiation gives

$$
\frac{g'}{2}=q\cdot v=0,\qquad
\frac{g''}{2}=v\cdot v+q\cdot a=0.
$$

The first expression has units square meters, the second square meters per second, and the third square meters per second squared. Their terms should not be combined into one unscaled loss merely because all are called residuals.

The formulas assume a fixed center and radius. Moving constraints add explicit time terms. Likewise, a string with one-sided tension restrictions adds force inequalities absent from this bilateral geometric example. State the actual constraint before applying a derivative rule or interpreting its force.

## Work through a circular-motion instant

Take R=1 meter, q=(1,0) meters and v=(0,1) m/s. The position and tangent-velocity conditions pass. If acceleration is zero, however, the second derivative condition gives one rather than zero. The proposed data would locally move away from the fixed circle.

Choosing a=(-1,0) m/s squared makes v dot v+q dot a equal to zero. This is the inward normal acceleration needed for unit-speed circular motion at that instant. A tangential acceleration could also be present while the required radial component remains fixed. Local geometry does not determine every tangential force or every future state.

The distinction matters when training a model to predict positions and derivatives: plausible positions are not enough if the associated velocities or accelerations are inconsistent with the same geometry.

## Explain the zero-position degeneracy

At q=0, the expression q dot v vanishes for every velocity. That does not make every velocity a valid tangent velocity on the circle. The position itself fails g(q)=0 because R is positive.

The gradient of g is 2q. It is nonzero on the valid circle, allowing a well-defined tangent line, but vanishes at the origin outside the manifold. A first-derivative check at an invalid point must not be promoted into a complete constraint certificate. The interface should report each algebraic residual and require position validity before calling the full state compatible.

The same location also makes radial projection ambiguous. Every circle point is equally close to the origin, so there is no unique direction to choose. Leaving the correction unavailable is more informative than arbitrarily selecting a point and disguising it as a determined solution.

## Project position with an explicitly geometric rule

For nonzero q, the nearest point on the circle in the Euclidean distance is

$$
q_P=R\frac{q}{\|q\|}.
$$

To see why, expand the squared distance from q to a candidate z on the circle. The only varying term is minus twice z dot q. Cauchy-Schwarz makes that dot product largest when z points along q, giving the formula above. At the origin all directions tie.

This is a correction rule, not a trajectory from q to qP under a physical force. A rendering may show the original and corrected positions without claiming that the connecting arrow is a measured or dynamically generated path. A larger correction may also signal that the unconstrained prediction was far outside the intended domain.

## Remove the radial velocity component

At the corrected point, project a candidate velocity onto the tangent line:

$$
v_P=v-q_P\frac{q_P\cdot v}{R^2}.
$$

Taking its dot product with qP gives zero, because qP dot qP=R squared. The removed component is radial and orthogonal to the retained tangent component. For constant unit mass, the kinetic-energy difference is

$$
K(v)-K(v_P)=\frac{(q_P\cdot v)^2}{2R^2}\ge0.
$$

For qP=(1,0),v=(1,1), the corrected velocity is(0,1). Kinetic energy falls from one joule to one half joule. The constraint correction passes the tangent test but is not an energy-preserving time step.

The example uses a Euclidean projection and a constant scalar mass. More general mechanical systems can require a mass-matrix metric. Do not silently transfer the same formula to arbitrary coordinates and inertial couplings.

## Correct the normal acceleration requirement

Using the corrected position and velocity, a closest Euclidean acceleration satisfying the local second-derivative condition is

$$
a_P=a-q_P\frac{q_P\cdot a+\|v_P\|^2}{R^2}.
$$

Substitution gives qP dot aP+vP dot vP=0. The operation changes the normal component while retaining the tangential component. For the unit circle with vP=(0,1) and candidate a=0 it produces(-1,0), recovering the earlier local circular-motion requirement.

This does not mean an arbitrary input acceleration has been evaluated from a valid external force law at the corrected position. It also does not guarantee that a subsequent numerical integrator will maintain the circle over finite time. Position, tangent velocity and acceleration consistency are necessary local conditions with a stated correction convention.

## Distinguish constraint work from state-projection energy

An ideal radial constraint force has the form lambda*qP. Its instantaneous power on tangent motion is lambda*qP dot vP=0. This force statement does not say a discrete state projection preserves energy. The projection above can delete a radial component from an inconsistent input velocity and thereby change kinetic energy.

Position correction can change other invariants too. With R=1, original q=(2,0),v=(0,1), the radial correction gives qP=(1,0) and leaves this already tangent velocity unchanged. Angular momentum about the origin for unit mass changes from two to one kg*m squared/s. Geometric validity was restored, but angular momentum was not automatically preserved.

These are different operations and assumptions. Avoid treating the word constraint as a promise that every quantity of interest survives any repair algorithm.

## State what numerical checks can establish

For rational input grids, the original position, velocity and acceleration residuals can be checked with exact integer numerators. The radial projection generally introduces a square root. Its algebraic identities can be proved analytically, while displayed projected values and residuals remain numerical approximations.

At q=0 the radial projection and downstream projected quantities should remain unavailable. At a valid input, a tiny rounded residual is implementation evidence, not a replacement for the derivation. A full trajectory would additionally require consistent initial data, a force law and an integration or analytic evolution argument.

## Connect the distinction to learned dynamics

A neural predictor might output points that look close to a surface while its derivatives fail tangent or acceleration constraints. A repair layer may lower geometric residuals while introducing energy or momentum changes. Evaluating only the repaired positions can hide those effects.

An informative benchmark reports the original residuals, the chosen correction, its domain of definition and changes in relevant invariants. If the intended task is physical time evolution, it also needs a dynamical model and temporal verification. Local algebraic consistency is useful, but it answers a more specific question.

## Research extension: Project velocities in the kinetic-energy metric

### Keep the native unit-mass assumption separate

The laboratory's Euclidean projection is appropriate for its declared unit isotropic mass. Now consider a different, generalized-coordinate kinetic model with a constant symmetric positive-definite mass matrix M. At a fixed valid configuration, let a full-row-rank constraint Jacobian J require tangent velocities Jw=0. This is a homogeneous fixed-configuration constraint; moving boundaries and position corrections are separate problems.

Choose the feasible velocity closest to v in the mass norm. The Lagrangian stationarity equation gives M(w−v)+J-transpose lambda=0. Solving the constraint for lambda yields

$$
\min_{Jw=0}\tfrac12(w-v)^TM(w-v),\qquad
P_M=I-M^{-1}J^T(JM^{-1}J^T)^{-1}J,\qquad w=P_Mv.
$$

Full row rank and positive-definite M make the middle matrix invertible. Strict convexity makes the feasible minimizer unique. Redundant constraints require rank reduction or a separately justified generalized inverse; an ordinary inverse must not be taken blindly.

### Prove the energy identity in the correct inner product

Direct substitution gives J P_M=0 and P_M squared=P_M. The correction lies in the mass-orthogonal complement of tangent velocities, so P_M-transpose M=M P_M. With T(v)=one half v-transpose Mv, the resulting Pythagorean identity is

$$
T(v)=T(P_Mv)+\tfrac12(Jv)^T(JM^{-1}J^T)^{-1}(Jv).
$$

The removed amount is nonnegative. This is a property of the local correction, not a claim that a full time integrator conserves energy. In fact the projection generally removes energy when the input has a normal component. Position projection can change other quantities, as the original angular-momentum example already shows.

### A Euclidean correction can increase this kinetic energy

Fix a unit circle at q=(3/5,4/5) meters and use J=q-transpose, equivalent to the circle's normal constraint up to a nonzero factor. Let M=diag(1,4) kilograms in these generalized coordinates, and v=(1,0) meters per second. The two feasible projections and their energies are

$$
P_Mv=(4/13,-3/13),\qquad P_Ev=(16/25,-12/25),
\qquad T(v)=1/2,\quad T(P_Mv)=2/13,\quad T(P_Ev)=416/625.
$$

The Euclidean projection raises kinetic energy from 0.5 to 0.6656 joules despite making the velocity tangent. It minimized the wrong distance for this different kinetic model. The mass projection instead removes 9/26 joules. There is no conflict with the native unit-mass result: setting M to the identity makes the two projections coincide.

![In velocity space, a tangent line intersects the initial kinetic-energy ellipse. The Euclidean projected velocity lies outside that energy level, whereas the mass-metric projection lies inside it.](/assets/learning/physical-mass-projection-en.svg)

| Requirement | Euclidean projection | Mass-metric projection |
|---|---|---|
| Tangent velocity at the fixed q | Yes | Yes |
| Minimum correction in the M norm | Generally no | Yes |
| Nonincrease of this quadratic kinetic energy | Not guaranteed | Follows from mass orthogonality |

**Worked question:** Is a zero geometric constraint residual enough to certify a physically consistent numerical correction? **Answer:** No. The metric, mass model, rank assumptions, and event or time-stepping interpretation matter too. The code checks tangency, mass orthogonality, exact energy decomposition and minimality against a family of other tangent velocities. The anisotropic example is standalone; the existing widget retains its unit isotropic mass and its explicit origin degeneracy.

```python
from fractions import Fraction as F
q=[F(3,5),F(4,5)];v=[F(1),F(0)];mass=[F(1),F(4)]
dot=lambda a,b:sum(x*y for x,y in zip(a,b))
T=lambda w:sum(m*x*x for m,x in zip(mass,w))/2
normal_inverse=[q[i]/mass[i] for i in range(2)]
den=dot(q,normal_inverse)
pm=[v[i]-normal_inverse[i]*dot(q,v)/den for i in range(2)]
pe=[v[i]-q[i]*dot(q,v)/dot(q,q) for i in range(2)]
assert pm==[F(4,13),F(-3,13)] and pe==[F(16,25),F(-12,25)]
assert dot(q,pm)==dot(q,pe)==0
removed=[v[i]-pm[i] for i in range(2)]
assert sum(mass[i]*pm[i]*removed[i] for i in range(2))==0
assert T(v)==T(pm)+T(removed) and T(removed)==dot(q,v)**2/(2*den)
assert T(pe)>T(v) and T(pm)==F(2,13) and T(pe)==F(416,625)
cost=lambda w:T([w[i]-v[i] for i in range(2)])
for t in [F(n,13) for n in range(-10,11)]:
    tangent=[4*t,-3*t]
    assert dot(q,tangent)==0 and cost(tangent)>=cost(pm)
    assert cost(tangent)==cost(pm)+T([tangent[i]-pm[i] for i in range(2)])
print('mass projection',pm,'Euclidean projection',pe)
print('energies initial, mass, Euclidean',T(v),T(pm),T(pe))
```

## Exercises with worked solutions

1. **Exercise:** At q=(1,0),v=(0,1),a=0 on a unit circle, which condition fails? **Solution:** The acceleration condition gives v squared+q dot a=1; position and tangency pass.
2. **Exercise:** What normal acceleration repairs that instant? **Solution:** a=(-1,0) gives the required inward component.
3. **Exercise:** Does q dot v=0 at the origin make the circle constraint valid? **Solution:** No. The position is outside every positive-radius circle centered there.
4. **Exercise:** Project v=(1,1) at q=(1,0). **Solution:** The tangent velocity is(0,1), losing one half joule of kinetic energy for unit mass.
5. **Exercise:** Why is radial projection unavailable at q=0? **Solution:** Every circle point is equally near; there is no uniquely selected radial direction.
6. **Exercise:** Does zero instantaneous work by an ideal normal force prove that a state-projection algorithm conserves energy? **Solution:** No. These are different operations, and the velocity example explicitly loses kinetic energy.

## Sources and further study

[MIT Underactuated Robotics on multibody dynamics](https://underactuated.mit.edu/multibody.html) gives background on bilateral position constraints and their velocity/acceleration derivatives. The circle formulas and projection-energy examples above are direct derivations. 
