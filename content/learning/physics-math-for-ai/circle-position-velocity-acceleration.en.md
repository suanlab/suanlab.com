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

## Exercises with worked solutions

1. **Exercise:** At q=(1,0),v=(0,1),a=0 on a unit circle, which condition fails? **Solution:** The acceleration condition gives v squared+q dot a=1; position and tangency pass.
2. **Exercise:** What normal acceleration repairs that instant? **Solution:** a=(-1,0) gives the required inward component.
3. **Exercise:** Does q dot v=0 at the origin make the circle constraint valid? **Solution:** No. The position is outside every positive-radius circle centered there.
4. **Exercise:** Project v=(1,1) at q=(1,0). **Solution:** The tangent velocity is(0,1), losing one half joule of kinetic energy for unit mass.
5. **Exercise:** Why is radial projection unavailable at q=0? **Solution:** Every circle point is equally near; there is no uniquely selected radial direction.
6. **Exercise:** Does zero instantaneous work by an ideal normal force prove that a state-projection algorithm conserves energy? **Solution:** No. These are different operations, and the velocity example explicitly loses kinetic energy.

## Sources and further study

[MIT Underactuated Robotics on multibody dynamics](https://underactuated.mit.edu/multibody.html) gives background on bilateral position constraints and their velocity/acceleration derivatives. The circle formulas and projection-energy examples above are direct derivations. 
