## Why describe a trajectory with momentum?

A free particle moves along a straight Cartesian path with constant velocity. In polar coordinates its radial speed and angular rate can both change. The previous lesson explained those changes using Lagrange equations. We now replace the velocity variables by quantities obtained by differentiating the Lagrangian: the canonical momenta.

This gives a concrete reason to track which variables a derivative holds fixed. The same kinetic energy can be written using angular velocity or angular momentum, but its derivative with respect to radius depends on which of those quantities remains independent. Hamilton equations use canonical momentum as an independent phase-space coordinate.

## Define the regular coordinate domain

Consider a planar point mass m>0 in the central potential U=kr²/2, with k≥0. Use a local polar chart with radius r>0 and angle theta measured from the positive x-axis. The Lagrangian is

$$
L(r,\theta,\dot r,\dot\theta)
=\frac m2(\dot r^2+r^2\dot\theta^2)-\frac k2r^2.
$$

Radius has units m, radial rate m/s and angular rate rad/s. A local angular branch is needed for a one-to-one coordinate description. A principal angle can jump at a branch cut even while physical position changes smoothly. Repeated angle labels and failure of the polar inverse at the origin are separate issues.

Our formulas below apply at regular positive radius. They do not extend a polar chart through r=0 by assigning an arbitrary angle or a small replacement radius.

## Differentiate with respect to the velocity arguments

Define the conjugate momenta by partial derivatives of L:

$$
p_r=\frac{\partial L}{\partial\dot r}=m\dot r,
\qquad
p_\theta=\frac{\partial L}{\partial\dot\theta}=mr^2\dot\theta.
$$

The radial momentum has units kg m/s, while angular momentum has units kg m²/s. The two components are work-conjugate to different coordinate changes. Their numerical equality at a particular point would not make them the same physical quantity.

In this natural central-force model, ptheta equals the physical out-of-plane angular momentum m(x vy−y vx). Radial momentum is the projection of Cartesian mechanical momentum onto the outward radial basis. That projection can change as the basis rotates, even when the Cartesian momentum vector is constant.

## Invert the velocity map

At fixed r>0, recover

$$
\dot r=\frac{p_r}{m},\qquad
\dot\theta=\frac{p_\theta}{mr^2}.
$$

The velocity Hessian is diag(m,mr²), with determinant m²r². It is positive and invertible in this model's regular domain. The diagonal entries have different units: kg and kg m². Its determinant has units kg²m². Do not interpret mixed-unit matrix eigenvalues as an invariant physical size.

This positive quadratic example has a unique inverse over each velocity space at a fixed regular position. More general mechanical Legendre maps require an appropriate nonsingularity condition locally; the positive-definite Hessian used here is a sufficient condition. Global angular uniqueness is still a separate coordinate question.

## Perform the Legendre transformation

Define H=pr rdot+ptheta thetadot−L, then eliminate both velocities using the inverse map:

$$
H(r,\theta,p_r,p_\theta)
=\frac{p_r^2}{2m}+\frac{p_\theta^2}{2mr^2}+\frac k2r^2.
$$

Each term has energy units J. This H is a function of position and canonical momentum, rather than an expression that leaves velocity as an independent argument. For this autonomous natural Lagrangian, it equals K+U, the physical mechanical energy.

The Legendre construction determines the function and its independent variables. Identifying it with mechanical energy uses properties of the model. The next lesson supplies an explicit time-dependent gauge representation in which the Hamiltonian differs from the conserved mechanical energy.

## Derive the four first-order equations

Hamilton equations are qdot=∂H/∂p and pdot=−∂H/∂q. Applying them gives

$$
\dot r=\frac{p_r}{m},\qquad
\dot\theta=\frac{p_\theta}{mr^2},
$$

$$
\dot p_r=\frac{p_\theta^2}{mr^3}-kr,\qquad
\dot p_\theta=0.
$$

The first two derivatives have units m/s and rad/s. The last two have force and torque units N and N m. These equations describe the same physical system as the earlier second-order Lagrange equations.

To check this directly, differentiate pr=m rdot and substitute ptheta=mr²thetadot. The radial equation becomes rddot=r thetadot²−kr/m. Differentiating ptheta gives r²thetaddot+2r rdot thetadot=0. The factor two has not disappeared; it is encoded in the momentum-to-velocity relation.

## Hold canonical momentum fixed in a partial derivative

When calculating ∂H/∂r, hold pr and ptheta fixed. The angular kinetic term then has derivative −ptheta²/(mr³). Therefore its contribution to prdot is positive.

If one instead writes angular kinetic energy as mr²thetadot²/2 and holds thetadot fixed, its radius derivative has the opposite sign. That is a derivative of an expression using different independent variables. Hamilton equations require the velocity elimination to happen first. This is a variable-choice issue, not a disagreement between Newton and Hamilton mechanics.

## Work through free Cartesian motion

Set m=1, k=0 and initially x=(1,0), v=(1,1). Initially r=1, rdot=1, thetadot=1, so pr=1 kg m/s, ptheta=1 kg m²/s and H=1 J. Hamilton equations give prdot=1 N and pthetadot=0.

After one second the physical position is (2,1). Now r=√5, pr=3/√5 and thetadot=1/5, while ptheta remains 1. Radial and angular kinetic energies are 9/10 J and 1/10 J, summing to the same H=1 J.

The nonzero radial momentum rate does not imply a nonzero Cartesian force. Radial momentum uses a rotating basis. The earlier acceleration formula includes its matching geometric correction, leaving the Cartesian acceleration zero.

## Work through a circular orbit

Choose m=1, natural frequency Ω=π/2, k=π²/4, radius one, zero initial radial rate and initial angular rate Ω. Then pr=0 and ptheta=π/2. The two terms in prdot cancel:

$$
\frac{p_\theta^2}{mr^3}=kr=\frac{\pi^2}{4}.
$$

The radius stays constant, angular velocity stays π/2 and H=π²/4 J. Angular kinetic energy and potential energy are each π²/8 J. The physical centripetal acceleration is nonzero even though both pr and prdot vanish.

Reversing the initial angular rate reverses ptheta and the direction of traversal. The Hamiltonian depends on ptheta², so the energy and radial force balance are unchanged. Sign information in momentum still matters for the actual motion.

## Understand the angular-energy bound and origin crossing

At finite H with nonzero ptheta, the nonnegative energy terms imply

$$
H\geq\frac{p_\theta^2}{2mr^2},\qquad
r\geq\frac{|p_\theta|}{\sqrt{2mH}}>0.
$$

This is a lower bound from the angular kinetic contribution, not necessarily the exact turning radius. Its assumptions include m>0, k≥0 and nonzero ptheta, which also ensures H>0.

If ptheta=0, a radial trajectory can pass through the Cartesian origin. For x=(cos(πt/2),0), the crossing at t=1 has finite velocity (−π/2,0) and energy π²/8 J. The polar expression containing ptheta²/r² is then outside its coordinate domain; replacing 0/0 with zero would silently invent a canonical chart state.

The model retains physical energy and Cartesian angular momentum, returns the polar canonical state as undefined and splits all canonical curves at the crossing. Analytic crossings between output samples are inserted as well. A radial momentum limit can reverse sign because the outward basis changes sides; drawing a line through that gap would suggest a physical impulse that is absent from Cartesian motion.

## Read phase diagrams with their units

A radial phase plot has radius in meters on one axis and radial momentum in kg m/s on the other. It is not a position-space orbit. Plot angular momentum and angular rate on axes with their own units rather than combining them into one norm.

The implementation reuses the independently verified Cartesian analytic flow and computes new canonical quantities along it. It does not run a new Hamilton ODE integrator. Trigonometric evaluations, inverse-map residuals and Legendre discrepancies are floating-point calculations. Sampled residual maxima describe the displayed samples, not all real times.

The initial controls retain Ω=πf/4 and k=mΩ². At fixed f, changing mass changes stiffness as well, preserving the physical frequency. The initial angular-rate code and query time have the meanings established in the preceding polar-flow unit.

## Exercises with explanations

1. Find the free example's momenta after one second. **Solution:** pr=m rdot=3/√5 kg m/s, while ptheta=m(x vy−y vx)=1 kg m²/s. Their different units explain why their numerical values should not be equated.
2. Evaluate ∂H/∂r initially in that example, holding canonical momenta fixed. **Solution:** It is −ptheta²/(mr³)=−1 N, hence prdot=1 N. Holding angular velocity fixed would calculate a different partial derivative.
3. On the circular orbit, why does prdot vanish? **Solution:** The angular kinetic derivative supplies π²/4 N and the potential derivative supplies −π²/4 N. Their cancellation keeps the radius constant while physical acceleration is centripetal.
4. Double mass at fixed f and the same initial coordinate data. **Solution:** The physical trajectory is unchanged, stiffness and both momenta double, and H doubles. The Hessian determinant m²r² increases by a factor of four.
5. At the radial origin crossing, can H be assigned through the expression ptheta²/r²? **Solution:** No. The coordinate expression is undefined there even though Cartesian energy is finite. Report the canonical chart state as undefined and separate the curve segments.
6. Does the angular-energy bound give the exact closest approach for every orbit? **Solution:** No. It drops the nonnegative radial kinetic and potential terms. It proves an exclusion radius for nonzero angular momentum; finding an exact turning point requires the complete energy equation.

## Research connection and further reading

When learning dynamics from velocities, the momentum map must match the chosen Lagrangian and coordinates. A learned function using velocities as independent arguments cannot simply be differentiated as though its arguments were canonical momenta. Test inverse maps, units and physical trajectories separately.

David Tong's [The Hamiltonian Formalism](https://davidtong.org/teaching/classical-dynamics/dynhtml/S4), especially the Legendre-transform and Hamilton-equation sections, provides the general framework. The polar formulas and worked examples here are derived directly and checked against symbolic elimination and independent high-precision trajectory derivatives. Later topics develop Poisson brackets, symplectic geometry and Noether's theorem in their own right.
