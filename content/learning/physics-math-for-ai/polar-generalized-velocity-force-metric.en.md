## Coordinates describe a configuration, not a second physical system

A camera records a point at Cartesian position (x, y). A range-and-bearing sensor describes the same point by radius r and angle theta. These descriptions need not give the same numerical velocity components: a changing angle moves the point farther when the radius is larger. The physical velocity and kinetic energy must nevertheless agree after the coordinate map is applied correctly.

This lesson studies one instantaneous state of a planar point mass. It computes compatible Newton acceleration and compares a small configuration variation with its linear approximation. The variation parameter epsilon is dimensionless. Moving its control does not integrate a particle trajectory in physical time.

## Define the map and its local validity

Use a nonnegative radius and an angle measured counterclockwise from the positive x-axis:

$$
\boldsymbol x(r, \theta)=\begin{pmatrix}r\cos\theta\\r\sin\theta\end{pmatrix}, \qquad
J=\frac{\partial\boldsymbol x}{\partial(r, \theta)}
=\begin{pmatrix}\cos\theta&-r\sin\theta\\\sin\theta&r\cos\theta\end{pmatrix}.
$$

The radial column has units of meters per meter; the angular column describes meters of motion per radian. Its length is r. The analytic determinant is r, so the map has a locally invertible differential when r>0. This is a statement about a local chart, not global uniqueness of angle representation.

Adding 2*pi to theta returns the same point. A genuine one-to-one polar chart restricts the angle to a suitable branch. A branch cut away from the origin and rank failure at the origin are different phenomena. The lab deliberately permits repeated angle representations so this distinction remains visible.

## Push generalized rates into Cartesian velocity

Let qdot=(rdot, omega), where omega=thetadot. Differentiating the map gives

$$
\boldsymbol v=J\dot q=\dot r\, \boldsymbol e_r+r\omega\, \boldsymbol e_\theta,
\quad \boldsymbol e_r=(\cos\theta, \sin\theta),
\quad \boldsymbol e_\theta=(-\sin\theta, \cos\theta).
$$

The second generalized rate is an angular rate in radians per second, not the tangential speed in meters per second. Tangential speed is r*omega. For r=1 m, theta=0, rdot=1 m/s and omega=2 rad/s, the Cartesian velocity is (1, 2) m/s. Doubling radius while holding omega fixed doubles only that angular contribution.

A plot of the physical position and a vector-space plot of velocity therefore require different axis units. They should not be overlaid with one unexplained numerical scale.

## The kinetic metric remembers the geometry

For mass m>0,

$$
K=\frac{m}{2}\boldsymbol v^T\boldsymbol v
=\frac12\dot q^T G(q)\dot q, \qquad
G=J^T(mI)J=\begin{pmatrix}m&0\\0&mr^2\end{pmatrix}.
$$

Thus K=m*(rdot²+r²*omega²)/2. The angular diagonal entry carries a length-squared factor. Replacing G by m times the identity would incorrectly treat angular rate as linear speed.

For the example above with m=2 kg, K=5 J. This agrees with the direct Cartesian calculation, m*(1²+2²)/2. Polar metric entries have different dimensional roles because the coordinates have different units. A raw Euclidean norm or eigenvalue comparison of mixed radial/angular entries is not an invariant physical error measure without a stated scaling convention.

## Accelerations also differentiate the moving basis

The radial and tangential basis vectors rotate with theta. Differentiating velocity therefore gives

$$
\boldsymbol a=(\ddot r-r\omega^2)\boldsymbol e_r
+(r\alpha+2\dot r\omega)\boldsymbol e_\theta,
\qquad \alpha=\ddot\theta.
$$

The term J*qddot alone omits the curvature contribution. Its error can be substantial even at a perfectly regular chart point. These terms arise from differentiating the coordinate map; they are not new applied forces in the Cartesian laboratory description.

With m=2 kg and the example state, set the applied Cartesian force to zero. Physical acceleration must vanish. Yet the compatible generalized accelerations are rddot=4 m/s² and alpha=−4 rad/s². Their contribution (4, −4) m/s² cancels the curvature contribution (−4, 4) m/s². Nonzero generalized accelerations do not imply nonzero Cartesian acceleration.

## Solve the instantaneous Newton-compatible acceleration

Resolve the applied force into radial and tangential components Fr and Ft, both in newtons. At r>0, Newton's equation gives

$$
\ddot r=F_r/m+r\omega^2, \qquad
\alpha=\frac{F_t/m-2\dot r\omega}{r}.
$$

The lab reconstructs the Cartesian acceleration and compares it with F/m. A separate deliberately naive J*qddot value shows the omitted geometric contribution. This solves acceleration at the selected state. Holding all displayed rates fixed while sweeping angle is a set of instantaneous comparisons, not a dynamically integrated motion.

For F=(3, 4) N at the previous state, Fr=3 N and Ft=4 N. The generalized accelerations are (5.5, −2) in their respective units, while Cartesian acceleration is (1.5, 2) m/s².

## Generalized force is defined by work

A virtual variation delta q=(delta r, delta theta) produces delta x=J*delta q to first order. Requiring the same scalar work gives

$$
\delta W=\boldsymbol F^T\delta\boldsymbol x
=(J^T\boldsymbol F)^T\delta q, \qquad Q=J^T\boldsymbol F.
$$

Here Qr=Fr is in newtons, whereas Qtheta=r*Ft is an angular work-conjugate quantity in newton meters per radian, conventionally expressed as torque. It is not simply the tangential force component.

For r=1 and F=(3, 4), Q=(3, 4) in the respective radial and angular units. With r=2, the tangential force is still 4 N but Qtheta becomes 8 N m. The corresponding power identity is F dot v=Q dot qdot. At the first state both give 11 W. The transpose in J^T maps a work-conjugate force; it is not a rule to push a force as though it were a velocity vector.

## A virtual displacement is not an arbitrary finite step

For a fixed direction delta q, compare the exact configuration x(q+epsilon*delta q) with

$$
\boldsymbol x_{\rm lin}(\varepsilon)
=\boldsymbol x(q)+\varepsilon J(q)\delta q.
$$

Their difference is generally second order in small epsilon. For the selected constant Cartesian force, F dot the exact finite position change is the finite work along this configuration variation; epsilon times delta W is its first-order approximation. Equal first-order work does not imply exact finite-step equality.

If r+epsilon*delta r becomes negative, the finite polar probe has left the declared parameter domain. The lab returns an undefined exact point and work for that query. It keeps the regularity of the original base point separate. Valid probe curves stop at the radius-zero boundary, including that boundary even when it lies between regular samples.

## The origin is a coordinate failure, not a singular force

At r=0 the angular Jacobian column vanishes. The map differential has rank one and the kinetic metric is singular. The physical Cartesian plane still has two configuration degrees of freedom. An invalid chart has not removed a physical direction.

Cartesian acceleration F/m remains finite. Inverse polar accelerations are undefined and are displayed as such; dividing by a small artificial radius would invent a different model. Supplied parameter rates can still produce a velocity along the chosen radial column, but they are not a complete regular polar state description at the origin.

Energy and power identities along this singular parameterization can still have zero residual. These identities are therefore insufficient evidence of chart invertibility. Rank and domain assumptions must be checked independently.

## Numerical checks and research interpretation

Trigonometric geometry, matrix values, reconstructed accelerations and finite-probe errors are numerical evaluations. Analytic identities state what should hold; floating-point residuals show how closely the evaluations agree. The reference checks use separately differentiated geometry and Hessians, linear-system solves and high-precision local derivatives rather than only reusing the same formulas.

In learned mechanics, an encoder can preserve energy on a sampled curve while collapsing an unobserved coordinate direction. Checking a scalar invariant alone does not establish a valid state chart. Validate local rank, units, force duality, curvature terms and the actual domain of a finite perturbation.

## Exercises with explained solutions

1. **Rates are not Cartesian components.** Set r=2 m, theta=0, rdot=1 m/s and omega=2 rad/s. **Solution:** The radial and tangential velocities are (1, 0) and (0, 4) m/s, so v=(1, 4). Treating the angular rate itself as tangential speed would give the wrong second component.

2. **Force-free generalized acceleration.** Use m=2, r=1, theta=0, rdot=1, omega=2 and F=0. **Solution:** rddot=4 and alpha=−4. J*qddot=(4, −4) is canceled by curvature (−4, 4), yielding zero physical acceleration. K=5 J at this instant.

3. **Force, torque and power.** At that state change F to (3, 4) N. **Solution:** Qr=3 N, Qtheta=4 N m, and power is 3*1+4*2=11 W. The compatible generalized acceleration is (5.5, −2), which reconstructs F/m=(1.5, 2).

4. **A valid base and invalid probe.** Take r=0.5, delta r=−1 and epsilon=1. **Solution:** The regular base has r>0, but the finite target radius is −0.5 and is outside the polar domain. Exact finite-probe outputs are undefined. The valid variation reaches the origin at epsilon=0.5; that target has a singular chart.

5. **A zero residual at a bad chart.** Set r=0 and verify the power identity. Does it prove two valid local coordinates? **Solution:** No. The identity still follows from multiplying by J, but J has rank one. Cartesian acceleration remains finite while inverse polar acceleration is undefined. The physical plane retains two degrees of freedom.

6. **Global and local uniqueness.** Compare theta=0 and theta=2*pi at r=1. **Solution:** They represent the same point and the same differential map up to numerical roundoff. Each is locally regular, but the unrestricted angle representation is not globally one-to-one. A branch restriction addresses this repetition; it does not repair the origin's rank failure.

## Sources and further study

[MIT Dynamics and Control I, Lecture 1](https://ocw.mit.edu/courses/2-003j-dynamics-and-control-i-spring-2007/resources/lec01/) develops polar velocity and acceleration. [Lecture 14](https://ocw.mit.edu/courses/2-003j-dynamics-and-control-i-spring-2007/resources/lec14/) discusses virtual work and generalized forces. The numeric examples and finite-domain probe here are derived teaching experiments. Moving coordinate systems, nonholonomic constraints, global atlases and integrated generalized-coordinate dynamics remain further study.
