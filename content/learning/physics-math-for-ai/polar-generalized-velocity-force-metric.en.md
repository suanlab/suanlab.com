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

## Research extension: potential curvature depends on how a probe moves

A positive Cartesian spring stiffness can produce a negative ordinary angular second derivative. That is a useful geometric warning, rather than evidence that changing coordinates made the spring unstable. Imagine moving a probe on a circle about the camera, while the spring center is elsewhere: the probe follows a curved physical path.

### Start with a displaced harmonic potential

Let the potential center be $(a,0)$, with mass $m$ in kg, stiffness $k$ in N/m, and lengths $a,r$ in m. This supplementary potential is evaluated by the Python cell below; the native controls retain their instantaneous geometry and applied-force interpretation.

$$
V(x,y)=\frac{k}{2}\big[(x-a)^2+y^2\big],\qquad
V(r,\theta)=\frac{k}{2}(r^2+a^2-2ar\cos\theta),\quad m,k,a,r>0.
$$

Ordinary differentiation in the polar coordinates gives:

$$
\partial_iV=\begin{pmatrix}k(r-a\cos\theta)\\kar\sin\theta\end{pmatrix},\qquad
\partial_i\partial_jV=
\begin{pmatrix}k&ka\sin\theta\\ka\sin\theta&kar\cos\theta\end{pmatrix}.
$$

At $r=a$ and $\theta=\pi$, the angular entry is $-ka^2$. This derivative correctly describes potential change along the constant-radius circle. It does not describe a straight Cartesian displacement or establish equilibrium. In fact the physical force there is nonzero. A real ring constraint would define a different mechanical system.

### Remove the curvature of the coordinate probe

The kinetic metric determines the connection that accounts for the turning basis. Its nonzero polar coefficients are:

$$
G=m\begin{pmatrix}1&0\\0&r^2\end{pmatrix},\qquad
\Gamma^r_{\theta\theta}=-r,\qquad
\Gamma^\theta_{r\theta}=\Gamma^\theta_{\theta r}=\frac1r.
$$

These are the Levi–Civita coefficients. For example, $\Gamma^r_{\theta\theta}=-(2m)^{-1}\partial_r(mr^2)=-r$ and $\Gamma^\theta_{r\theta}=(2mr^2)^{-1}\partial_r(mr^2)=1/r$.

The covariant Hessian subtracts the potential gradient contracted with this connection. Sum over the repeated upper/lower index $\ell$:

$$
(\nabla^2V)_{ij}=\partial_i\partial_jV-\Gamma^\ell_{ij}\partial_\ell V
=\begin{pmatrix}k&0\\0&kr^2\end{pmatrix}
=J^T(kI)J=\frac{k}{m}G.
$$

For example, the mixed entry is $ka\sin\theta-(kar\sin\theta)/r=0$. The angular entry is $kar\cos\theta+r k(r-a\cos\theta)=kr^2$. This agrees with pulling the Cartesian Hessian $kI$ back through $J$. The correction vanishes at a critical point because the gradient vanishes there. Away from a critical point, an ordinary Hessian is not a coordinate tensor.

### Compare directional curvatures without inventing modes

Divide the pure angular quadratic coefficient by its kinetic coefficient to compare quantities with units of s$^{-2}$:

$$
c_{\theta,\rm ordinary}=\frac{\partial_\theta^2 V}{G_{\theta\theta}}
=\frac{ka\cos\theta}{mr},\qquad
c_{\theta,\rm covariant}=\frac{(\nabla^2V)_{\theta\theta}}{G_{\theta\theta}}=\frac{k}{m}.
$$

![Ordinary angular curvature and covariant curvature for a displaced spring](/assets/learning/analytical-polar-hessian.en.svg)

| $\theta$ | Ordinary angular curvature / s$^{-2}$ | Covariant curvature / s$^{-2}$ |
|---|---|---|
| $0$ | 1 | 1 |
| $\pi/2$ | 0 | 1 |
| $\pi$ | -1 | 1 |

The table uses $m=k=a=r=1$ in the stated SI units. Its ordinary column is a directional Rayleigh quotient, not the full spectrum of the coupled Hessian. At $\theta=\pi/2$ the mixed ordinary entry is nonzero. Even a covariant curvature value alone is not a general stability verdict along a moving trajectory; the reference motion and full perturbation equation matter.

For this specific Cartesian spring the complete perturbation equation is available directly:

$$
\ddot{\boldsymbol x}=-\frac{k}{m}(\boldsymbol x-(a,0)),\qquad
\delta\ddot{\boldsymbol x}=-\frac{k}{m}\delta\boldsymbol x,\qquad
(r_*,\theta_*)=(a,0),\quad \omega^2=\frac{k}{m}.
$$

At the regular equilibrium $(a,0)$, both Hessians coincide and the two small-oscillation frequencies are $\sqrt{k/m}$. At $r=0$ the polar inverse remains invalid, even though the Cartesian potential is smooth. For AI energy models, compare derivatives with an explicit metric and domain, and state whether the tested point is an equilibrium.

### Execute and interpret

The standard-library cell reproduces the table and the cancelled mixed term.

```python
from math import cos, sin, pi

# SI values: m=1 kg, k=1 N/m, a=r=1 m.
for theta in [0, pi/2, pi]:
    r = a = m = k = 1.0
    gradient = [k*(r-a*cos(theta)), k*a*r*sin(theta)]
    ordinary = [[k, k*a*sin(theta)],
                [k*a*sin(theta), k*a*r*cos(theta)]]
    covariant = [[ordinary[0][0], ordinary[0][1]-gradient[1]/r],
                 [ordinary[1][0]-gradient[1]/r,
                  ordinary[1][1]+r*gradient[0]]]
    assert abs(covariant[0][1]) < 1e-14
    assert abs(covariant[1][1]-k*r*r) < 1e-14
    print(round(theta/pi, 6), round(ordinary[1][1]/(m*r*r), 6),
          round(covariant[1][1]/(m*r*r), 6))
# The second column is a directional curvature, not a stability verdict.
# At theta=pi the physical point is accelerating toward x=a.
```

**Research questions with answers**

- Is the negative angular derivative a calculation error? **Answer:** No. It describes a curved constant-radius probe; treating it as unconstrained equilibrium stability would be the error.
- When does an ordinary Hessian transform by congruence alone? **Answer:** At a critical point, or under a linear coordinate change; otherwise the gradient contributes additional chain-rule terms.
- Does adding a small radius repair the origin? **Answer:** No. Use a valid Cartesian chart or another regular description.

## Exercises with explained solutions

1. **Rates are not Cartesian components.** Set r=2 m, theta=0, rdot=1 m/s and omega=2 rad/s. **Solution:** The radial and tangential velocities are (1, 0) and (0, 4) m/s, so v=(1, 4). Treating the angular rate itself as tangential speed would give the wrong second component.

2. **Force-free generalized acceleration.** Use m=2, r=1, theta=0, rdot=1, omega=2 and F=0. **Solution:** rddot=4 and alpha=−4. J*qddot=(4, −4) is canceled by curvature (−4, 4), yielding zero physical acceleration. K=5 J at this instant.

3. **Force, torque and power.** At that state change F to (3, 4) N. **Solution:** Qr=3 N, Qtheta=4 N m, and power is 3*1+4*2=11 W. The compatible generalized acceleration is (5.5, −2), which reconstructs F/m=(1.5, 2).

4. **A valid base and invalid probe.** Take r=0.5, delta r=−1 and epsilon=1. **Solution:** The regular base has r>0, but the finite target radius is −0.5 and is outside the polar domain. Exact finite-probe outputs are undefined. The valid variation reaches the origin at epsilon=0.5; that target has a singular chart.

5. **A zero residual at a bad chart.** Set r=0 and verify the power identity. Does it prove two valid local coordinates? **Solution:** No. The identity still follows from multiplying by J, but J has rank one. Cartesian acceleration remains finite while inverse polar acceleration is undefined. The physical plane retains two degrees of freedom.

6. **Global and local uniqueness.** Compare theta=0 and theta=2*pi at r=1. **Solution:** They represent the same point and the same differential map up to numerical roundoff. Each is locally regular, but the unrestricted angle representation is not globally one-to-one. A branch restriction addresses this repetition; it does not repair the origin's rank failure.

## Sources and further study

[MIT Dynamics and Control I, Lecture 1](https://ocw.mit.edu/courses/2-003j-dynamics-and-control-i-spring-2007/resources/lec01/) develops polar velocity and acceleration. [Lecture 14](https://ocw.mit.edu/courses/2-003j-dynamics-and-control-i-spring-2007/resources/lec14/) discusses virtual work and generalized forces. The numeric examples and finite-domain probe here are derived teaching experiments. Moving coordinate systems, nonholonomic constraints, global atlases and integrated generalized-coordinate dynamics remain further study.
