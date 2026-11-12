## Zero torque can still produce changing body components

Release an asymmetric object into free rotation. Its angular momentum stays fixed in the laboratory, but axes painted on the object move. Measurements expressed along those moving axes can therefore change even when the physical vector is conserved. This is the central distinction in torque-free rigid-body motion.

The previous lesson evaluated an instantaneous velocity field and its inertia tensor. Here we solve an initial-value problem: both angular velocity and orientation evolve in actual time. We keep the center of mass fixed in an inertial frame, apply no external torque and omit contact, gravity gradients, friction and deformation. The result is a freely rotating rigid body, not an animation with a prescribed constant drawing angle.

## Choose realizable principal moments

Let the constant body-frame tensor be I=diag(I₁,I₂,I₃). Instead of allowing unrelated positive numbers, construct it from positive mass second moments A,B,C:

$$
I_1=B+C,\qquad I_2=C+A,\qquad I_3=A+B.
$$

These parameters have units kg m². For example A can be interpreted as a sum of mx² in principal coordinates. The construction gives I₁+I₂−I₃=2C>0 and the other two cyclic inequalities. Every control combination is therefore realizable by a positive three-dimensional mass distribution. A planar physical distribution can give equality in a triangle inequality; our strictly positive family deliberately excludes that boundary without claiming it is impossible in general.

The initial body components ω₀ are independent inputs. An initial rotation about laboratory z specifies how the principal axes start in space. The center-of-mass position does not translate in this experiment. A prescribed off-center pivot and its support forces belong to a different model.

## Derive the frame-transport term

For any vector a, differentiation in the inertial and rotating body frames is related by

$$
\left(\frac{da}{dt}\right)_{\rm lab}
=R\left(\dot a_{\rm body}+\omega_{\rm body}\times a_{\rm body}\right),
$$

where R maps body components to laboratory components. Body angular momentum is Lbody=Iωbody. Zero inertial torque gives

$$
I\dot\omega+\omega\times(I\omega)=0.
$$

The cross term comes from transporting the basis. Dropping it would incorrectly make every body angular-velocity component constant. A constant inertia tensor in body coordinates does not make that coordinate system inertial.

## Write Euler's equations component by component

In the principal basis,

$$
I_1\dot\omega_1=(I_2-I_3)\omega_2\omega_3,
$$

$$
I_2\dot\omega_2=(I_3-I_1)\omega_3\omega_1,\qquad
I_3\dot\omega_3=(I_1-I_2)\omega_1\omega_2.
$$

The products couple components. A pure principal-axis spin is a special exact solution: its other two components remain zero. An arbitrary combination generally evolves. For a sphere all three moments coincide, the coupling terms vanish and body angular velocity stays constant.

Euler's dynamical equations are distinct from Euler angles, which are one possible orientation parametrization. Solving the three component equations alone does not tell us where the body axes point in the laboratory. Orientation must be advanced consistently as well.

## Transport the actual orientation

For the body-to-laboratory matrix,

$$
\dot R=R[\omega_{\rm body}]_\times,\qquad R^\mathsf TR=\operatorname{Id},\quad\det R=1.
$$

The skew matrix acts by [ω]×v=ω×v. Our implementation uses a scalar-first Hamilton quaternion q=(q₀,qv) to carry the same orientation. With multiplication

$$
(a,u)\otimes(b,v)=(ab-u\cdot v,\;av+bu+u\times v),
$$

its evolution is

$$
\dot q=\frac12q\otimes(0,\omega_{\rm body}).
$$

The multiplication order and the body-to-laboratory convention matter. Swapping the order while retaining body components describes a different transport equation. The unit quaternions q and −q represent the same physical rotation. This representation avoids choosing an Euler-angle chart near a coordinate singularity, but it still requires the unit-norm constraint and an explicit convention.

## Prove the energy and momentum invariants

The rotational energy and angular-momentum norm are

$$
K=\frac12\omega^\mathsf TI\omega,\qquad |L|^2=|I\omega|^2.
$$

Their derivatives vanish because a cross product is perpendicular to each factor:

$$
\dot K=-\omega\cdot[\omega\times(I\omega)]=0,
$$

$$
\frac12\frac d{dt}|L|^2=-(I\omega)\cdot[\omega\times(I\omega)]=0.
$$

The stronger frame-sensitive check is Llab(t)=R(t)Iω(t)=Llab(0). Body components Iω can move while this inertial vector remains fixed. Constant energy and constant norm alone would not guarantee the correct orientation or phase, so the calculation also compares full position-of-axes matrices and angular-velocity components with independent references.

## Understand the symmetric analytic limit

When I₁=I₂=Iperp, ω₃ is constant. Set ν=(I₃−Iperp)ω₃/Iperp. Then

$$
\omega_1(t)=\omega_{10}\cos\nu t-\omega_{20}\sin\nu t,
$$

$$
\omega_2(t)=\omega_{10}\sin\nu t+\omega_{20}\cos\nu t.
$$

This is precession of the body components. Its sign depends on whether the symmetry-axis moment is larger or smaller than the transverse moment. It is not automatically the same angular rate as a laboratory Euler-angle precession. Orientation still follows its transport equation.

The spherical limit ν=0 and principal-axis spins provide additional exact checks. Zero initial angular velocity gives a stationary orientation, including when the initial orientation is not the identity.

## Derive local intermediate-axis instability

Assume distinct moments I₁<I₂<I₃ and perturb the exact spin ω=(0,Ω,0). Keeping only first-order terms gives

$$
\delta\dot\omega_1=\frac{I_2-I_3}{I_1}\Omega\,\delta\omega_3,
\qquad
\delta\dot\omega_3=\frac{I_1-I_2}{I_3}\Omega\,\delta\omega_1.
$$

Hence the squared growth rate is

$$
\lambda^2=\Omega^2\frac{(I_3-I_2)(I_2-I_1)}{I_1I_3}>0.
$$

The corresponding products for the extreme axes are negative and produce oscillatory linearized perturbations. Repeated moments remove the strict three-distinct-axis classification. An exactly unperturbed intermediate-axis spin remains an exact solution; instability concerns nearby states.

This is a local linear result, not an exponential formula for the entire nonlinear trajectory. Conservation laws bound the allowed motion, and a finite perturbation can leave the linear regime. A large visible reorientation need not be numerical failure, but its timing and invariants still require verification.

## Know what the numerical solver controls

The experiment integrates the three body components and four raw quaternion components. It compares one RK4 step with two half steps, estimates error from their difference divided by15, and accepts a Richardson-corrected state when the scaled error is small enough. Absolute and relative tolerances control the local numerical estimate; they are not a promise that every global error equals those numbers.

The raw quaternion is not projected back to unit length during integration. For display, a normalized copy produces an orthogonal rotation matrix, while raw norm drift is reported separately. A normalized-looking picture cannot hide the original norm error. Likewise a tiny energy residual does not by itself validate phase.

Independent DOP853 references integrate body momentum transport and a rotation matrix directly, without using this quaternion or RK4 implementation. Analytic symmetric, spherical and principal-axis cases provide further checks. Initial rational K and |L|² are exact baselines; later states and residuals are numerical.

## Read the visualizations in their stated frames

Body angular-velocity components use rad/s. Laboratory angular-momentum components use kg m²/s. Energy uses J and squared angular momentum uses kg² m⁴/s². These belong on separate labeled panels. A projected triad shows orientation, not a spatial orbit of the center of mass.

All time histories span0 to10s and include the selected query time as an actual integration stop. The orientation at that time is the computed solution, not a nearby precomputed drawing frame. Curves compare the numerical invariants with their exact initial baselines. Unit-spin stability labels describe the linearized principal-axis problem; they do not classify every arbitrary current state globally.

## Research extension: conserved quantities do not certify the rotation clock

### Connect local instability to one exact nonlinear trajectory

A model can reproduce energy and angular-momentum magnitude while predicting the wrong body components at a particular time. To see why, use a body with fixed inertial center of mass, no external torque and principal tensor I=diag(3,4,5) kg m². This tensor is realizable by the native second-moment controls A=3,B=2,C=1. Euler's equations become

$$
\dot\omega=f(\omega)=\left(-\frac{\omega_2\omega_3}{3},\frac{\omega_3\omega_1}{2},-\frac{\omega_1\omega_2}{5}\right).
$$

For any Ω>0 in rad/s, consider the following body-frame angular velocity:

$$
\lambda=\frac{\Omega}{\sqrt{15}},\qquad
\omega_1=\sqrt{\frac23}\,\Omega\operatorname{sech}(\lambda t),\quad
\omega_2=\Omega\tanh(\lambda t),\quad
\omega_3=\sqrt{\frac25}\,\Omega\operatorname{sech}(\lambda t).
$$

Here sech u=1/cosh u and λ has units s⁻¹. This is a special nonlinear solution, rather than a formula for every initial condition. Differentiate it using

$$
\frac{d}{du}\operatorname{sech}u=-\operatorname{sech}u\tanh u,\quad
\frac{d}{du}\tanh u=\operatorname{sech}^2u,\qquad
\lambda\sqrt{\frac23}=\frac{\Omega}{3}\sqrt{\frac25},\quad
\lambda\Omega=\frac{\Omega^2}{2}\sqrt{\frac23}\sqrt{\frac25}.
$$

The first component's derivative equals −ω₂ω₃/3 by the first coefficient identity. The second equals ω₃ω₁/2 by the second identity. The remaining equality follows from λ√(2/5)=Ω√(2/3)/5. All three Euler equations are satisfied, not just their linearization.

As t→−∞ the body components approach (0,−Ω,0); as t→∞ they approach (0,Ω,0). The trajectory connects two opposite intermediate-axis spins. This is a separatrix branch: it lies at the boundary between neighboring types of motion in the invariant-level geometry. The local rate λ agrees with the intermediate-axis growth rate, while tanh and sech keep the complete nonlinear motion bounded. Extending an exponential approximation through the whole turnover would miss this saturation.

### Derive the invariant level before using it

Using sech²u+tanh²u=1 gives

$$
K=\frac12(3\omega_1^2+4\omega_2^2+5\omega_3^2)=2\Omega^2,\qquad
|L|^2=9\omega_1^2+16\omega_2^2+25\omega_3^2=16\Omega^2.
$$

These numerical coefficients use the declared SI inertia; K is measured in J and |L|² in kg² m⁴/s². For example Ω=1 gives K=2 J and |L|²=16 kg² m⁴/s². Eliminating the common level gives

$$
|L|^2-2I_2K=-3\omega_1^2+5\omega_3^2=0\qquad(I_2=4).
$$

The positive first and third components of our branch obey ω₃=√(3/5)ω₁. The invariant condition also admits other branches and the limiting principal-axis spins; it does not choose a unique clock or initial state.

At t=0 our solution starts at (√(2/3)Ω,0,√(2/5)Ω). For Ω=1 these irrational components cannot be represented exactly by the native 0.1 rad/s input grid. The closed-form solution below is therefore a **standalone Python example**. It does not claim to extend the experiment's control range.

For comparison, native input `[3,2,1,0,10,0,0,100]` is the exact unperturbed intermediate-axis spin at t=10 s. Its body components stay (0,1,0), although its orientation rotates. Native input `[3,2,1,1,10,0,0,100]` is a different, finitely perturbed trajectory with K₀=403/200 J and |L₀|²=1609/100. It is not the analytic separatrix above. Matching a moment tensor does not make different initial conditions interchangeable.

### A wrong clock with the correct initial state and invariants

Let the true solution be ω(t), and define a candidate prediction by running the same curve at speed c=1.1:

$$
\widetilde\omega(t)=\omega(ct),\quad\widetilde\omega(0)=\omega(0),\qquad
\frac{d\widetilde\omega}{dt}-f(\widetilde\omega)=(c-1)f(\widetilde\omega).
$$

The initial body components agree exactly. Every energy and angular-momentum-norm value agrees as well. Nevertheless the candidate does not solve Euler's equations at the declared physical time whenever f is nonzero. At t=√15 s with Ω=1, the maximum component error is approximately 0.039780 rad/s and the maximum Euler derivative residual is approximately 0.010114 rad/s².

| Time t (s) | Correct ω₂ (rad/s) | Wrong-clock ω₂ (rad/s) | Maximum component error (rad/s) |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| √15 | 0.761594 | 0.800499 | 0.039780 |
| 2√15 | 0.964028 | 0.975743 | 0.038280 |

![Two clocks with the same initial body velocity and invariant values](/assets/learning/classical-rotation-clock.en.svg)

The upper panel compares the second body component. The lower panel reports the maximum absolute error over all three components, so it can change which component attains the maximum. The horizontal coordinate λt is dimensionless and Ω=1 throughout. The orange point marks λt=1, the table's t=√15 s case. Both curves have the same constant K and |L|². Small invariant residuals alone cannot validate timing.

This is directly relevant to physics-informed learning: invariant penalties can constrain a trajectory to an allowed set without enforcing how it moves through that set. Time-indexed observations, differential-equation residuals and an independent phase reference add information that these penalties omit. Derivative residuals estimated by finite differences also have discretization error; the example uses analytic derivatives to isolate the issue.

### Execute the analytic checks separately from orientation

This standard-library cell tests 483 combinations of Ω and time, directly compares analytic derivatives with the Euler vector field, checks both invariants and rejects the wrong clock. Floating-point tolerances are numerical checks rather than a rigorous interval ODE certificate. The printed last rows reproduce the table.

```python
from math import cosh, sqrt, tanh


def separatrix(t, omega=1.0):
    """Body components and their analytic time derivative; I=(3,4,5)."""
    rate = omega / sqrt(15)
    u = rate * t
    sech = 1 / cosh(u)
    v = tanh(u)
    w = [sqrt(2 / 3) * omega * sech, omega * v,
         sqrt(2 / 5) * omega * sech]
    derivative = [-rate * w[0] * v, omega * rate * sech ** 2,
                  -rate * w[2] * v]
    return w, derivative


def euler_rhs(w):
    x, y, z = w
    return [-y * z / 3, z * x / 2, -x * y / 5]


def invariants(w):
    return (sum(i * x * x / 2 for i, x in zip([3, 4, 5], w)),
            sum((i * x) ** 2 for i, x in zip([3, 4, 5], w)))


def separatrix_checks():
    count = 0
    for omega in [.5, 1, 2]:
        for j in range(-80, 81):
            t = j * sqrt(15) / (20 * omega)
            w, derivative = separatrix(t, omega)
            rhs = euler_rhs(w)
            assert max(abs(a - b) for a, b in zip(derivative, rhs)) < 2e-14
            k, l2 = invariants(w)
            assert abs(k - 2 * omega ** 2) < 2e-14
            assert abs(l2 - 16 * omega ** 2) < 2e-13
            count += 1
    t = sqrt(15)
    w, _ = separatrix(t)
    wrong, wrong_derivative = separatrix(1.1 * t)
    error = max(abs(a - b) for a, b in zip(w, wrong))
    assert .038 < error < .040
    residual = max(abs(1.1 * a - b)
                   for a, b in zip(wrong_derivative, euler_rhs(wrong)))
    assert residual > .009
    assert separatrix(0)[0] == separatrix(1.1 * 0)[0]
    assert abs(invariants(wrong)[0] - 2) < 1e-14
    assert abs(invariants(wrong)[1] - 16) < 1e-13
    print(f'{count} analytic Euler/invariant samples passed')
    print('Wrong clock at t=sqrt(15): max angular-velocity error, ODE residual:',
          error, residual)


separatrix_checks()
for t in [0, sqrt(15), 2 * sqrt(15)]:
    w, _ = separatrix(t)
    wrong, _ = separatrix(1.1 * t)
    print(t, w[1], wrong[1], max(abs(a-b) for a,b in zip(w,wrong)))
```

Only the **body angular velocity** is supplied by this formula. To obtain laboratory orientation or the fixed laboratory angular-momentum direction, one must also specify the initial orientation and solve its transport equation. Conserving |L|² does not by itself show that the laboratory vector L is correct. The lesson's independent rotation-matrix references and quaternion diagnostics remain necessary.

### Research questions with worked answers

1. **Can a model match the initial state and both scalar invariants yet predict the wrong time?** Yes. ω(1.1t) is an explicit counterexample. Its derivative residual is 0.1f at the predicted state, except where the vector field vanishes.
2. **Why does an exactly intermediate-axis spin stay there?** It is an equilibrium of the body-component equations. Instability concerns neighboring states. The separatrix approaches that equilibrium only at infinite time, so it does not contradict uniqueness for a finite-time initial-value problem.
3. **Does checking this analytic branch certify the full orientation solver?** No. Body angular velocity and attitude transport are different parts of the coupled problem. The analytic branch is one reference for body dynamics, not a replacement for orientation, frame-convention and general initial-condition checks.

The transport equations and local stability framework are covered in [MIT16.07 Lecture28](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/5e1d8699338146e5127080b880b906d6_MIT16_07F09_Lec28.pdf). The explicit branch and wrong-clock construction here are verified by substitution under the assumptions above.

## Worked examples and exercises with solutions

1. **A sphere.** Set A=B=C=1 and ω₀=(1,1,1), with identity initial orientation. **Solution:** all moments are2, K=3 and |L|²=12. Body angular velocity stays(1,1,1), while the body rotates about that direction with angular speed√3. A fixed body vector generally changes its laboratory components.

2. **A symmetric body.** Set A=B=1,C=2 and ω₀=(1,1/2,1). **Solution:** I=(3,3,2), ν=−1/3, K=23/8 and |L|²=61/4. At t=1 the first two components are cos(1/3)+(1/2)sin(1/3) and −sin(1/3)+(1/2)cos(1/3); the third remains1. Laboratory angular momentum remains fixed.

3. **A pure principal-axis spin.** Set A=3,B=2,C=1, giving I=(3,4,5), with ω₀=(0,0,1) and initial laboratory z orientationπ/2. **Solution:** body components stay unchanged and R(t)=Rz(π/2+t). This simple case checks the quaternion multiplication order and sign of physical rotation.

4. **An unstable axis.** Use A=3,B=2,C=1 for I=(3,4,5) and spin about axis2 with Ω=1. **Solution:** λ²=1/15 for a small perturbation. Exact data(0,1,0) stays on that axis, while a nearby input such as(0.1,1,0) evolves away. Do not use exp(λt) as an exact nonlinear long-time answer.

5. **Move the coordinate orientation.** Change only the initial orientation angle with the same principal moments and body ω₀. **Solution:** body-component histories, K and |L|² are unchanged. The initial and subsequent laboratory vectors rotate consistently. Leaving the initial laboratory momentum reference unrotated would produce a false conservation error.

6. **A stationary pose.** Use any allowed A,B,C, set ω₀=0 and choose a nonzero initial angle. **Solution:** K=|L|²=0 and the orientation remains that chosen pose. It need not become the identity matrix. Quaternion q and−q still give the same pose; a zero quaternion gives no valid orientation.

## Research connection and sources

A learned world model can preserve an energy scalar while rotating the body incorrectly, or interpret body-coordinate variation as external torque. Evaluating inertia realizability, frame transport, vector momentum, orientation phase and numerical drift separately makes these failures observable. The intermediate-axis case is useful precisely because nearby initial states can diverge while the physical invariants remain conserved.

For the underlying transport framework, see [MIT16.07 Lecture28 on Euler's equations](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/5e1d8699338146e5127080b880b906d6_MIT16_07F09_Lec28.pdf) and [Lecture29 on three-dimensional free rotation](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/f5b25c0af8ec4f76776221c148ea15b3_MIT16_07F09_Lec29.pdf). The control family, quaternion convention, adaptive solver and independent validation above specify this experiment.
