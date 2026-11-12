## Four point coordinates can describe only two configuration freedoms

A planar double pendulum has two point masses connected by two rigid, massless links to a fixed pivot. Recording the x and y positions of both masses gives four numbers. They are not four independent configuration coordinates: the two fixed lengths impose two independent constraints.

Two angles provide a compact description. This lesson connects that description to physical point velocities, kinetic energy and applied work. It compares absolute and relative angle conventions without changing the physical state. All calculations concern an instantaneous configuration and compatible rates, or a dimensionless configuration variation. They do not integrate a swinging double pendulum in time or compute its reaction forces.

## Declare the geometry before borrowing a formula

Let m1, m2, l1, l2 be positive. Measure both absolute angles a, b from the downward vertical:

$$
\boldsymbol p_1=l_1\begin{pmatrix}\sin a\\-\cos a\end{pmatrix}, \qquad
\boldsymbol p_2=\boldsymbol p_1+l_2\begin{pmatrix}\sin b\\-\cos b\end{pmatrix}.
$$

The second angle is measured from the vertical, not from the first link. A formula derived with a relative second angle must be transformed before use. The local angle descriptions are regular for the stated positive lengths, although adding full revolutions repeats the same configuration globally.

The four-component ambient position is z=(p1x, p1y, p2x, p2y). The constraints are h1=|p1|²−l1² and h2=|p2−p1|²−l2². The actual configuration map satisfies both. No tolerance-based deletion of a constraint is needed at aligned links.

## Differentiate the configuration map

Writing q=(a, b), the kinematic Jacobian is

$$
J=\begin{pmatrix}
l_1\cos a&0\\l_1\sin a&0\\
l_1\cos a&l_2\cos b\\l_1\sin a&l_2\sin b
\end{pmatrix}, \qquad \dot z=J\dot q.
$$

The first angular rate moves both masses, whereas the second moves only the second mass relative to the first. The Cartesian tip velocity is the sum of two link contributions. A stationary tip need not imply two zero angular rates.

For unit lengths, a=b=0 and absolute rates (1, −1) rad/s, the first mass moves at (1, 0) m/s while the second has zero instantaneous velocity. The opposite contributions cancel at the tip. This is an instantaneous statement; it does not assert that the tip stays fixed during the later physical motion.

## The constraint tangent space explains compatible velocities

Let C=partial h/partial z. Differentiating h(z(q))=0 gives

$$
C J=0, \qquad C\dot z=0.
$$

The two rows of C are independent for positive link lengths: the second constraint has a nonzero contribution involving the second mass, while the first concerns only the first mass. Thus ambient dimension four minus constraint rank two leaves two configuration degrees of freedom. The two columns of J span the compatible velocity space.

The matrix identity is an analytic statement. Its numerical evaluation, the length residuals and C times velocity are displayed as residual checks. A tiny floating-point residual is not itself the proof of constraint independence.

## Pull the Cartesian kinetic energy into angle coordinates

Define W=diag(m1, m1, m2, m2). Direct point kinetic energy gives

$$
K=\frac12\dot z^T W\dot z
=\frac12\dot q^T M_{\rm abs}(q)\dot q, \qquad M_{\rm abs}=J^T WJ,
$$

$$
M_{\rm abs}=\begin{pmatrix}
(m_1+m_2)l_1^2&m_2l_1l_2\cos(a-b)\\
m_2l_1l_2\cos(a-b)&m_2l_2^2
\end{pmatrix}.
$$

The cross term comes from two link contributions entering one point velocity. It is not an extra mass or an optional correction. At aligned unit links and unit masses, M_abs=[[2, 1], [1, 1]]. With rates (1, −1), its diagonal contribution is 1.5 J and its coupling contribution is −1 J, leaving K=0.5 J. The same value follows directly from the moving first mass and stationary second mass.

## Positive energy does not require every expanded term to be positive

For a nonzero angular-rate vector xi,

$$
\xi^TM_{\rm abs}\xi=(J\xi)^TW(J\xi)>0.
$$

Positive masses make W positive definite, and positive lengths give J independent columns. Equivalently,

$$
\det M_{\rm abs}=m_2l_1^2l_2^2\left[m_1+m_2\sin^2(a-b)\right]>0.
$$

A negative coupling contribution therefore does not imply negative total kinetic energy. Aligned links do not make this mass matrix singular when m1>0. Zero mass or zero length would change the assumptions and potentially the physical model; those inputs are rejected rather than silently preserving the same degree count.

The formula for positive definiteness is analytic. Trigonometric matrix entries, determinants and eigenvalues in the calculator are numerical evaluations.

## Change to relative angles without changing the physical motion

Use u1=a and u2=b−a. Then

$$
q=A u, \quad A=\begin{pmatrix}1&0\\1&1\end{pmatrix}, \qquad
\dot q=A\dot u, \quad M_{\rm rel}=A^TM_{\rm abs}A.
$$

For the aligned unit example, relative rates are (1, −2), not (1, −1), and M_rel=[[5, 2], [2, 1]]. The quadratic energy remains 0.5 J. Keeping the same two numerical rates while switching angle conventions would instead change the physical state.

Angular representations may be wrapped for a display, but derivatives use unwrapped differences. If a=pi and b=−pi, the displayed relative representation can be −2*pi; a rate difference of −4 rad/s is not reduced modulo 2*pi. A periodic position coordinate does not make angular velocity periodic in that way.

## Matrix eigenvalues are not coordinate-invariant energies

With unit masses and lengths, a=0, b=pi/2 gives M_abs=diag(2, 1), while M_rel=[[3, 1], [1, 1]]. Their eigenvalues are (1, 2) and (2−sqrt(2), 2+sqrt(2)). Both describe the same physical kinetic form after the rates transform correctly.

A coordinate transformation changes the basis and the coordinate Euclidean norm. It need not preserve matrix eigenvalues. It does preserve the scalar energy of a fixed physical velocity. Reporting an improved metric condition number without stating the coordinate scaling and physical error measure can therefore be misleading in learned mechanics.

## Pull applied forces back through virtual work

Use gravity g=10 m/s² and an optional constant Cartesian tip force (Fx, Fy). The ambient applied-force vector is

$$
f=(0, -m_1g, F_x, F_y-m_2g)^T, \qquad
Q_{\rm abs}=J^Tf, \qquad Q_{\rm rel}=A^TQ_{\rm abs}.
$$

Gravity contributes [−(m1+m2)*g*l1*sin(a), −m2*g*l2*sin(b)] in the absolute convention. These work-conjugate quantities multiply angle variations or angular rates. They are not automatically the motor torques at relative joints.

The power identity is f dot zdot=Q_abs dot qdot=Q_rel dot udot. At the aligned unit state with rates (1, −1) and tip force (3, 0) N, Q_abs=(3, 3) and Q_rel=(6, 3), yet power is zero because the tip is instantaneously still. Nonzero generalized-force components can cancel in their contraction with rates.

## A virtual tangent step is not an exact finite configuration

At fixed time, delta z=J*delta q is tangent to the constraints. For a finite dimensionless epsilon, compare z+epsilon*J*delta q with the exact mapped configuration z(q+epsilon*delta q). Only the latter automatically satisfies the finite length constraints.

For aligned unit links and delta q=(1, −1), the tangent approximation at epsilon=0.5 has h1=h2=0.25 m². Its first-order constraint variation is zero, but the second-order terms are not. The exact mapped probe keeps both lengths, up to numerical roundoff.

Virtual work f dot delta z is also first-order. The selected applied forces are constant in ambient coordinates, so their exact finite work along the configuration probe is f dot the actual finite change. With gravity and no first-order vertical movement, virtual work can be zero while finite work is nonzero. Epsilon is not physical time, and energy need not be conserved along this prescribed configuration variation.

## What the model has and has not solved

The lab supplies geometry, compatible instantaneous velocities, a mass metric, potential energy and applied work. Ideal constraint reactions are orthogonal to admissible virtual displacements, but their magnitudes have not been computed. Nor has a future physical trajectory been selected by stepping epsilon.

In particular, M(q)*qddot=Q alone is not generally the full nonlinear double-pendulum equation. Velocity-dependent geometric terms and the chosen force convention matter. The later Lagrange-equation lesson derives those terms. For AI applications, a valid geometric decoder, a positive kinetic metric and a correct force pullback are essential checks before claiming a learned time-evolution model.

## Research extension: normal frequencies use both mass and stiffness

The metric eigenvalues changed when we switched from absolute to relative angles. A measured oscillation frequency must not change with that bookkeeping choice. The missing ingredient is the potential Hessian, expressed in the same coordinates as the kinetic metric.

### Linearize at a stated equilibrium

Use positive masses, lengths and gravity, no applied tip force, and the downward equilibrium $q_*=(a,b)=(0,0)$. Angles are dimensionless. Evaluate the existing mass metric there:

$$
M_*=\begin{pmatrix}(m_1+m_2)l_1^2&m_2l_1l_2\\m_2l_1l_2&m_2l_2^2\end{pmatrix},\qquad
V=-(m_1+m_2)gl_1\cos a-m_2gl_2\cos b.
$$

Expand the potential to second order and keep first-order equations in the small displacement $\xi$. The reference velocity is zero, so the nonlinear velocity terms contribute no first-order term:

$$
K_*=\nabla_q^2V(0)=\begin{pmatrix}(m_1+m_2)gl_1&0\\0&m_2gl_2\end{pmatrix},\qquad
M_*\ddot\xi+K_*\xi=0,\qquad K_*v=\omega^2M_*v.
$$

Here $M_*$ has units of kg m² and $K_*$ of J, so $\omega^2$ has units of s$^{-2}$. Positive definiteness of both matrices gives positive squared frequencies. This conclusion uses the equilibrium, positive parameters and conservative model; it is not a rule for arbitrary driven or damped systems.

### Transform both quadratic forms

Relative displacement uses the same angle convention as the earlier geometry:

$$
\xi=A\eta,\quad A=\begin{pmatrix}1&0\\1&1\end{pmatrix},\qquad
M_{\rm rel}=A^TM_*A,\quad K_{\rm rel}=A^TK_*A,\quad v_{\rm rel}=A^{-1}v.
$$

Both energy and the spectral equation are preserved. In particular:

$$
M_{\rm rel}^{-1}K_{\rm rel}=A^{-1}(M_*^{-1}K_*)A,\qquad
\det(K_{\rm rel}-\lambda M_{\rm rel})=\det(A)^2\det(K_*-\lambda M_*).
$$

An invertible $A$ multiplies the characteristic determinant by a nonzero constant, leaving its roots unchanged. Equivalently, the two dynamical matrices are similar. A congruence transformation of $M$ alone does not preserve its ordinary eigenvalues. In numerical work, a symmetric generalized eigensolver or a mass-normalized factorization avoids treating $M^{-1}K$ as an ordinary symmetric matrix.

### Work a coupled two-mass example

Choose $m_1=2$ kg, $m_2=1$ kg, $l_1=l_2=1$ m, and $g=1$ m/s² for transparent arithmetic. The following matrices list their numerical SI values; mass and stiffness retain their different physical units:

$$
M_*=\begin{pmatrix}3&1\\1&1\end{pmatrix},\quad
K_*=\begin{pmatrix}3&0\\0&1\end{pmatrix},\quad
M_{\rm rel}=\begin{pmatrix}6&2\\2&1\end{pmatrix},\quad
K_{\rm rel}=\begin{pmatrix}4&1\\1&1\end{pmatrix}.
$$

Writing $\lambda$ for the numerical squared frequency in s$^{-2}$ gives:

$$
\det(K_*-\lambda M_*)=2\lambda^2-6\lambda+3,\qquad
\lambda_\pm=\frac{3\pm\sqrt3}{2},\quad
\omega_\pm=\sqrt{\lambda_\pm},\quad
v_-=\begin{pmatrix}1\\\sqrt3\end{pmatrix},\quad
v_+=\begin{pmatrix}1\\-\sqrt3\end{pmatrix}.
$$

| Quantity | Absolute angles | Relative angles |
|---|---|---|
| $\mathrm{tr}(M)$ / (kg m²) | 4 | 7 |
| $\mathrm{tr}(K)$ / J | 4 | 5 |
| $\omega_-^2$ / s$^{-2}$ | 0.633975 | 0.633975 |
| $\omega_+^2$ / s$^{-2}$ | 2.366025 | 2.366025 |

The ordinary matrix traces change, while the two frequencies agree. Relative mode components are $(1,\sqrt3-1)$ and $(1,-\sqrt3-1)$; applying $A$ recovers the absolute vectors above. To compare with the native lab gravity $g=10$ m/s², multiply squared frequencies by 10 and frequencies by $\sqrt{10}$. The native lab still evaluates instantaneous geometry; this cell performs the supplementary linearization.

### Read mode shapes and approximation limits

Distinct modes are orthogonal in the mass inner product, rather than necessarily in the coordinate Euclidean dot product:

$$
v_-^TM_*v_+=0,\qquad
\widehat v_i=\frac{v_i}{\sqrt{v_i^TM_*v_i}},\qquad
\xi_i(t)=\varepsilon v_i\cos(\omega_it),\quad \dot\xi_i(0)=0.
$$

![Linearized physical point displacements for the two pendulum modes](/assets/learning/analytical-pendulum-modes.en.svg)

The figure uses $\varepsilon=0.03$ rad. The horizontal point displacements are $x_1\simeq l_1\xi_1$ and $x_2\simeq l_1\xi_1+l_2\xi_2$, so applying the coordinate change does not alter the plotted physical motion. These are linearized curves, not full nonlinear trajectories. A fixed-time check against an independently integrated nonlinear model shows angular errors shrinking by roughly eight when amplitude halves in these two examples. That finite experiment is consistent with the cubic leading correction of this symmetric equilibrium, and does not prove a uniform long-time error bound.

The Taylor expansions $\sin\theta=\theta+O(\theta^3)$ and $\cos(a-b)=1+O((a-b)^2)$ explain the scaling: with angles and rates proportional to amplitude, the first omitted acceleration terms are cubic. The statement is about a fixed finite interval near this regular equilibrium.

For AI systems, comparing spectra of learned mass matrices alone cannot validate a predicted oscillation frequency. State the equilibrium, align both quadratic forms, compare physical mode shapes, and separate linearization error from integration error.

### Execute the generalized eigenvector check

The standard-library cell checks the algebra and prints linearized physical positions at three times.

```python
from math import sqrt, cos

# SI: m1=2 kg, m2=1 kg, l1=l2=1 m, g=1 m/s^2.
M = [[3, 1], [1, 1]]
K = [[3, 0], [0, 1]]
for sign in [-1, 1]:
    lam = (3+sign*sqrt(3))/2
    omega = sqrt(lam)
    v = [1, -sign*sqrt(3)]
    relative = [v[0], v[1]-v[0]]
    assert max(abs(sum((K[i][j]-lam*M[i][j])*v[j] for j in range(2)))
               for i in range(2)) < 2e-15
    assert abs(relative[0]-v[0]) < 1e-14
    assert abs(relative[0]+relative[1]-v[1]) < 1e-14
    print(round(lam, 6), round(omega, 6), [round(x, 6) for x in relative])
    for t in [0, 1, 2]:
        q = [.03*x*cos(omega*t) for x in v]
        # Linearized horizontal positions, not nonlinear integration.
        x1, x2 = q[0], q[0]+q[1]
        print(t, round(x1, 6), round(x2, 6))
```

**Research questions with answers**

- Must the eigenvalues of $M$ match between angle conventions? **Answer:** No. They depend on the coordinate basis; the generalized eigenvalues of the matched pair $(K,M)$ give the frequencies.
- Are both modes necessarily orthogonal in the ordinary dot product? **Answer:** No. Here $v_-\cdot v_+=-2$, while $v_-^TM_*v_+=0$.
- Does an exact eigenpair certify a large-amplitude trajectory? **Answer:** No. It solves the linearized system; nonlinear motion and numerical time integration require separate error checks.

## Exercises with explained solutions

1. **Count compatible configuration freedoms.** Why are four point coordinates not four independent coordinates here? **Solution:** Two independent positive-length constraints remove two freedoms. C has rank two and J has two independent columns, so two generalized angles describe local configurations.

2. **A stationary tip with moving coordinates.** Use unit lengths, a=b=0 and rates (1, −1). **Solution:** v1=(1, 0), while v2=(1, 0)+(−1, 0)=0. For unit masses the direct kinetic energy is 0.5 J, despite both angular rates being nonzero.

3. **Check the relative convention.** Convert that example to u=(a, b−a). **Solution:** Relative rates are (1, −2). M_rel=[[5, 2], [2, 1]], so K=[5−8+4]/2=0.5 J. Reusing the absolute numerical rates would describe another velocity.

4. **Power with nonzero force components.** Add tip force (3, 0) N to that aligned state. **Solution:** Q_abs=(3, 3), Q_rel=(6, 3). Their contractions with (1, −1) and (1, −2) both vanish. This matches F_tip dot v_tip=0; generalized components are not individual powers.

5. **Different spectra, same kinetic form.** At a=0, b=pi/2 use rates (1, 1), unit masses and lengths. **Solution:** v1=(1, 0), v2=(1, 1), giving K=1.5 J. Absolute eigenvalues are (1, 2); relative eigenvalues are (2−sqrt(2), 2+sqrt(2)). Transforming the rates preserves K, not those eigenvalue lists.

6. **First-order versus finite length.** At aligned unit links take delta q=(1, −1), epsilon=0.5. **Solution:** The tangent point locations are (0.5, −1) and (0, −2), giving h1=h2=0.25. The exact configuration is (sin(0.5), −cos(0.5)) and (0, −2cos(0.5)), satisfying both lengths. Zero first-order constraint variation does not prove exact finite validity.

## Sources and further study

[MIT Lecture 13](https://ocw.mit.edu/courses/2-003j-dynamics-and-control-i-spring-2007/resources/lec13/) introduces generalized coordinates and forces; [Lecture 14](https://ocw.mit.edu/courses/2-003j-dynamics-and-control-i-spring-2007/resources/lec14/) develops virtual work. [Tedrake's multibody notes](https://underactuated.mit.edu/multibody.html) derive double-pendulum and manipulator mechanics using a relative second joint angle. The absolute-angle examples here explicitly transform that convention. Full Lagrange dynamics, reaction-force recovery, nonholonomic constraints and structure-preserving integration remain subsequent topics.
