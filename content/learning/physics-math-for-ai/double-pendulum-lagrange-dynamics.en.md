## From a configuration diagram to moving masses

The earlier double-pendulum lesson computed positions, instantaneous velocities, constraint tangents and a mass matrix. Those calculations did not determine how angular velocities change. We now derive the missing equations of motion and integrate them over two seconds of physical time.

This is a planar system with two positive point masses, two positive-length massless rigid rods and a fixed pivot. Gravity is uniform with g=10 m/s². There is no friction, actuator input or moving support. A rigid rod can support compression as well as tension; it is not a string that must remain taut. These assumptions explain which forces do work and which numerical conservation checks are appropriate.

## Choose angles and write Cartesian positions

Let a and b be absolute angles measured from downward vertical. They are not shoulder angle plus relative elbow angle. With lengths l₁ and l₂,

$$
\boldsymbol p_1=l_1\begin{pmatrix}\sin a\\-\cos a\end{pmatrix},\qquad
\boldsymbol p_2=\boldsymbol p_1+l_2\begin{pmatrix}\sin b\\-\cos b\end{pmatrix}.
$$

Write wa=adot and wb=bdot. Differentiate these positions before constructing kinetic energy. The second mass inherits the velocity of the first as well as the relative velocity along its own rotating rod. This is the physical origin of the kinetic cross term.

MIT's [Multi-Body Dynamics](https://underactuated.mit.edu/multibody.html) develops Lagrange and manipulator equations, including a double pendulum expressed with a relative elbow angle. Our absolute-angle component formulas below are independently derived from these Cartesian positions. Copying a relative-angle mass matrix without transforming the coordinates would give a different model.

## Build the energy and mass matrix

Let β=m₂l₁l₂ and δ=a−b. Summing the point-mass kinetic energies gives

$$
K=\frac12(m_1+m_2)l_1^2w_a^2+\frac12m_2l_2^2w_b^2
+\beta\cos\delta\,w_aw_b
=\frac12\dot{\boldsymbol q}^{T}M(\boldsymbol q)\dot{\boldsymbol q},
$$

$$
M=\begin{pmatrix}(m_1+m_2)l_1^2&\beta\cos\delta\\
\beta\cos\delta&m_2l_2^2\end{pmatrix},\qquad \boldsymbol q=(a,b)^T.
$$

The matrix entries have units kg m². The cross contribution can be negative, while the complete kinetic energy is nonnegative. Its Cartesian point-velocity calculation provides a separate check of the same physical quantity.

We choose potential energy zero when both rods point down:

$$
U=g\left[(m_1+m_2)l_1(1-\cos a)+m_2l_2(1-\cos b)\right].
$$

This differs from a height-based potential by a constant. It changes displayed absolute energy values but not forces, accelerations or energy differences. The previous instantaneous-coordinate lesson may use a different potential zero; compare the convention before comparing numbers.

## Apply partial and total derivatives carefully

Use L=K−U and the unforced equations d/dt(∂L/∂wa)−∂L/∂a=0, with a corresponding equation for b. For example,

$$
\frac{\partial K}{\partial w_a}=M_{11}w_a+\beta\cos\delta\,w_b.
$$

The total derivative differentiates both velocities and the angle-dependent cosine. Meanwhile ∂K/∂a=−β sinδ wa wb. After subtracting the coordinate derivative, the surviving velocity term in the first equation is +β sinδ wb². Performing the second calculation gives −β sinδ wa².

This is why simply writing M qddot=−∇U is generally incorrect. A configuration-dependent mass matrix carries extra velocity-dependent terms. Even if its off-diagonal entry happens to be zero at one configuration, its derivatives need not vanish there.

## State and solve the coupled equation

Define

$$
\boldsymbol c=\begin{pmatrix}\beta\sin\delta\,w_b^2\\-\beta\sin\delta\,w_a^2\end{pmatrix},\qquad
\boldsymbol G=\begin{pmatrix}(m_1+m_2)gl_1\sin a\\m_2gl_2\sin b\end{pmatrix}.
$$

The result is

$$
M\ddot{\boldsymbol q}+\boldsymbol c+\boldsymbol G=0.
$$

Each row has torque units N m. G is the potential gradient; the generalized gravitational force is −G. The vector c collects inertial velocity terms in this convention and is not an additional actuator or friction force.

The determinant simplifies to

$$
\det M=m_2l_1^2l_2^2\left(m_1+m_2\sin^2\delta\right)>0.
$$

Positive masses and lengths guarantee invertibility at every angular configuration. With h=−c−G, solve aa=(M₂₂h₁−M₁₂h₂)/det M and ab=(M₁₁h₂−M₁₂h₁)/det M. This solves instantaneous accelerations; a separate ODE method is needed to obtain an entire trajectory.

## Check mechanical energy at the equation level

The exact unforced system conserves E=K+U. Differentiating the Cartesian energy or expanding the matrix expression gives

$$
\frac{dE}{dt}=\dot{\boldsymbol q}^{T}
\left(M\ddot{\boldsymbol q}+\boldsymbol c+\boldsymbol G\right)=0.
$$

The velocity-dependent contribution is essential to this cancellation. The implementation evaluates the energy derivative separately from the energy differences on a computed time trajectory. A tiny instantaneous equation residual does not make a finite-step integrator exactly energy-conserving.

If later adding an actuator, external power would appear on the right. Then mechanical energy can change physically, and a correct diagnostic compares energy change with supplied work. This unit has no such input, so zero external work is an explicit model assumption.

## Recover Cartesian acceleration and rod forces

Differentiate the position map twice. For the first mass,

$$
\boldsymbol a_1=l_1\ddot a\begin{pmatrix}\cos a\\\sin a\end{pmatrix}
-l_1w_a^2\begin{pmatrix}\sin a\\-\cos a\end{pmatrix}.
$$

Add the analogous second-rod contribution for a₂. Let gravity acceleration be gvec=(0,−g). The force of rod two on mass two is R₂=m₂(a₂−gvec), and the force of rod one on mass one is R₁=m₁(a₁−gvec)+R₂. Rod two exerts −R₂ on mass one.

For ideal rods these forces have zero tangential components along their own rods. The complete constraint power is (R₁−R₂)·v₁+R₂·v₂=0. A nonzero force can therefore redirect motion without doing work. A signed tension is defined as minus the force's projection onto the outward rod direction; a negative value represents compression and is permitted by this rigid-rod model.

## Worked example: a diagonal matrix still has inertial terms

Set m₁=m₂=l₁=l₂=1, a=π/2, b=0, wa=1 and wb=−1. The mass matrix is diag(2,1), but c=(1,−1) N m and G=(20,0) N m. Hence aa=−10.5 rad/s² and ab=1 rad/s².

Omitting c would predict (−10,0), a physically different acceleration. Here K=1.5 J, U=20 J and E=21.5 J. The instantaneous kinetic-energy rate is −20 W and potential-energy rate is +20 W. The forces are R₁=(−1,0) N and R₂=(0,0.5) N; their constraint power is zero.

If both angular rates instead start at zero in this configuration, c vanishes and the accelerations are (−10,0). Both masses initially accelerate downward at g, and the instantaneous rod forces are zero. This does not mean the rods remain force-free later.

## Convert the equation to an ODE

Use the state z=(a,b,wa,wb). Its derivative is (wa,wb,aa,ab), with accelerations obtained by the mass-matrix solve at the current state. The model uses classical explicit fourth-order Runge–Kutta. Each step evaluates the derivative at four intermediate states, including their current mass matrices and velocity terms.

The base step is 0.0025 s divided by 2 to the power resolution−1, with four resolution choices. A second trajectory uses half that step. Output curves are sampled every 0.01 s over a two-second horizon. The query-time control selects a stored physical state; it does not change the initial conditions or the internal integration interval.

RK4 has fourth-order global accuracy under suitable smoothness on a fixed finite interval. This does not promise an exact factor of sixteen for every measured endpoint error. Cancellations, accumulated dynamics and roundoff can affect observed ratios. The short horizon here is a numerical experiment setting, not a limit on the length of the educational curriculum.

## Read error diagnostics honestly

Compare the chosen-step and half-step energy errors E(t)−E(0). Also inspect their angular-position gap in radians and angular-rate gap in rad/s on separate axes. A refinement gap is a useful diagnostic, not a certified bound on the unknown exact solution. Energy preservation alone also cannot certify that a phase or timing prediction is accurate. Because Cartesian positions are regenerated from the angles, rod lengths are satisfied by construction even when time integration is inaccurate. A tiny geometric length residual alone does not establish accurate time evolution.

An independent reference generator differentiates Cartesian energies symbolically and solves those equations with DOP853, using tighter tolerances and a smaller maximum step to check reference consistency. We compare the runtime trajectories against those references. Neither implementation reads the other's acceleration formulas.

Downward rest and perfectly upright rest are equilibrium checks. Exact cardinal identities keep the represented equilibria stationary. Upright equilibrium is physically unstable to perturbations; a zero plotted derivative at the exactly specified state does not establish robustness. A short complex-looking trajectory or numerical energy drift is insufficient evidence of chaos.

## Research extension: identify the mass scale with physical observations

A video can reveal angles and angular velocities very accurately while leaving an absolute mass scale undetermined. In this frictionless gravity-only model, multiplying both masses changes energy and rod forces but does not change the timed trajectory. This provides a concrete identifiability test for a learned Lagrangian.

### Scale both inertial and gravitational masses

Keep positive lengths, known $g$, the fixed pivot and ideal rods. There is no applied force besides gravity and constraint reactions. Scale both masses by the same dimensionless $\sigma>0$:

$$
m_i^{(\sigma)}=\sigma m_i,\quad\sigma>0,\qquad
L_\sigma=\sigma L,\quad M_\sigma=\sigma M,\quad
\boldsymbol c_\sigma=\sigma\boldsymbol c,\quad G_\sigma=\sigma G,\qquad
\mathcal E(L_\sigma)=\sigma\mathcal E(L).
$$

Dividing the unforced equations by $\sigma$ recovers the same acceleration field. With the same complete angular initial state, the solutions agree on their common interval of existence, without rescaling physical time. In the same potential-zero convention:

$$
q_\sigma(t)=q_1(t),\quad\dot q_\sigma(t)=\dot q_1(t),\qquad
E_\sigma=\sigma E_1,\quad p_\sigma=\sigma p_1,\quad
R_{j,\sigma}=\sigma R_{j,1}.
$$

The rod-force relation follows directly from $R_2=m_2(a_2-g_{\rm vec})$ and $R_1=m_1(a_1-g_{\rm vec})+R_2$. Positive definiteness of the mass matrix is retained at every positive scale; it does not select kilograms. This scaling changes the kinetic Hessian and is not a total derivative.

Use the earlier moving state $a=\pi/2$, $b=0$, rates $(1,-1)$ rad/s, unit lengths, $g=10$ m/s² and base masses $(1,1)$ kg. Exact initial values are:

| $\sigma$ | Initial $\ddot a$ / (rad/s²) | Initial energy / J | Initial rod-one $R_x$ / N |
|---|---|---|---|
| 1/2 | -21/2 | 43/4 | -1/2 |
| 1 | -21/2 | 43/2 | -1 |
| 2 | -21/2 | 43 | -2 |

The fractional scale is part of this supplementary family. Native integer mass controls can directly compare the middle and last rows. A calibrated force sensor or energy measurement supplies information absent from angles alone.

### A small raw residual can be a scale artifact

Evaluate the Euler–Lagrange residual at one fixed candidate path or local trial state, rather than silently changing the candidate with the parameters. In this same absolute-angle basis:

$$
r_\sigma=\sigma r_1,\qquad
\|r_\sigma\|_2^2=\sigma^2\|r_1\|_2^2,\qquad
M_\sigma^{-1}r_\sigma=M^{-1}r_1
=\ddot q_{\rm trial}-\ddot q_{\rm model}.
$$

An unconstrained learner can send $\sigma$ toward zero while keeping every mass positive, reducing an unnormalized squared residual without improving the predicted acceleration. The zero-mass endpoint itself is outside the model. The native mass interval remains unchanged; this limiting argument concerns a broader learned positive-mass family. Mass normalization removes this trivial loss collapse but still does not identify the scale. It also does not make a fixed coordinate norm invariant under arbitrary coordinate changes.

### Calibrate with an informative known force

A known physical force on the second point provides a generalized input. Keep its newton value fixed while changing the candidate masses:

$$
Q_{\rm tip}=J_{\rm tip}^T\boldsymbol F,\qquad
\sigma\left(M\ddot q_\sigma+\boldsymbol c+G\right)=Q_{\rm tip}.
$$

At one fixed configuration and velocity, subtract the two forced acceleration equations:

$$
\ddot q_\sigma-\ddot q_1
=\left(\frac1\sigma-1\right)M^{-1}Q_{\rm tip}
\quad\text{at the same }(q,\dot q).
$$

Thus a nonzero $Q_{\rm tip}$ breaks this common-scale ambiguity within the stated model. A force absorbed entirely by the ideal constraints has $Q_{\rm tip}=0$ and does not distinguish the scale through angular acceleration. Calibration must excite an observable direction.

For a separate downward-rest probe with unit base masses and lengths, a horizontal tip force of 1 N gives the following numerical SI values:

$$
M=\begin{pmatrix}2&1\\1&1\end{pmatrix},\qquad
Q_{\rm tip}=\begin{pmatrix}1\\1\end{pmatrix},\qquad
\ddot q_\sigma=\begin{pmatrix}0\\1/\sigma\end{pmatrix}
\quad\text{at downward rest, horizontal tip force }1\,\mathrm N.
$$

The mass matrix is in kg m², generalized force in N m and angular acceleration in rad/s². A vertical force at that same downward configuration instead has zero generalized input. The known horizontal force is a Python companion; the native dynamics lab remains the specified gravity-only system.

![Identical gravity-only trajectories and mass-scale calibration at downward rest](/assets/learning/analytical-lagrange-scale.en.svg)

The left panel starts from the moving reference used in the table; the right panel starts from downward rest. They are different specified experiments. The independent Cartesian-projection/RK4 construction compares three mass scales over 2 s at 201 output times. Scale agreement checks the symmetry of the model, not the absolute accuracy of a finite step. The existing refined DOP853 comparisons and step diagnostics remain separate evidence for trajectory accuracy.

For AI dynamics learning, specify which parameters are identifiable from which sensors, keep input calibration fixed, and test more than energy conservation or a raw residual.

### Execute the scale and calibration checks

The standard-library cell reproduces the table, the horizontal-force response and an off-equation trial whose raw loss shrinks with mass scale.

```python
from fractions import Fraction as Q

def solve(M, rhs):
    d = M[0][0]*M[1][1]-M[0][1]*M[1][0]
    return [(M[1][1]*rhs[0]-M[0][1]*rhs[1])/d,
            (M[0][0]*rhs[1]-M[1][0]*rhs[0])/d]

for sigma in map(Q, ['1/2', '1', '2']):
    # Moving reference: a=pi/2, b=0, rates=(1,-1), g=10, unit lengths.
    M = [[2*sigma, Q(0)], [Q(0), sigma]]
    acceleration = solve(M, [-21*sigma, sigma])
    energy = 43*sigma/2
    rod1_x = -sigma
    assert acceleration == [Q(-21, 2), Q(1)]
    print(sigma, acceleration[0], energy, rod1_x)
    # A DIFFERENT configuration: downward rest with horizontal tip force 1 N.
    down_M = [[2*sigma, sigma], [sigma, sigma]]
    forced = solve(down_M, [Q(1), Q(1)])
    assert forced == [0, 1/sigma]
    print('calibrated downward-rest response:', forced)

for sigma in map(Q, ['1', '1/2', '1/4']):
    M = [[2*sigma, sigma], [sigma, sigma]]
    residual = [2*sigma, sigma]  # Trial acceleration (1,0) at downward rest.
    assert sum(r*r for r in residual) == 5*sigma*sigma
    assert solve(M, residual) == [1, 0]
# A smaller raw residual did not improve this acceleration mismatch.
```

**Research questions with answers**

- Does exact angle reconstruction fix the absolute masses? **Answer:** No. Common positive mass scaling preserves this gravity-only timed motion.
- Does mass-normalizing the residual calibrate kilograms? **Answer:** No. It removes a scale-dependent loss effect while leaving the physical scale unidentified.
- Does every known tip force identify the scale from angles? **Answer:** No. Its generalized input must be nonzero; a purely normal force at downward rest is absorbed by the constraints.

## Exercises with explanations

1. Why can M be diagonal while c is nonzero? **Solution:** A zero cosine cross term at one angle does not make its sine derivative zero. At δ=π/2, the cross entry is zero but the velocity-dependent terms remain.
2. Compute initial accelerations in the worked example. **Solution:** The right side is −c−G=(−21,1). Dividing by the diagonal entries gives (−10.5,1) rad/s². Ignoring c gives the wrong result.
3. At downward rest with m₁=2 and m₂=3, what are the rod forces? **Solution:** R₂=(0,30) N supports mass two. R₁=(0,50) N supports both masses. Their power is zero because the velocities are zero.
4. Does subtracting a constant from potential change the motion? **Solution:** No. Its coordinate gradient vanishes. Absolute energy shifts, while accelerations, energy differences and the chosen-step energy error remain unchanged.
5. Does halving the integration step prove the finer trajectory is exact? **Solution:** No. Both are numerical approximations. Compare refinement gaps, independent reference paths and energy diagnostics; do not treat their agreement as an exact-solution certificate.
6. If a signed rod tension is negative, must the simulation be discarded? **Solution:** Not for these rigid rods, which can transmit compression. A taut-string model has a different admissible force constraint and would need a different physical formulation.

## Connection to AI models and future topics

Learning a configuration-dependent mass matrix without the corresponding inertial terms can produce plausible local energies and incorrect time evolution. Verify the complete equation, its Cartesian force interpretation, and trajectory accuracy separately. Keep coordinate conventions and potential zeros explicit when comparing learned models or datasets.

Later units develop Hamiltonian variables, Poisson brackets, Noether's theorem and structure-preserving integration. A short RK4 experiment and the direct energy identity here do not complete those broader topics. They supply concrete dynamics against which later formulations can be compared.
