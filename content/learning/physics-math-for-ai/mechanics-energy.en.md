## Motion is not the same as optimization

A mass attached to an ideal spring oscillates because inertia carries it past equilibrium. Plain gradient descent instead uses a first-order update to reduce an objective. Both may contain a gradient, but their state variables, units and energy balance differ. We will derive one physical model before discussing analogies to learning.

Assume a one-dimensional spring, fixed mass $m=1$ kg, stiffness $k>0$ in N/m, no friction, no external forcing and displacement $q$ in meters from equilibrium. Initial velocity is zero. These assumptions define the system boundary and determine what conservation claim is appropriate.

## From force to the exact trajectory

Hooke’s law and Newton’s second law give

$$
F=-kq,\qquad m\ddot q=-kq.
$$

With $q(0)=q_0,v(0)=0$, the exact solution is $q(t)=q_0\cos(\omega t)$ and $v(t)=-q_0\omega\sin(\omega t)$, where $\omega=\sqrt{k/m}$. At $m=k=q_0=1$ in the stated units, the period is $2\pi$ seconds.

The negative force restores the mass toward equilibrium. It does not make position move monotonically toward zero: velocity is an independent state variable.

## Derive the conserved energy

Potential energy is $V=kq^2/2$, whose negative position derivative is the spring force. Kinetic energy is $mv^2/2$. Thus

$$
E=\frac12mv^2+\frac12kq^2,\qquad
\frac{dE}{dt}=v(m\dot v+kq)=0.
$$

The cancellation uses the equation of motion and the absence of friction or external work. Adding damping changes the energy balance; conservation cannot simply be retained as an unquestioned label.

In the reference case $q_0=1$, initial energy is 0.5 J. At a quarter period, position is zero and all energy is kinetic. At half a period, position is $-1$ m and the energy is again potential.

## A discrete method with a chosen order

The lab uses velocity-first symplectic Euler:

$$
v_{n+1}=v_n-h(k/m)q_n,\qquad
q_{n+1}=q_n+h v_{n+1}.
$$

The position uses the **new** velocity. This differs from forward Euler using the old velocity in both updates. For $h=0.1,m=k=q_0=1$, the first new velocity is $-0.1$ and position is 0.99. Its energy is 0.49505 J, not exactly 0.5.

Symplectic structure preservation does not mean exact preservation of the original energy at every step. This distinction matters when evaluating a “physics-preserving” learned model: specify which structure is preserved and at what numerical tolerance.

## Derive the stability range

In state coordinates $(q,v)$, the update matrix is

$$
M=\begin{pmatrix}1-h^2\omega^2&h\\-h\omega^2&1\end{pmatrix},\qquad\det M=1.
$$

Its trace is $2-h^2\omega^2$. Within $0<h\omega<2$, eigenvalues lie on the unit circle and the oscillator remains bounded for this linear problem. At the boundary $h\omega=2$, repeated eigenvalues need not yield bounded powers: a defective update can produce linear growth. Volume preservation alone does not guarantee stability.

The lab computes 100 steps and reports actual maximum energy deviation over those states. Changing $h$ changes final time. To measure convergence order, hold the physical final time fixed and adjust the step count.

## Read and extend the experiment

Blue is exact position and orange is numerical position at matching displayed times. The table includes terminal position and velocity, initial energy, final energy and maximum energy error. A visually close position curve does not prove energy conservation or long-time accuracy.

For AI research, compare learned forces against known force derivatives and inspect trajectories, units, invariants and resolution sensitivity. A network that matches a short trajectory may fail at longer times or different initial states. A Hamiltonian parameterization supplies structure only under its stated state variables and solver assumptions.

## Exercises and reference

1. Recompute the first numerical step and its energy. **Answer:** $q=0.99,v=-0.1,E=0.49505$.
2. Double stiffness and explain how frequency and stable step size change.
3. Add a damping force $-cv$ on paper and derive $dE/dt=-cv^2$.
4. Compare methods at the same physical final time, not merely the same number of updates.

See [MIT Classical Mechanics](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/) for forces, energy and oscillation. The numerical method and checks above are specified directly for this teaching oscillator, not a validated model of contact or rigid-body motion.
