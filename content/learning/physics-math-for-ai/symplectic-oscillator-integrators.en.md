## Compute structure as well as position

The oscillator q′=p, p′=-ω²q starts from q(0)=1, p(0)=0. Its exact trajectory q=cos(ωt), p=-ω sin(ωt) forms a closed ellipse in phase space. A short observation with small position error says little about long-time energy or geometric behavior. This chapter displays actual update points, physical energy and a different conserved quadratic separately.

The upper diagram uses equal scales for the dimensionless q and p coordinates. Rescaling the figure therefore preserves their relative lengths. Below, blue shows computed physical energy H=(p²+ω²q²)/2 and orange shows initial energy. Straight connectors indicate the order of discrete points; they are not an exact continuous Hamiltonian orbit.

## Three update rules

Forward Euler updates q and p using only their old values. Kick-then-drift symplectic Euler changes p first and moves q with the new p. Velocity Verlet applies half a kick, a full drift, and another half kick:

$$
p_{1/2}=p_n-\tfrac h2\omega^2q_n,\qquad
q_{n+1}=q_n+h p_{1/2},\qquad
p_{n+1}=p_{1/2}-\tfrac h2\omega^2q_{n+1}.
$$

In this linear two-dimensional example, forward Euler has determinant 1+(ωh)² while the other two methods have determinant one. This determinant condition connects to symplecticity in two dimensions. In higher dimensions, determinant one alone is insufficient: the general matrix condition is MᵀJM=J.

## Area, physical energy and modified energy

Symplecticity does not mean exact conservation of H. For this oscillator, symplectic Euler preserves H-hω²qp/2 in exact arithmetic. Verlet instead preserves

$$
H_* = \tfrac12\{p^2+\omega^2[1-(\omega h)^2/4]q^2\}.
$$

This quadratic is positive definite only for ωh<2. At or beyond the boundary, area preservation alone does not imply bounded motion. Reduce the step count to explore that distinction. Also separate small floating-point drift in the modified quantity from oscillations of the original physical H. These properties refer to the stated fixed-step linear model.

## Worked example and exercise

With ω=1 and h=0.2, forward Euler first reaches (1,-0.2), giving H=0.52. Symplectic Euler first changes p to -0.2 and then q to 0.96, giving H=0.4808. Adding its cross-term correction -hqp/2=0.0192 recovers the initial modified value 0.5.

Exercise: compute the corresponding first Verlet state. Solution: p_half=-0.1, q_next=0.98 and p_next=-0.198. Physical H is (0.198²+0.98²)/2=0.499802, different from 0.5. However, the modified quadratic equals its initial value 0.495. A method can therefore preserve one meaningful quantity without preserving every physical observable exactly.

## Structure in AI research

Hamiltonian neural networks and HMC proposals can be sensitive to integrator structure. A low short-time state loss and faithful long-time behavior are separate validation tasks. Record stage ordering, step size, energy drift and the appropriate geometric residual. HMC additionally requires its Metropolis correction and probabilistic assumptions; an energy plot alone cannot establish sampler correctness.

Reference: [MIT numerical methods course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
