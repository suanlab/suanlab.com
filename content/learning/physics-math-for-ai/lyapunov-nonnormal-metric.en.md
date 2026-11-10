## Stable eigenvalues do not promise monotone Euclidean decay

Consider a linear system whose two eigenvalues are negative. Its state converges to zero, but a nonnormal coupling can temporarily increase the ordinary Euclidean norm. A Lyapunov function can certify stability in another positive-definite metric. The apparent disagreement is between metrics, not between two solutions of the differential equation.

The lab studies a continuous-time autonomous system, without a control input:

$$
\dot x=Ax,\qquad A=\begin{pmatrix}-a&k\\0&-b\end{pmatrix},\quad a,b>0.
$$

The eigenvalues are -a,-b. The off-diagonal k transfers the second component into the first; at large coupling the eigenvectors need not form an orthogonal coordinate basis.

## Construct a certificate for every state

Seek symmetric P=[[p,r],[r,s]] satisfying AᵀP+PA=-I. Solving its three scalar equations gives

$$
p=\frac1{2a},\qquad r=\frac{k}{2a(a+b)},\qquad
s=\frac1{2b}+\frac{k^2}{2ab(a+b)}.
$$

Its first principal minor is positive and

$$
\det P=\frac1{4ab}+\frac{k^2}{4ab(a+b)^2}>0.
$$

Therefore P is positive definite. Define V(x)=xᵀPx. For every state, not just the displayed initial condition,

$$
\dot V=x^T(A^TP+PA)x=-\|x\|^2.
$$

V is radially unbounded and strictly decreases away from zero. The matrix identity and positive definiteness provide a global certificate; a finite trajectory picture is only an illustration of it.

## Why transient growth remains possible

For the Euclidean squared norm,

$$
\frac{d}{dt}\|x\|^2=2x^TAx
=-2ax_1^2+2kx_1x_2-2bx_2^2.
$$

Take a=b=1,k=4,x(0)=(1,1). Its initial derivative is -2+8-2=4, while the initial V derivative is -2. Here P=[[0.5,1],[1,4.5]] and V(0)=7. Ordinary squared size initially grows, but the weighted metric decreases.

The general eigenvalue bounds lambda_min(P)||x||²≤V≤lambda_max(P)||x||² yield exponential decay of V, with a possibly conservative Euclidean prefactor. That prefactor permits finite transient amplification. Comparing the numerical height of V with Euclidean energy as an efficiency ranking would be meaningless because their scales differ.

## Exact trajectories and the repeated-pole limit

The second state is x2(t)=exp(-bt)x2(0). For a different from b,

$$
x_1(t)=e^{-at}x_1(0)+k\frac{e^{-bt}-e^{-at}}{a-b}x_2(0).
$$

At a=b the second term becomes kt exp(-at)x2(0). The implementation uses an expm1 expression near equal rates rather than subtracting nearly equal exponentials. In the default example, x1(t)=exp(-t)(1+4t) and x2(t)=exp(-t).

The blue phase curve is the actual trajectory. Its orange contour is the initial level V=V(0), computed from P; the orange point is the initial state and blue point the terminal state. This contour is a mathematical sublevel boundary, not a physical resource limit or probabilistic confidence ellipse.

## Exercise and solution

Set k=0,a=1,b=2,x(0)=(1,1). Compute P and both initial derivatives. Then restore coupling and compare their roles.

**Solution.** P=diag(0.5,0.25), so V(0)=0.75. Euclidean squared-size derivative is -2-4=-6, while V derivative is -2. Without coupling both quantities decrease directly. Restoring k can change the Euclidean derivative's sign, but the constructed Lyapunov derivative remains -||x||². The state zero is handled as a zero trajectory, without inventing a normalized ratio 0/0.

## Connection to optimization and learned dynamics

Nonnormal transients matter in dynamical models, recurrent computations and sensitivity analysis. A finite response can amplify despite asymptotic stability, and a certificate depends on its chosen coordinates and metric. This example assumes known linear dynamics with positive a,b. A learned nonlinear system, discretized integrator, uncertain model or feedback controller needs its own stability argument; continuous-time decay does not automatically certify an arbitrary numerical update.

## Primary reference

Russ Tedrake, [Underactuated Robotics: Lyapunov Analysis](https://underactuated.mit.edu/lyapunov.html). The triangular family, explicit P, trajectories and worked examples here are original constructions.
