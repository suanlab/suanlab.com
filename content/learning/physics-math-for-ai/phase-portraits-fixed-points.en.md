## A phase portrait organizes many possible initial states

For a two-dimensional autonomous system z'=Az, a phase portrait shows the vector field and trajectories in the state plane. It differs from plotting each component against time. The upper lab diagram uses horizontal x and vertical y with equal physical scale; the lower diagram shows x(t) and y(t) separately.

Blue arrows indicate the actual field direction after normalization for readability. Their lengths are not physical speeds. The orange orbit comes from the exact matrix flow, and the marked point corresponds to the selected time. This makes direction, geometric path and time parameterization distinct visual objects.

## Rotation and dilation have separate roles

For the matrix with rows (alpha,-beta) and (beta,alpha),

$$
z(t)=e^{\alpha t}
\begin{pmatrix}\cos\beta t&-\sin\beta t\\
\sin\beta t&\cos\beta t\end{pmatrix}z(0),
\qquad \|z(t)\|=e^{\alpha t}\|z(0)\|.
$$

Negative alpha contracts radii, positive alpha expands them, and beta controls angular motion. For nonzero beta, alpha zero gives a center: nearby trajectories remain near the origin without approaching it. Lyapunov stability therefore differs from asymptotic stability. With beta zero the same formulas give radial repeated-node behavior instead of spirals.

At alpha=-0.3 and initial radius one, the radius at time one is exp(-0.3). At alpha zero it remains one. Comparing these cases prevents a closed orbit from being mistaken for convergence.

## A stable direction does not make a stable fixed point

The second matrix is diagonal with entries alpha and -beta. With alpha positive and beta positive, the origin is a saddle. The x axis expands and the y axis contracts:

$$
x(t)=x_0e^{\alpha t},\qquad y(t)=y_0e^{-\beta t}.
$$

Starting exactly on x0=0 produces a trajectory converging to the origin. An arbitrarily small nonzero x perturbation eventually grows, so the fixed point is not stable. Stability quantifies a neighborhood of initial states, not just the orbit a particular simulation selected.

## Check whether the equilibrium is isolated

Fixed points solve Az=0. Nonzero determinant makes the origin the only equilibrium. A zero eigenvalue can instead produce a line of equilibria; if A is zero, every point is fixed. The readout distinguishes nonisolated states from ordinary sinks, sources, centers and saddles. For nonlinear systems, zero-real-part eigenvalues can require higher-order analysis rather than a strict-sign linear test.

## Exercise: perturb the stable axis

Use alpha=0.5, beta=1 and initial state (0,1) in diagonal mode. At t=1 its radius is e^-1, despite the saddle classification. Replace x0 by 0.01 and solve for the time when x reaches one: t=log(100)/0.5. Explain why a finite simulation that ends before this time can hide instability. Finally set both coefficients to zero and identify the complete equilibrium set.

Further reading: [MIT first-order systems and phase portraits](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). The exact trajectories and equal-scale visualization are independently implemented.
