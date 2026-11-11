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

## Stable eigenvalues can coexist with transient amplification

The live rotation/dilation family is normal, so its Euclidean radius follows eᵅᵗ. This simplification does not extend to all stable matrices. Consider A=[[-1,K],[0,−1]], K≥0. Its eigenvalues are both −1, but from (0,1) the solution is e⁻ᵗ(Kt,1). For K=4 its norm at t=1 exceeds its initial norm. Asymptotic decay and monotone Euclidean decay are different questions.

A suitable quadratic energy resolves the apparent contradiction. Set

$$
P=\begin{pmatrix}1/2&K/4\\K/4&1/2+K^2/4\end{pmatrix},
\quad A^TP+PA=-I,\quad \frac d{dt}(z^TPz)=-\|z\|_2^2.
$$

P is positive definite because its first leading minor is 1/2 and determinant is 1/4+K²/16. Thus its energy decreases even during Euclidean growth. Eigenvalue signs identify asymptotic stability for linear systems; norms and eigenvector conditioning determine transient amplification and robustness.

![Transient Euclidean amplification for three stable nonnormal systems, compared at the same time.](/assets/learning/differential-transient-en.svg)

| K | Radius at time 1 | Asymptotically stable |
|---|---|---|
| 0 | e⁻¹ | yes |
| 2 | √5/e | yes |
| 4 | √17/e | yes |

```python
from fractions import Fraction as F
from math import exp, hypot
def multiply(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
for K in (0, 2, 4):
    A = [[F(-1), F(K)], [F(0), F(-1)]]
    P = [[F(1, 2), F(K, 4)], [F(K, 4), F(1, 2)+F(K*K, 4)]]
    AT = list(map(list, zip(*A)))
    lhs, rhs = multiply(AT, P), multiply(P, A)
    assert [[lhs[i][j]+rhs[i][j] for j in range(2)] for i in range(2)] == [[-1, 0], [0, -1]]
    radius = exp(-1)*hypot(K, 1)
    if K == 4: assert radius > 1
    print(K, radius, 'det(P):', P[0][0]*P[1][1]-P[0][1]**2)
```

The rational matrix calculation verifies the Lyapunov identity without numerical differentiation. In the live structured families, isolation is now decided from the coefficients themselves. A determinant that underflows to zero for tiny nonzero coefficients is still shown as a rounded diagnostic, and no longer changes a genuinely isolated origin into a continuum of equilibria.

## Exercise: perturb the stable axis

Use alpha=0.5, beta=1 and initial state (0,1) in diagonal mode. At t=1 its radius is e^-1, despite the saddle classification. Replace x0 by 0.01 and solve for the time when x reaches one: t=log(100)/0.5. Explain why a finite simulation that ends before this time can hide instability. Finally set both coefficients to zero and identify the complete equilibrium set.

Further reading: [MIT first-order systems and phase portraits](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/). The exact trajectories and equal-scale visualization are independently implemented.
