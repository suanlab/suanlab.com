## Zero training loss does not identify one model

The single equation a w₁+b w₂=y does not uniquely determine two weights. Its solutions lie on a line in weight space. They fit the same training target but can disagree in an unobserved test direction. Implicit bias asks which solution an optimization method and initialization select without an explicit norm penalty.

The upper plot uses equal scales for w₁ and w₂. Blue is the actual GD path and endpoint, orange dashed is the training constraint, and the hollow circle is the minimum-Euclidean-norm solution. The lower plot shows training loss; geometric edge length is not a generalization score.

## Gradient direction and nullspace

Let x=(a,b) and L(w)=(x·w-y)²/2. Then

$$
w_{k+1}=w_k-\eta(x\cdot w_k-y)x.
$$

Every update is parallel to x. For orthogonal v=(-b,a)/||x||, the projection v·w remains unchanged in exact arithmetic. Initial nullspace information is therefore invisible to the training equation. On an unstable numerical trajectory, cancellation and roundoff can spoil the observed invariant; distinguish reported floating-point values from the exact identity.

## Residual dynamics and convergence

$$
r_{k+1}=(1-\eta\|x\|^2)r_k,\qquad 0<\eta\|x\|^2<2.
$$

Under this strict condition the residual vanishes and the initialization-dependent limit is

$$
w_\infty=w_0+\frac{y-x\cdot w_0}{\|x\|^2}x.
$$

Zero initialization has no nullspace component, selecting yx/||x||², the minimum Euclidean norm. A different gradient metric, momentum rule, parameterization or nonlinear network need not select the same object. Eta zero does not learn; a boundary multiplier of minus one generally oscillates rather than damps.

## The selected norm follows the optimization metric

Replace Euclidean GD by w_next=w−eta*P*x*(xᵀw−y), where P is fixed symmetric positive definite. Residuals contract by 1−eta*xᵀPx. Under the strict stability condition, summing the row-direction increments gives

$$
w_\infty=w_0+\frac{y-x^Tw_0}{x^TPx}Px.
$$

A Lagrange-multiplier calculation minimizes (w−w0)ᵀP^(−1)(w−w0)/2 subject to xᵀw=y and produces exactly that limit. Any other feasible perturbation z has xᵀz=0, so its cross term with w_infinity−w0 in the inverse metric is zero. Pythagoras proves uniqueness and optimality in that metric.

For x=(1,1), y=1 and zero initialization, diagonal preconditioners change the solution even though every solution fits the same equation. This explains why a change of parameter scale can change an optimizer's implicit choice.

| Diagonal P | Selected solution |
|---|---|
| (1,1) | (1/2,1/2) |
| (4,1) | (4/5,1/5) |
| (1,4) | (1/5,4/5) |

```python
from fractions import Fraction as F
for diagonal in ((1,1),(4,1),(1,4)):
    a,b=map(F,diagonal);solution=(a/(a+b),b/(a+b))
    norm=lambda w:w[0]**2/a+w[1]**2/b
    for t in (F(-1),F(0),F(2,3),F(1)):
        alternative=(t,1-t)
        difference=tuple(u-v for u,v in zip(alternative,solution))
        assert norm(alternative)==norm(solution)+norm(difference)
    print(diagonal,solution)
```

The code checks the inverse-metric Pythagorean identity at feasible alternatives using exact fractions. The live controls still implement Euclidean GD; the preconditioner is an auxiliary derivation. Euclidean null projections need not be preserved by a general P. What remains invariant is every vᵀP^(−1)w with xᵀv=0. Momentum, changing metrics and nonlinear networks require separate arguments; equal interpolation is insufficient to transfer a norm-selection theorem.

## Worked example and exercise

With a=b=y=1 and eta=.25, residual multiplier is .5. From zero, the first two states are (.25,.25) and (.375,.375), converging to (.5,.5). Starting instead at (1,-1) adds the same row-space increments and converges to (1.5,-.5). Both limiting weights sum to one.

Exercise: compare predictions on test feature z=(1,-1). Solution: the minimum-norm solution predicts zero, while the other initialization predicts two. Equal training loss does not fix unseen directions. Reversing the chosen null-vector orientation changes the projection sign but not the retained information.

Euclidean norm also depends on coordinates. Reparameterize w1+w2=1 by v1=2w1 and v2=w2. The constraint becomes .5v1+v2=1, whose minimum-norm v is (.4,.8). Mapping back gives w=(.2,.8), different from (.5,.5). The function class is unchanged but the optimization geometry differs, so record parameterization and metric.

## Implicit regularization in AI research

Observed good generalization does not establish that every optimizer finds minimum norm. Specify design rank, initialization, step stability, metric and the particular norm. This linear example admits an explicit selection proof and invariant. Implicit bias of neural networks requires its own assumptions and analysis rather than an unconditional transfer of this result.

References: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) and [MIT learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/).
