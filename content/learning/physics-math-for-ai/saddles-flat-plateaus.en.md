## A small gradient needs a curvature diagnosis

A stationary point has zero gradient, but it can be a minimum, maximum or saddle. A very small gradient can also occur on a nearly flat region far from a desired solution. The Hessian supplies additional local information: a negative eigenvalue at a stationary point identifies a descent direction, whereas a zero eigenvalue leaves higher-order terms relevant.

The lab contrasts f(x,y)=x²/2+cos y with f(x,y)=x²/2+y⁴/4. Both have zero gradient at the origin, but they have different geometry and different global objective values.

## An exact saddle can trap deterministic GD

For the cosine model,

$$
\nabla f=(x,-\sin y),\qquad H=\operatorname{diag}(1,-\cos y).
$$

At the origin the Hessian eigenvalues are one and minus one. Moving in x increases the objective, while a small nonzero y decreases it. The global minimum value is −1, attained at x=0 and odd multiples of π in y. Thus the origin has objective excess two.

GD gives x⁺=(1−η)x and y⁺=y+η sin y. Starting with y exactly zero keeps y zero forever. Deterministic gradient descent does not invent a perturbation because an escape theorem sounds plausible. Starting instead from y=0.01 explicitly changes the initialization. Near zero, sin y≈y, so the unstable direction initially grows by approximately 1+η each step.

## A zero Hessian eigenvalue can belong to a minimum

For the quartic model,

$$
\nabla f=(x,y^3),\qquad H=\operatorname{diag}(1,3y^2),\qquad y^+=y-\eta y^3.
$$

The origin is the unique global minimum, despite its zero y-curvature. No negative curvature is present. A second-order test with a zero eigenvalue is inconclusive by itself, but the nonnegative quartic term resolves this particular case.

The shrinking cubic gradient makes progress slow near the optimum. A continuous-time approximation dy/dk=−ηy³ suggests y²≈1/(y₀⁻²+2ηk). This is intuition for a sublinear tail, not an equality for discrete GD or a theorem about arbitrary plateaus.

## Read the cross-section and the actual trajectory

Blue plots the x=0 potential against y. Orange plots actual full objective values at each iterate's y coordinate. If the initial x is nonzero, orange starts above the blue cross-section because x²/2 contributes additional energy. It would be wrong to read the orange line as a pure one-dimensional potential or to infer iteration speed from unequal horizontal gaps.

## Quantify escape and derive the discrete quartic tail

The local quadratic saddle f=x²/2−γy²/2 has yₖ=(1+ηγ)ᵏy₀. For nonzero y₀, the first step reaching radius r>|y₀| is the ceiling of log(r/|y₀|)/log(1+ηγ). For y₀=0 it never escapes. This is a local linear model, unbounded below globally; it is not the full cosine objective or a general stochastic escape theorem.

The quartic tail admits a discrete proof rather than only an ODE analogy. If 0<ηy₀²<1, then y remains positive and decreases to zero. With sₖ=1/yₖ², exact algebra gives

$$
s_{k+1}-s_k=\frac{2\eta-\eta^2y_k^2}{(1-\eta y_k^2)^2}\longrightarrow2\eta,
\qquad y_k\sim(2\eta k)^{-1/2}.
$$

Cesàro averaging of these increments proves the asymptotic relation. The quartic objective then decays like k⁻² while its gradient decays like k⁻³ᐟ². A singular Hessian at this minimum changes rates without introducing negative curvature.

| Diagnostic | Strict saddle | Quartic minimum |
|---|---|---|
| Local unstable mode | Present | Absent |
| Exact-zero initialization | Can remain trapped | Already optimal |
| Slow small-gradient motion | Needs curvature and initialization analysis | Consistent with a proved algebraic tail |

The 80-digit experiment below extends the iteration count beyond the lab's finite controls. It checks the tail scale k yₖ² approaching 1/(2η); finite ratios are not asserted equal to their asymptotic limit.

## Reproduce the calculation in Python

```python
from decimal import Decimal as D,localcontext
with localcontext() as ctx:
    ctx.prec=80; eta=D('0.2'); y=D('0.2')
    for k in range(1,10001):
        nxt=y-eta*y**3
        assert 0<nxt<y
        y=nxt
        if k in [1,40,100,1000,10000]:
            print(k,'y:',float(y),'k*y^2:',float(k*y*y))
    print('asymptotic scale:',float(1/(2*eta)))
```

## Exercise and explained answer

Run the cosine model from (0,0) with η=0.2 for thirty iterations. Gradient remains zero and excess remains two. Restart from (0,0.01): final y is approximately 1.821715 and excess approximately 0.751706. The experiment demonstrates a specific escape after an explicit perturbation, not global convergence of every nonconvex method.

For the quartic model start y=0.2 and run forty steps. It reaches approximately 0.155888, with gradient magnitude approximately 0.003788. Explain why slow motion here does not indicate a saddle, and why negative curvature, initialization and higher-order terms must be checked separately.

Further reading: [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
