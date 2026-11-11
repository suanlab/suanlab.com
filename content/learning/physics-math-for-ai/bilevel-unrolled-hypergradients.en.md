## Training and validation define two levels

A hyperparameter can affect a validation objective through the parameters produced by training. This dependence is the central feature of bilevel optimization. The lab uses inner training loss L(w,λ)=(w−2)²/2+λw²/2 and outer validation loss V(w)=(w−b)²/2. Lambda is nonnegative, while b is a validation target distinct from the training target two.

The exact inner optimizer and its sensitivity are

$$
w^*(\lambda)=\frac{2}{1+\lambda},\qquad
\frac{dw^*}{d\lambda}=-\frac{2}{(1+\lambda)^2},\qquad
\frac{dV(w^*)}{d\lambda}=(w^*-b)\frac{dw^*}{d\lambda}.
$$

These formulas differentiate the solution of the inner problem. They do not automatically describe a partially trained model.

## Differentiate the actual finite training procedure

Start at w₀=0, independent of λ, and perform K GD steps with fixed η. Let sₖ=dwₖ/dλ. Differentiating the update, including its dependence on the previous w, gives

$$
w_{k+1}=(1-\eta(1+\lambda))w_k+2\eta,
\quad s_{k+1}=(1-\eta(1+\lambda))s_k-\eta w_k,
\quad s_0=0.
$$

The finite-step hypergradient is (w_K−b)s_K. The old wₖ must be used in the sensitivity recurrence. Updating w first and then substituting its new value changes the differentiated computation.

Blue plots the exact-solution hypergradient across λ. Orange plots the derivative of the actual K-step procedure. Both are valid derivatives of their respective outer objectives; their difference is not necessarily an autodifferentiation bug.

## A zero hypergradient can reflect truncated training

At λ=1,b=0.5,η=0.25,K=1, the first update is w₁=0.5. Because w₀=0, its regularization gradient λw₀ vanishes. The first update is independent of λ, so s₁=0 and the unrolled hypergradient is zero.

The exact optimizer is one, its sensitivity −0.5, and its validation hypergradient −0.25. A zero finite-step derivative therefore does not certify an optimal hyperparameter for the exact-solution objective. K=0 has the same issue for an even simpler reason: no training dependence has been introduced.

## Accuracy requires a stable inner procedure

The inner linear contraction factor is q=1−η(1+λ). For |q|<1, w_K approaches w*, and the differentiated recurrence also approaches its exact sensitivity. At unstable choices, longer unrolling can amplify both state and derivative errors. Merely increasing K does not repair an unstable step size.

The plotted λ range includes points with different stability properties when η is large. Check the inner residual and the derivative discrepancy rather than assuming every part of the orange curve is an approximation of equal quality.

## State agreement does not imply derivative agreement

![State and sensitivity errors converge at different rates; a one-step exact inner state can still have a different derivative from the exact solution map.](/assets/learning/optimization-hypergradient-en.svg)

Write a=1+λ and q=1−ηa. Solving the training recurrence and differentiating its finite sum gives

$$
w_K=\frac2a(1-q^K),\qquad
s_K=-\frac2{a^2}(1-q^K)+\frac{2\eta K}{a}q^{K-1}.
$$

For K=0 use s₀=0 separately. The sensitivity error contains both qᴷ and Kqᴷ⁻¹ terms. A bound on the state error alone does not bound it without additional information about the differentiated procedure.

At λ=1,η=1/2, the first step reaches the exact optimizer w₁=1 and has zero training residual, but s₁=0 while the exact solution sensitivity is −1/2. With validation target 1/2, the two hypergradients are zero and −1/4. The first-step function is constant in λ; it only coincides with the optimizer at this parameter. A second step repairs the derivative at this point.

| K at λ=1,η=1/2 | State wK | Sensitivity sK |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 1 | -1/2 |

At a fixed solved state, H⁻¹=ηΣⱼ≥₀(I−ηH)ʲ when the spectral radius is below one. A truncated Neumann inverse has a geometric error bound under an operator-norm contraction, but actual training unrolls additionally carry state-error forcing. These are different approximations.

The exact-solution readout now retains its analytic outer residual. With λ=10⁻²⁰,target=2, the true hypergradient is approximately 4×10⁻²⁰ even though the displayed optimizer rounds to two.

## Reproduce the calculation in Python

```python
from fractions import Fraction as Q
lam,eta,target=Q(1),Q(1,2),Q(1,2)
w=s=Q(0)
for k in range(3):
    print(k,'w:',w,'s:',s,'hyper:',(w-target)*s)
    old=w
    w=(1-eta*(1+lam))*w+2*eta
    s=(1-eta*(1+lam))*s-eta*old
tiny=Q(1,10**20)
exact_hyper=4*tiny/(1+tiny)**3
assert exact_hyper>0
print('tiny-lambda exact hypergradient:',float(exact_hyper))
```

## Exercise and explained answer

Keep λ=1,b=0.5,η=0.25 and increase K from one to eight. The inner state becomes 0.99609375, sensitivity −0.482421875 and hypergradient approximately −0.239326. Compare with exact −0.25. Explain why a small training residual helps but does not, without conditioning information, supply a universal bound on hypergradient error.

In research, specify whether the outer objective includes finite training, an exact optimum or an approximate stationary solution. Keep final test data separate from hyperparameter selection.

Further reading: [Pedregosa, Hyperparameter Optimization with Approximate Gradient](https://arxiv.org/abs/1602.02355).
