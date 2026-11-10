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

## Exercise and explained answer

Keep λ=1,b=0.5,η=0.25 and increase K from one to eight. The inner state becomes 0.99609375, sensitivity −0.482421875 and hypergradient approximately −0.239326. Compare with exact −0.25. Explain why a small training residual helps but does not, without conditioning information, supply a universal bound on hypergradient error.

In research, specify whether the outer objective includes finite training, an exact optimum or an approximate stationary solution. Keep final test data separate from hyperparameter selection.

Further reading: [Pedregosa, Hyperparameter Optimization with Approximate Gradient](https://arxiv.org/abs/1602.02355).
