## Linearize the learned function locally

Near initialization, a network function can be linearized using J₀(x)=∂fθ(x)/∂θ. This lab trains both the original network and its frozen-Jacobian model from the same initialization.

$$
f_\theta(x)=\frac1{\sqrt m}\sum_{j=1}^m a_j\tanh(w_jx+b_j),\qquad f_{\rm lin}(x)=f_0(x)+J_0(x)\delta.
$$

All a,w,b parameters are trained, giving three coordinates per unit. Removing the 1/√m scaling changes the kernel and dynamics.

## Empirical NTK and residual dynamics

For training Jacobian J, the empirical NTK is JJᵀ. With loss ∑rᵢ²/(2n), the frozen model has exact discrete dynamics

$$
K_0=J_0J_0^T,\qquad r_{t+1}=(I-\eta K_0/n)r_t.
$$

The nonlinear network changes its Jacobian as parameters move. A single small-step discrepancy is typically second order in step size, but accumulation and larger steps require separate analysis.

## Inspect the finite experiment

Inputs are −1,0,1 and labels 1,0,1. At step zero, both models match exactly and kernel drift is zero. After training, changing Jacobians create differences. Blue is nonlinear training, orange frozen-Jacobian training and dots the labels.

The table reports both losses, maximum training prediction gap and Frobenius norm of Kₜ−K₀. These measurements concern three finite inputs. Initialization is a reproducible bounded pseudorandom example, not an implementation of every limiting assumption in the original theorem.

## An infinite-width conclusion needs hypotheses

Check width limits, parameterization, initialization law, activation regularity, step size or gradient flow, time horizon and data assumptions. A width-eight plot proves neither a constant-kernel limit nor generalization. Small measured drift does not automatically bound error at every input.

Fixed-kernel analysis can explain some regimes while missing substantial feature learning in others.

## Exercises with explanations

1. What if η=0? **Neither model changes and drift stays zero.** More iterations alone do not create learning.
2. Parameter count at m=2? **Six**, from two a,w,b triples.
3. Is every step size stable even for a frozen kernel? **No.** Check the spectrum of K₀ and sample-count scaling; nonlinear training needs additional care.

Further reading: [Jacot, Gabriel and Hongler, Neural Tangent Kernel](https://arxiv.org/abs/1806.07572).
