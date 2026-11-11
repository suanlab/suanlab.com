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

## Derive the spectral stability condition and its nullspace limit

For a fixed training Jacobian J₀ and n inputs, parameter GD on ‖r‖²/(2n) gives δ_{t+1}=δ_t−ηJ₀ᵀr_t/n. Multiplying by J₀ proves the residual recurrence. Since K₀ is PSD, diagonalize it with eigenvalues μ_j≥0. Each residual coordinate is

$$
r_{t,j}=\left(1-\frac{\eta\mu_j}{n}\right)^t r_{0,j}.
$$

Positive-eigenvalue components contract when 0<η<2n/μ_max. At the upper boundary, a top component oscillates without decreasing its magnitude. Above it, that component grows. Zero-eigenvalue components stay unchanged: even perfect convergence in range K₀ need not give zero training loss. Under the strict stability condition the limiting loss is ‖P_ker K₀ r₀‖²/(2n).

The following separate n=2 example uses J₀=diag(1,0) and r₀=(1,1). It is not a new control range for the live tanh network, whose allowed rate ends at .5.

| η | First residual after 4 steps | Loss after 4 steps |
|---|---|---|
| 1 | 1/16 | .250977 |
| 4 | 1 | .5 |
| 5 | 81/16 | 6.657227 |

![One frozen-kernel residual contracts, oscillates at the stability boundary, or grows above it; a nullspace residual stays one in all three cases.](/assets/learning/approximation-ntk-stability-en.svg)

## State what the infinite-width theorem adds

The original paper's fixed-depth setting uses its specified 1/√width parameterization and independent Gaussian initialization. Its training theorem assumes a Lipschitz, twice differentiable activation with bounded second derivative and a bounded finite-time training-direction integral in probability. Hidden widths are sent to infinity sequentially in the appendix statement, giving uniform-in-time kernel convergence on that fixed horizon. These are assumptions of that result, not a universal definition of all NTK limits.

The finite lab uses tanh and bounded seeded initialization, and compares discrete GD trajectories. The finite recurrence above is exact without an infinite-width limit, while transfer to the nonlinear model needs control of changing Jacobians and Taylor remainders. Smoothness alone does not prove globally accurate frozen features or generalization.

**Exercise:** If K₀ is singular, can reducing η eliminate the nullspace residual? **Explanation:** No. J₀ᵀ maps that residual to zero, so frozen-feature GD cannot update it.

```python
from fractions import Fraction as F

# Separate frozen-Jacobian example: J=diag(1,0), K=diag(1,0), n=2.
# Loss is sum r_i^2/(2n); the second residual lies in ker K.
for eta in [F(1),F(4),F(5)]:
    r=[F(1),F(1)]
    for t in range(4):
        r=[r[0]-eta*r[0]/2,r[1]]
    assert r[0]==(1-eta/2)**4 and r[1]==1
    loss=sum(v*v for v in r)/4
    print(str(eta),str(r[0]),round(float(loss),6))
# Positive-eigenvalue stability: 0<eta<4. At 4 the component oscillates;
# above 4 it grows. No step size removes a nonzero nullspace residual.
```

## Exercises with explanations

1. What if η=0? **Neither model changes and drift stays zero.** More iterations alone do not create learning.
2. Parameter count at m=2? **Six**, from two a,w,b triples.
3. Is every step size stable even for a frozen kernel? **No.** Check the spectrum of K₀ and sample-count scaling; nonlinear training needs additional care.

Further reading: [Jacot, Gabriel and Hongler, Neural Tangent Kernel](https://arxiv.org/abs/1806.07572).
