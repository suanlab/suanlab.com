## An optimization solution can define another objective

Let an inner variational energy be E(y,λ)=yᵀH(λ)y/2−bᵀy, with b=(1,−1) and H=[[1+λ,c],[c,2+λ]]. The controlled range keeps H symmetric positive definite, so the unique state y solves Hy=b. This is a two-dimensional Galerkin analogue of a variational state equation.

The outer objective is V(y)=‖y−t‖²/2 with t=(r,0). Changing λ changes the state and therefore validation loss. The lab solves the state analytically; the iteration control affects only the adjoint calculation.

## Direct sensitivity and the adjoint

Because ∂H/∂λ=I and b is fixed, differentiating stationarity gives

$$
H\frac{dy}{d\lambda}=-y,\qquad
\frac{dV}{d\lambda}=(y-t)^T\frac{dy}{d\lambda}.
$$

Instead solve Hᵀv=y−t. Then dV/dλ=−vᵀy. In a large problem with many hyperparameters and one scalar outer loss, a transpose solve can avoid explicitly storing a full state sensitivity matrix.

For a general state equation g(y,λ)=0 and outer objective V(y,λ), the formula also includes direct dependence: dV/dλ=V_λ−vᵀg_λ, where g_yᵀv=V_y. The simplified expression here relies on the model's specified dependencies.

## Approximate only the adjoint solve

Starting from v₀=0, the experiment takes actual Richardson steps v⁺=v+(y−t−Hv)/6. Its step size is stable throughout the displayed SPD range. Orange plots −v_Kᵀy; blue plots the exact-adjoint derivative across λ.

A small or zero approximate derivative need not mean the outer objective is stationary. At λ=c=r=0, y=(1,−0.5), exact adjoint v=(1,−0.25) and exact hypergradient −1.125. Before any adjoint work, the approximate derivative is zero simply because v₀ is zero.

## Certify error with residual and conditioning

Let residual r_adj=y−t−Hᵀv_K. Then v*−v_K=H⁻ᵀr_adj. Since the smallest eigenvalue is positive,

$$
|(-v_K^Ty)-(-v^{*T}y)|
\le\frac{\|y\|\,\|r_{adj}\|}{\lambda_{min}(H)}.
$$

The readout compares actual error with this computable upper bound. Residual alone is insufficient without conditioning information. For nonsymmetric systems, a singular-value or inverse-operator bound would replace the SPD eigenvalue argument.

## Include state error in the hypergradient certificate

Keep H exact and SPD with λmin(H)≥m>0. Let the approximate state be ỹ, with state residual s=b−Hỹ, and let the approximate adjoint be ṽ, with residual ρ=ỹ−t−Hṽ. Define εy=‖s‖/m, so ‖y−ỹ‖≤εy. The exact adjoint v satisfies Hv=y−t; subtracting equations gives ‖v−ṽ‖≤(εy+‖ρ‖)/m. Split the error in the product −vᵀy to obtain

$$
|{-\tilde v^T\tilde y}-({-v^Ty})|
\le(\|\tilde y\|+\varepsilon_y)
\frac{\varepsilon_y+\|\rho\|}{m}
+\|\tilde v\|\varepsilon_y.
$$

Unlike the earlier exact-state certificate, this includes both residuals. All norms in the derivation are Euclidean. It still assumes exact H, fixed b,t and ∂H/∂λ=I. Operator approximation, model error or direct outer dependence require further terms.

For H=diag(1,2), b=(1,−1),t=0, perturb the state to ỹ=(1+ε,−1/2) and solve its adjoint exactly. Then ρ=0 but the hypergradient error is |2ε+ε²|. A zero adjoint residual does not eliminate a state error. Replacing Euclidean vector norms by their L1 upper bounds gives these conservative rational certificates:

| ε | Actual error | Bound |
|---|---:|---:|
| 0 | 0 | 0 |
| 1/100 | 201/10000 | 139/5000 |
| 1/10 | 21/100 | 61/200 |

```python
from fractions import Fraction as F
true_hyper = -F(9, 8)
for eps in (F(0), F(1, 100), F(1, 10)):
    state = (1+eps, -F(1, 2))
    adjoint = (state[0], state[1]/2)
    state_error_bound = abs(eps)  # m=1; state residual is (-eps,0).
    adjoint_residual = (state[0]-adjoint[0], state[1]-2*adjoint[1])
    assert adjoint_residual == (0, 0)
    approximate = -sum(v*y for v, y in zip(adjoint, state))
    error = abs(approximate-true_hyper)
    # L1 upper bounds on the Euclidean norms in the proved certificate.
    bound = ((sum(map(abs, state))+state_error_bound)*state_error_bound
             +sum(map(abs, adjoint))*state_error_bound)
    assert error == abs(2*eps+eps*eps) and error <= bound
    print(eps, error, bound)
```

This additional program perturbs the state explicitly. The interactive lab above continues to use its analytic state and to vary only Richardson adjoint work. Do not interpret an exact adjoint for an approximate state as the exact derivative of the true inner optimum.

## Exercise and explained answer

At λ=c=r=0 use eight adjoint steps. Approximate hypergradient is about −0.887555, error 0.237445 and upper bound 0.260932. Verify the bound contains the error. Then select the coupled example and check that direct sensitivity and the adjoint give the same exact derivative.

This certificate assumes an exact state and exact model operators. A numerical PDE with an approximate state, discretization error or an approximate Hessian requires those additional errors to be accounted for. A successful adjoint solve alone cannot certify the entire scientific model.

Further reading: [MIT, Matrix Calculus and Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
