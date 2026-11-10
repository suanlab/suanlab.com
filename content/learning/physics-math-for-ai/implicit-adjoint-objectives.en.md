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

## Exercise and explained answer

At λ=c=r=0 use eight adjoint steps. Approximate hypergradient is about −0.887555, error 0.237445 and upper bound 0.260932. Verify the bound contains the error. Then select the coupled example and check that direct sensitivity and the adjoint give the same exact derivative.

This certificate assumes an exact state and exact model operators. A numerical PDE with an approximate state, discretization error or an approximate Hessian requires those additional errors to be accounted for. A successful adjoint solve alone cannot certify the entire scientific model.

Further reading: [MIT, Matrix Calculus and Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
