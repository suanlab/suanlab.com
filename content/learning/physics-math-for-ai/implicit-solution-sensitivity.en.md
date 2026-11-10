## Differentiate a solution equation

Suppose a parameter θ determines w through an equation F(w,θ)=0. Implicit differentiation uses this equation rather than differentiating every iteration of the numerical solver. The lab studies

$$
F(w,\theta)=w^3+\delta w-\theta=0,\qquad\delta\ge0.
$$

For δ>0 the left side is strictly increasing in w with positive derivative everywhere. For δ=0 the cubic is still strictly increasing, so every θ has a unique real solution. Uniqueness therefore holds even in a case where differentiability of the inverse can fail.

The numerical root is computed from a bounded scalar problem; the returned residual checks the equation. A small residual by itself does not determine sensitivity accuracy when the Jacobian is near singular.

## Check the implicit-function hypothesis

Differentiating F=0 gives

$$
(3w^2+\delta)\frac{dw}{d\theta}=1,
\qquad\frac{dw}{d\theta}=\frac{1}{3w^2+\delta}
\quad\text{when }3w^2+\delta\ne0.
$$

For outer loss V=(w−b)²/2, its hypergradient is (w−b)/(3w²+δ). The target b affects this outer quantity but not the root or inverse sensitivity.

At δ=1,θ=0 the solution is zero and sensitivity one. With b=1 the outer derivative is −1. At δ=0,θ=0.125 the solution is 0.5 and sensitivity 4/3. These two examples satisfy the nonzero-Jacobian condition.

## A unique solution can have a singular inverse

At δ=θ=0, w=0 but F_w=0. The inverse is the cube-root function, whose derivative is not finite at zero. The lab withholds a finite implicit derivative; it does not divide by zero and label the result a valid derivative certificate.

A centered difference with h=0.0001 still returns approximately 464.159. That finite number is h⁻²ᐟ³ for the cube root. Shrinking h makes it grow, so one finite numerical quotient does not establish differentiability. At δ>0 but extremely small, the derivative can be mathematically finite while exceeding machine-number range. This separate numerical limitation is explicitly distinguished from an exactly singular Jacobian.

## Read the forward map and solve linear systems

Blue plots θ=w³+δw against w. Orange is its tangent at the solved point. The displayed inverse sensitivity is the reciprocal slope, not the slope itself. Near a horizontal tangent, small changes in θ can require large changes in w. The plot does not draw an enormous inverse tangent as if it were a well-conditioned map.

For vector equations, the corresponding relation is J_w dw/dθ=−J_θ. One usually solves a linear system rather than forming an explicit inverse. With a scalar outer objective an adjoint transpose solve can avoid storing the full solution Jacobian. Nonsingularity and numerical conditioning still matter; an approximate root or iterative linear solve introduces additional error.

## Exercise and explained answer

Compare the three presets and verify their roots directly. At the singular cubic origin, halve the finite-difference step on paper and predict growth by factor 2²ᐟ³. Explain why solver convergence, root uniqueness and differentiable parameter dependence are three different claims. Finally set θ=0 and a small positive δ: the exact root remains zero while sensitivity becomes 1/δ.

Further reading: [Pedregosa, Hyperparameter Optimization with Approximate Gradient](https://arxiv.org/abs/1602.02355).
