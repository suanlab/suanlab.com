## A differential is not yet a gradient vector

A functional derivative maps a perturbation h to a scalar δJ[h]. A gradient is its Riesz representative in a chosen Hilbert inner product. Changing that inner product can change the gradient and optimization trajectory without changing the objective or its minimizer.

The lab uses zero-endpoint functions and the fit-plus-smoothness energy

$$
J[u]=\frac12\int_0^1\bigl((u-f)^2+\kappa(u')^2\bigr)dt,
\quad f=\sin\pi t+\tfrac12\sin(n\pi t).
$$

Only the two displayed sine modes are optimized. This finite Galerkin space makes every calculation explicit and avoids treating a sampled plot as an infinite-dimensional convergence theorem.

## Derive the differential before choosing a metric

For u=a₁sin πt+a₂sin nπt, define frequencies k=(π,nπ), target coefficients fᵢ=(1,0.5) and Aᵢ=1+κkᵢ². Orthogonality gives coefficient differential ∂J/∂aᵢ=(Aᵢaᵢ−fᵢ)/2.

The L2 mass matrix is I/2. Therefore the function's L2 gradient coefficients are gᵢ=Aᵢaᵢ−fᵢ, not the coefficient differential itself. The distinction is a Riesz-map calculation, not a discrepancy between two derivative implementations.

## Choose L2 or H1 geometry

The H1-type inner product is ∫hv+ℓ²∫h′v′. Its modal mass factors are Mᵢ/2 with Mᵢ=1+ℓ²kᵢ². Thus

$$
(\operatorname{grad}_M J)_i=\frac{A_i a_i-f_i}{M_i},
\qquad a_i^+=a_i-\eta\frac{A_i a_i-f_i}{M_i},
\qquad a_i^*=\frac{f_i}{A_i}.
$$

The optimum is independent of metric. Higher frequencies receive stronger H1 weighting, acting like a preconditioner for the update. Setting ℓ=0 recovers L2 exactly in this finite space.

For a sufficiently regular unrestricted function, the strong L2 gradient is u−f−κu″ with appropriate boundary treatment. An arbitrary H₀¹ function need not have an L2 second derivative, so an L2 gradient need not exist there. The weak differential and an H1 Riesz representation can still be meaningful. Function spaces and regularity cannot be omitted when leaving the finite model.

## Stability and descent use the selected metric

The modal error factors are 1−ηAᵢ/Mᵢ. The readout also verifies δJ[−grad_M J]=−‖grad_M J‖²_M, where the squared norm is Σgᵢ²/(2Mᵢ). This directional statement does not guarantee decrease for an arbitrarily large finite step.

At n=4,κ=0.1,η=0.3, L2 has a high-mode factor about −4.03741 and diverges. With H1 length 0.2 the worst factor magnitude is about 0.572631. Both still have the same optimal coefficients, approximately (0.503281,0.029777).

## Exercise and explained answer

Run twelve H1 steps: objective excess is about 1.94×10⁻⁷. Switch only the metric to L2 and observe the high-frequency coefficient grow. Then set H1 length to zero and reproduce the L2 result. Explain why changing geometry improved this iteration without changing the statistical fit criterion, regularization parameter or minimizing function.

Further reading: [MIT, Function Inner Products and Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
