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

## When the preconditioning remains useful as resolution grows

Extend the modal argument to frequency ω=nπ. For κ>0 and ℓ>0, set r=κ/ℓ² and s=ℓ²ω². The generalized curvature is a weighted average:

$$
\frac{1+\kappa\omega^2}{1+\ell^2\omega^2}
=\frac{1+rs}{1+s}\in[m,L],\qquad
m=\min(1,r)>0,\quad L=\max(1,r).
$$

For a fixed 0<η<2/L, every modal error factor has magnitude at most ρ=max(|1−ηm|,|1−ηL|)<1. Summing the modal energy gaps gives an energy contraction of at most ρ² per step, uniformly in the number of retained modes. Extending the sum to an infinite expansion also requires finite initial energy. This is a spectral argument, not evidence from one sampled two-mode plot.

If ℓ²=κ, all generalized curvatures are one. With η=1 the exact metric step solves the quadratic in one step. Applying this metric inverse is itself an elliptic solve; the observation does not promise a free algorithm. The program below uses η=1/4, inside the lab's step range.

Two contrasting limits explain the conditions. For L2 with κ>0, curvatures grow as ω² and no positive fixed step is stable for arbitrarily high modes. For κ=0 but ℓ>0, generalized curvatures tend to zero, so each finite truncation can contract while there is no common contraction factor strictly below one over all frequencies.

| Setting | High-frequency curvature | Uniform conclusion |
|---|---|---|
| κ>0, ℓ=0 | grows without bound | no fixed stable step |
| κ=ℓ²>0 | 1 | factor 1−η |
| κ=0, ℓ>0 | tends to 0 | no strict common factor |

```python
from math import pi
eta, kappa = 0.25, 0.1
for n in (2, 8, 128):
    omega_squared = (n*pi)**2
    A = 1+kappa*omega_squared
    matched_metric = A  # ell^2=kappa, avoiding a rounded sqrt-and-square.
    matched_factor = 1-eta*A/matched_metric
    l2_factor = 1-eta*A
    unregularized_factor = 1-eta/matched_metric
    assert matched_factor == 0.75
    print(n, "L2:", round(l2_factor, 6), "matched:", matched_factor,
          "kappa=0,H1:", round(unregularized_factor, 6))
```

The larger frequencies are a separate resolution study, beyond the lab control's range 2–8. The minimizer is unchanged by the metric; objective regularization κ and metric length ℓ play distinct roles even when deliberately matched.

![Generalized curvatures and absolute modal error factors compare L2, matched H1, and unregularized H1 as frequency increases.](/assets/learning/variational-metric-en.svg)

## Exercise and explained answer

Run twelve H1 steps: objective excess is about 1.94×10⁻⁷. Switch only the metric to L2 and observe the high-frequency coefficient grow. Then set H1 length to zero and reproduce the L2 result. Explain why changing geometry improved this iteration without changing the statistical fit criterion, regularization parameter or minimizing function.

Further reading: [MIT, Function Inner Products and Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
