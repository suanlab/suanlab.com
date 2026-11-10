## Small per-coordinate costs multiply

A d-dimensional grid with q candidates per coordinate contains qᵈ points. Already q=4,d=16 gives 4,294,967,296 points. A central L∞ neighborhood of radius .1 under uniform [0,1]ᵈ has mass .2ᵈ. These are specific grid, metric and distribution models, not universal lower bounds for every algorithm.

Counts can exceed floating-point integer precision. Readouts show log₁₀(qᵈ); the table computes exact decimal strings using BigInt, including 10³².

## Ambient dimension versus spectral structure

A separate designed covariance spectrum has λᵢ=ρⁱ⁻¹. It is not fitted to observed data. At ρ=1 all directions have equal scale; at ρ=0 only the first direction is active. Smaller decay ratios concentrate energy near the leading directions.

## Distinguish effective-dimension definitions

Participation rank is (Σλ)²/Σλ². Trace divided by maximum eigenvalue is covariance effective rank, corresponding to stable rank of a data matrix whose Gram spectrum is λ. Ridge effective dimension is Σλ/(λ+τ) for τ>0. These are different quantities.

$$
r_{\rm part}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2},\quad
r_{\rm eff}=\frac{\sum_i\lambda_i}{\max_i\lambda_i},\quad
d_{\rm eff}(\tau)=\sum_i\frac{\lambda_i}{\lambda_i+\tau}.
$$

Do not call trace/λmax the stable rank of the covariance matrix itself: its stable rank is Σλ²/λmax². The screen labels each convention explicitly.

## Different structure at the same ambient dimension

At d=16,ρ=.5, participation rank is about three. At ρ=1 it is sixteen. Grid count qᵈ remains unchanged because it does not depend on ρ. An algorithm must exploit structure to benefit from a smaller effective dimension.

Increasing τ downweights weak directions and decreases ridge dimension. Compare such numbers only after specifying the estimator and scale. With one active direction, ridge dimension is 1/(1+τ), which can be less than one.

## Structure is not a complete sample-complexity theorem

Small participation rank does not universally determine required observations. Regression, classification and recovery also depend on observation models, noise, target accuracy and identifiability. The plot compares grid growth with spectral summaries as an introduction.

## Exercises with explanations

1. What happens when q doubles? **Grid count multiplies by 2ᵈ.**
2. What is participation rank at ρ=0? **One.**
3. Must all effective dimensions agree? **No: definitions and regularization matter.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
