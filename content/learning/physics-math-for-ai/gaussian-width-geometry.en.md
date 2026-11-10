## Does ambient dimension determine complexity?

A set restricted to axes differs from a set containing every direction in the same Rᵈ. Gaussian width averages the largest projection in a standard Gaussian direction. This chapter uses the following convention; widths of difference sets in other references can have different factors.

$$
w(T)=E\sup_{t\in T}\langle g,t\rangle,\qquad g\sim N(0,I_d).
$$

## Axis endpoints versus the whole sphere

For T={±Reᵢ}, the supremum is R maxᵢ|gᵢ|. The radius-R sphere can align with g, giving R||g||₂. Ambient dimension and radius match, but directional geometry differs.

$$
w(T_{\rm axes})=R E\max_i|g_i|,\qquad w(T_{\rm sphere})=R E\|g\|_2.
$$

## Calculate the expectations

The survival function of the largest absolute coordinate is 1−P(|G|≤t)ᵈ. Integrating it gives the nonnegative variable's mean. The implementation integrates over [0,10] using a numerical normal CDF. Omitted tails are very small, but values are explicitly labeled approximate.

Sphere width uses the chi mean, starting from √(2/π) at d=1 and √(π/2) at d=2, then applying a two-dimension recurrence. At dimension one the sets are identical, so their theoretical widths coincide. A maximum from one seed is not substituted for an expectation.

## Bounds and growth

Jensen gives E||g||≤√E||g||²=√d. A finite Gaussian-maximum bound gives E max|gᵢ|≤√(2ln(2d)). Axis width therefore grows on a √log d scale, while sphere width grows on a √d scale; these are not exact finite-dimensional equalities.

Blue and orange compare both sets across dimensions. Set selection changes the selected value and dot, while preserving both curves. Multiplying radius by two doubles each width.

## Research scope

Widths can enter random-measurement, estimation-error and sample-complexity results. This does not make w² a universal required sample count. Observation models, noise, accuracy and localization assumptions matter. Covering numbers are related geometric quantities, not interchangeable names for width.

## Exercises with explanations

1. Is g uniform on T? **No: it is standard Gaussian.**
2. What happens when radius doubles? **Width doubles.**
3. Does equal ambient dimension imply equal width? **No: directional structure matters.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
