## Start with an infinite diagonal

The Hilbert space ℓ2 consists of square-summable sequences. Standard basis eₖ has one nonzero coordinate. For Teₖ=λₖeₖ, a single basis input reveals amplification |λₖ|.

$$
\|Tx\|_2^2=\sum_k|\lambda_k|^2|x_k|^2.
$$

Finite values at individual coordinates do not supply a common bound. This diagonal defines a bounded map on all ℓ2 exactly when supₖ|λₖ| is finite; that supremum is its operator norm.

## Bounded is not compact

Identity has norm one. Yet its basis images stay distance √2 apart and have no convergent subsequence. A compact operator must turn every bounded sequence into an image sequence with a convergent subsequence.

For λₖ=k⁻ᵅ with α>0, truncate after N coordinates:

$$
\|T-T_N\|=\sup_{k>N}k^{-\alpha}=(N+1)^{-\alpha}.
$$

The tail tends to zero. Finite-rank operators are compact and their operator-norm limits remain compact, proving compactness here. Merely observing several small basis images would not prove a general compactness claim.

## An unbounded map needs a domain

For λₖ=k, image norms of eₖ grow without bound. The sequence xₖ=1/k belongs to ℓ2 but its image is constant and does not. Use the natural dense domain

$$
D(T)=\left\{x\in\ell^2:\sum k^2|x_k|^2<\infty\right\}.
$$

Finite-support sequences lie in this domain and make it dense. An unbounded operator with a specified domain differs from a bounded map on all ℓ2. Differential operators require similar care.

## Read the experiment

At α=1, N=4, the infinite norm is one and tail norm is .2. In growth mode, finite truncation norm is N, giving four at N=4. Displayed ∞ denotes mathematical unboundedness, not floating-point overflow. Blue shows multipliers, orange truncation. The alpha control is used only in decay mode.

For neural operators, finite-matrix stability implies continuous stability only with suitable norms, domains and resolution-independent bounds.

## A compactness criterion for diagonal operators

A bounded diagonal T=diag(λ_k) is compact exactly when λ_k→0. For sufficiency, finite-rank truncations have norm error supk>N|λ_k|→0. Their norm limit is compact because the unit-ball image can first be approximated within ε by a bounded finite-dimensional image, then covered by a finite ε-net. The image is totally bounded; completeness makes its closure compact.

For necessity, if the multipliers do not tend to zero, some ε>0 and infinite subsequence satisfy |λ_kj|≥ε. Distinct basis images are separated by sqrt(|λ_ki|²+|λ_kj|²)≥√2ε and have no convergent subsequence. Small values among sixteen displayed coordinates cannot replace this infinite-tail condition.

| λ_k | Bounded on all ℓ² | Compactness | Rank-N tail norm |
|---|---|---|---|
| 1 | Yes | No | 1 |
| 1/k | Yes | Yes | 1/(N+1) |
| k | No | Not a bounded compact map | ∞ |

## An unbounded operator can have a closed graph

Consider Tx=(k x_k) on its natural domain. If x⁽ⁿ⁾→x and Tx⁽ⁿ⁾→y in ℓ² norm, coordinate convergence gives y_k=kx_k. Since y∈ℓ², x belongs to D(T) and Tx=y. The graph is closed. This does not assert that D(T) itself is a closed subset under the original ℓ² norm.

Check the adjoint domain too. For y∈D(T*), testing finitely supported x forces (T*y)_k=k y_k. This output must lie in ℓ², requiring Σk²|y_k|²<∞. Conversely that condition and Cauchy–Schwarz give ⟨Tx,y⟩=⟨x,(k y_k)⟩. Thus D(T*)=D(T) and T*=T: the natural diagonal operator is self-adjoint. Real coefficients alone were not the proof.

The graph norm ‖x‖_T²=Σ(1+k²)|x_k|² is a weighted ℓ² norm making D(T) complete. It gives ‖Tx‖₂≤‖x‖_T, while T is unbounded under the original ℓ² norm. Specify operator, norm and domain together.

**Exercise:** Does self-adjointness of every finite truncation automatically prove self-adjointness of the original operator? **Explanation:** No. The conclusion here used equality of adjoint domains. Boundary conditions can be decisive for differential operators.

## Exercises with explanations

1. Tail norm for α=2, N=3? **1/16:** the first omitted multiplier is largest.
2. Identity truncation tail? **Always one**, attained by eₙ₊₁.
3. Why do compact finite identity truncations not prove identity compact? **They converge strongly, not in operator norm.**

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
