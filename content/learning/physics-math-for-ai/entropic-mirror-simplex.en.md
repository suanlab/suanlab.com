## Choose geometry suited to probability constraints

A probability vector belongs to the simplex Δ₃={q:qᵢ≥0,Σqᵢ=1}. Ordinary GD can leave this set, so projected GD returns to it using Euclidean distance. Mirror descent instead uses a convex generator to measure the cost of moving from the current point.

The lab minimizes the linear cost c·q, starting from q₀=(0.2,0.3,0.5). The exact optimum value is min cᵢ. A tie in minimum costs creates an optimal face rather than a unique vertex.

## Entropy gives multiplicative updates

For the negative-entropy generator, the simplex Bregman divergence is KL(q||q_old). An entropic mirror step solves

$$
q^+=\arg\min_{q\in\Delta_3}
\left[\eta c^Tq+D_{KL}(q\|q_{old})\right],
\qquad
q_i^+=\frac{q_{old,i}e^{-\eta c_i}}{\sum_jq_{old,j}e^{-\eta c_j}}.
$$

The KL orientation matters: the new variable is the first argument. Differentiate with a normalization multiplier to obtain the exponential formula. This is a proximal problem in entropy geometry, not Euclidean clipping followed by normalization.

With constant costs, repeated steps give qₖ,ᵢ∝q₀,ᵢe⁻ᵏηᶜⁱ. The implementation subtracts the largest log weight before exponentiating, preserving normalization and positive support throughout the controlled range.

## Compare Euclidean projection

Blue repeatedly applies Euclidean projection of q−ηc; orange repeatedly applies the entropic step. Euclidean projection finds a common threshold τ with qᵢ⁺=max(vᵢ−τ,0) and sum one. It can create exact zeros in finite steps. Positive entropic weights remain positive for finite steps, although they can approach zero in a limit.

The triangle displays probability coordinates q₂+q₃/2 and √3q₃/2. Equal drawing scale preserves that Euclidean embedding, not Fisher distance or KL geometry. Read the probability table to interpret the optimization variables.

## Work a distinct-cost and a tied-cost example

For c=(0,1,2),η=1 and four steps, mirror probabilities are approximately (0.972467,0.026717,0.000816), with objective gap approximately 0.028348. Projected GD has reached (1,0,0), with zero gap. This finite example does not make one method universally superior.

For c=(0,0,1), the first two mirror probabilities keep their initial ratio 2:3. As the third mass vanishes they approach (0.4,0.6,0). Entropic geometry and initialization select a particular point of the optimal face. Euclidean projection need not select the same point.

## A finite-time guarantee for changing bounded costs

Now allow a different full cost vector cₜ∈[0,1]³ each round. Predict with qₜ before observing that round's vector, then update qₜ₊₁,ᵢ∝qₜ,ᵢexp(−ηcₜ,ᵢ). Let Zₜ=Σᵢq₀,ᵢexp(−ηΣₛ₌₀ᵗ⁻¹cₛ,ᵢ). The log-normalizer increment is bounded by −ηqₜ·cₜ+η²/8. To see the constant, the second derivative of a centered log moment-generating function is a tilted variance, at most 1/4 for values in [0,1]. Integrating twice gives the remainder η²/8.

Sum these increments and lower-bound Z_T by one comparator term. For every fixed component j and η>0,

$$
\sum_{t=0}^{T-1}q_t^Tc_t-\sum_{t=0}^{T-1}c_{t,j}
\le\frac{\log(1/q_{0,j})}{\eta}+\frac{\eta T}{8}.
$$

With q₀=(0.2,0.3,0.5), use log5 to bound all comparator terms. Choosing η=√(8log5/T) bounds average regret by √(log5/(2T)). This is a full-information online guarantee against the best fixed component, not a per-round advantage or a bandit guarantee. For constant costs in [−2,2], rescale or replace the squared range factor accordingly.

| T | Average regret bound |
|---|---:|
| 4 | 0.448531 |
| 16 | 0.224265 |
| 64 | 0.112133 |

```python
from math import exp, log, sqrt, fsum
initial = (0.2, 0.3, 0.5)
cycle = ((0., 1., .5), (1., .5, 0.), (.5, 0., 1.))
for T in (4, 16, 64):
    eta = sqrt(8*log(5)/T)
    cumulative, loss = [0., 0., 0.], 0.
    for t in range(T):
        logits = [log(q)-eta*c for q, c in zip(initial, cumulative)]
        largest = max(logits)
        weights = [exp(x-largest) for x in logits]
        q = [w/fsum(weights) for w in weights]
        cost = cycle[t % 3]
        loss += fsum(p*c for p, c in zip(q, cost))
        cumulative = [s+c for s, c in zip(cumulative, cost)]
    regret = loss-min(cumulative)
    bound = log(5)/eta+eta*T/8
    assert regret <= bound+1e-12
    print(T, "regret:", round(regret, 6), "average bound:", round(bound/T, 6))
```

The separate program actually updates the probabilities using changing costs. It evaluates incurred loss before updating and compares cumulative losses, preserving the timing required by the proof. The interactive lab above remains its original constant-cost experiment.

![Actual average regret and the proved bound for separate full-information runs with horizon-dependent step sizes.](/assets/learning/variational-regret-en.svg)

## Exercise and limits

Set all costs equal. Every simplex point is optimal and the mirror distribution stays at its initial values. Then add the same constant to every cost: prove it cancels from normalized exponential weights. Explain why an initially zero component cannot be revived by this multiplicative formula, motivating the strictly positive initialization used here.

General mirror-descent convergence rates require assumptions on convexity, gradient bounds, generator geometry and step schedules. A linear three-component example demonstrates the update and its support behavior, rather than proving those general rates.

Further reading: [Nemirovski, Mirror Descent Tutorial](https://www2.isye.gatech.edu/~nemirovs/COLT2012Tut.pdf).
