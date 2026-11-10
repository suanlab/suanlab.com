## Stability is a conditional quantitative statement

A small visual movement of points needs a specified distance, correspondence and filtration before it can support a stability claim. We compare a labeled square P with Q obtained by moving vertex two by scale·(δ,−δ). All other vertices stay fixed. The maximum labeled displacement is η=√2·scale·|δ|.

Pairwise distances change by at most the sum of the two endpoint movements. Let D be the actual maximum change over all pairs. Then

$$
D=\max_{i,j}|d_P(i,j)-d_Q(i,j)|\le2\eta.
$$

The factor-two displacement bound is convenient but can be loose. The readout distinguishes it from the actual pair-distance distortion.

## A common labeling gives filtration inclusions

Both Rips filtrations use the same abstract vertex set. A simplex's filtration value is the maximum of its edge distances, so it also changes by at most D. Hence identity-on-vertices maps give inclusions in both directions with parameter shift D:

$$
\mathrm{VR}_\varepsilon(P)\subseteq\mathrm{VR}_{\varepsilon+D}(Q),\qquad
\mathrm{VR}_\varepsilon(Q)\subseteq\mathrm{VR}_{\varepsilon+D}(P).
$$

For these finite filtrations over a field, persistence stability bounds diagram bottleneck distance by D. This is an explicit common-complex correspondence argument. It is not an unqualified application of a function sup-norm theorem to arbitrarily unrelated point clouds.

## Compute the bottleneck distance, including the diagonal

Persistence diagrams contain birth/death pairs with multiplicity. A bottleneck matching minimizes the largest L∞ pair cost. Features may match the diagonal, with cost half their lifetime:

$$
\|(b,d)-(b',d')\|_\infty
=\max(|b-b'|,|d-d'|),\qquad
\operatorname{dist}_\infty((b,d),\Delta)=\frac{d-b}{2}.
$$

The implementation solves this finite matching exactly using assignment states, including diagonal copies. Essential H₀ classes have the same birth zero and match at zero cost; finite H₀ and H₁ distances are calculated separately. At δ=0.25 and unit scale, matching both H₁ points to the diagonal is better than directly matching them. The resulting H₁ distance is (√2−1)/2≈0.2071.

## Close diagrams need not have identical threshold Betti numbers

For δ=0.1 and ε=1.05, the reference square already has its cycle, while the perturbed quadrilateral has not completed the final side. Thus β₁ values are one and zero even though H₁ bottleneck distance is only about 0.1045. Stable birth/death locations do not make an indicator evaluated near a birth continuous.

## Exercise and limits

Set δ=0: distortion, both diagram distances and labeled displacement vanish. Then double scale while fixing δ and doubling ε. Every distance bound and finite persistence coordinate doubles, while the rescaled Betti pattern is unchanged. Explain why a long feature, with lifetime greater than twice a proposed bottleneck error, cannot disappear into the diagonal within that error.

For real data, document the metric, filtration convention, coefficients, noise model and any correspondence or sampling theorem. Diagram stability controls perturbation of a chosen construction. It does not automatically prove the topology or causal meaning of the unknown system that generated the observations.

Further reading: [Cohen-Steiner, Edelsbrunner and Harer, Stability of Persistence Diagrams](https://scholars.duke.edu/publication/770168).
