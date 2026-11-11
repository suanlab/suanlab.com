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

## What a bottleneck certificate actually preserves

![H1 intervals for three square perturbations remain close while their alive indicators disagree at threshold epsilon 1.05.](/assets/learning/geometry-persistence-en.svg)

For every simplex, the difference of its maximum edge lengths is at most the maximum pairwise distortion D. The two inclusion maps shift filtration by D and compose to the usual structure maps shifted by 2D. Applying homology gives a D-interleaving. The algebraic stability theorem for these finite persistence modules then provides a diagram matching of cost at most D. The final step is a theorem about modules, not a consequence of merely seeing nearby plotted points. See [Chazal, de Silva, Glisse and Oudot](https://arxiv.org/abs/1207.3674).

A bar with lifetime greater than 2D cannot match the diagonal at that cost, because diagonal distance is half its lifetime. It must match an actual bar. For a matched pair with endpoint shifts at most D, its alive/dead status at ε agrees if ε lies more than D from both original endpoints. This statement is per matched feature; short unmatched features can still change the total Betti number.

For one finite H₁ bar per diagram, the exact bottleneck distance is the smaller of direct endpoint matching and deleting both bars to the diagonal. The code independently evaluates this formula for perturbed rectangles, without using a persistence library or an assignment solver.

| Perturbation δ | Reference H₁ | Perturbed H₁ |
|---:|---|---|
| 0 | [1,√2) | [1,√2) |
| 0.1 | [1,√2) | [√1.22,√2) |
| 0.25 | [1,√2) | [√1.625,√2) |

## Reproduce the calculation in Python

```python
from math import hypot, sqrt
for delta in [0, .1, .25]:
    P = [(0,0),(1,0),(1,1),(0,1)]
    Q = [(0,0),(1,0),(1+delta,1-delta),(0,1)]
    birth = max(hypot(Q[(i+1)%4][0]-Q[i][0],Q[(i+1)%4][1]-Q[i][1]) for i in range(4))
    death = min(hypot(Q[2][0]-Q[0][0],Q[2][1]-Q[0][1]), sqrt(2))
    distortion = max(abs(hypot(P[i][0]-P[j][0],P[i][1]-P[j][1])-hypot(Q[i][0]-Q[j][0],Q[i][1]-Q[j][1])) for i in range(4) for j in range(4))
    direct = max(abs(birth-1), abs(death-sqrt(2)))
    diagonal = max((sqrt(2)-1)/2, max(0,death-birth)/2)
    bottleneck = min(direct,diagonal)
    assert bottleneck <= distortion+1e-12
    print(delta, round(birth,9), round(death,9), round(bottleneck,9))
```

## Exercise and limits

Set δ=0: distortion, both diagram distances and labeled displacement vanish. Then double scale while fixing δ and doubling ε. Every distance bound and finite persistence coordinate doubles, while the rescaled Betti pattern is unchanged. Explain why a long feature, with lifetime greater than twice a proposed bottleneck error, cannot disappear into the diagonal within that error.

For real data, document the metric, filtration convention, coefficients, noise model and any correspondence or sampling theorem. Diagram stability controls perturbation of a chosen construction. It does not automatically prove the topology or causal meaning of the unknown system that generated the observations.

Further reading: [Cohen-Steiner, Edelsbrunner and Harer, Stability of Persistence Diagrams](https://scholars.duke.edu/publication/770168).
