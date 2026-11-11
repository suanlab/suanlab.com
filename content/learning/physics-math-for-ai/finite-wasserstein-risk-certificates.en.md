## Replace a probability divergence with transport geometry

KL reweighting compares probabilities directly. A transport ambiguity set also uses a geometry between outcomes: moving probability to a nearby outcome can cost less than moving it to a distant one. This unit studies a finite example whose primal and dual values can be compared exactly.

Source and allowed destination positions are the three points−1,0 and1. The nominal masses are0.2,0.3 and0.5. Destinations remain restricted to those points. A certificate for this finite support is not automatically a certificate for every real-valued destination or for an unknown future data distribution.

## Define a coupling and preserve the source distribution

Let pi_ij be probability transported from source i to destination j. The source row sums are fixed:

$$
\pi_{ij}\ge0,\qquad\sum_j\pi_{ij}=p_i.
$$

Destination column sums form the alternative distribution q. Source and destination probabilities serve different roles: the first describes the reference population, while the second is chosen adversarially within the cost budget. Every transported unit still has an origin, even though the final distribution alone does not record the whole plan.

Ground cost is absolute distance c_ij=|x_i−x_j|. The total cost is the average mass-times-distance, not the largest distance traveled by an individual unit.

## State the finite worst-risk problem

For fixed destination losses, solve

$$
\max_{\pi\ge0}\sum_{ij}\pi_{ij}l_j
\quad\text{subject to}\quad
\sum_j\pi_{ij}=p_i,\qquad\sum_{ij}\pi_{ij}c_{ij}\le\rho.
$$

The model and losses stay fixed. We are solving the inner distributional risk problem, not training a predictor or asserting that a chosen transport budget covers reality. Identity transport is feasible for every nonnegative radius, so the feasible set is nonempty and worst risk is at least nominal risk.

The finite feasible polytope is bounded because its nonnegative row masses are fixed. An optimum therefore exists. Constant or tied losses can leave multiple optimal plans, so the displayed plan is a selected witness rather than a uniqueness claim.

## Enumerate all necessary extreme-point forms

A linear objective on a compact polytope has an optimal extreme point. With three positive row masses and an inactive cost inequality, an extreme plan can assign each row to one destination. If the cost constraint is active, a basic feasible plan has at most four positive variables: one row may split between two destinations while the other rows remain deterministic.

The enumerator checks all deterministic row assignments and every such single-row split that uses the budget. Equal-cost split directions do not create an additional isolated vertex; their endpoint assignments are already present. This argument explains why the candidate enumeration is complete for this small LP, rather than trusting a dense sampling grid.

A maximizing candidate is not the complete set of optimal couplings. An entire face can be optimal when objective coefficients or marginal budget values tie.

## Derive a dual upper bound from a transport price

Choose a nonnegative price lambda. For every feasible coupling,

$$
\sum_{ij}\pi_{ij}l_j
=\lambda\sum_{ij}\pi_{ij}c_{ij}
+\sum_{ij}\pi_{ij}(l_j-\lambda c_{ij})
\le\lambda\rho+\sum_i p_i\max_j(l_j-\lambda c_{ij}).
$$

The last expression is a valid upper bound at every nonnegative price. It first prices the total transport cost and then bounds each row's remaining loss by its best destination. Minimizing it gives the finite transport dual.

For each source, the maximum is an envelope of affine functions of lambda. Its slope can change only at their pairwise intersections. Check those nonnegative intersections and zero to find a dual optimizer. If a feasible primal plan and a valid dual price have the same value, weak duality certifies the global optimum immediately.

## Preserve exact arithmetic on the supported grid

The probability masses are represented by integer units(8,12,20) over40. The radius is an integer control divided by20. With costs0,1 and2, the candidate split and dual-price denominators are controlled: plan masses stay integer in these units and prices are half-integers.

Both primal and dual risks consequently have integer numerators over40. The code checks source sums, nonnegativity, cost feasibility and numerator equality. This is an exact certificate for the stated finite model, distinct from the numerical logarithmic solve in the KL unit.

The property depends on the fixed support, masses and control grid. Arbitrary real probabilities or ground costs would require a different arithmetic analysis; displaying many decimals would not restore an exact proof.

## Work the zero-budget and small-budget guides

For losses(0,1,3), identity transport has nominal risk

$$
0.2\cdot0+0.3\cdot1+0.5\cdot3=9/5.
$$

At radius zero, only zero-cost diagonal moves are possible because distinct support points have positive distance. The destination distribution therefore remains the baseline.

At radius0.1, move mass0.1 from the middle point to the right point, a distance1. The average cost is0.1 and q=(0.2,0.2,0.6). Risk is2. A dual price2 gives the same upper value: the rowwise maxima of loss minus twice cost are0,1 and3, and the priced budget contributes0.2.

This example directly separates average transport cost from a per-record radius. A pointwise radius0.1 on this discrete support would allow no record to move to a different destination, whereas the average-cost constraint allows a small mass to move distance1.

## Follow the budget's marginal value

Initially, moving middle mass right earns2 units of loss per unit distance. After the entire middle mass has moved, moving left mass right earns3/2 per distance. At radius0.5 the selected destination distribution is(0.1,0,0.9), risk27/10, and a dual price3/2 matches it.

At radius0.7 all mass can reach the maximal-loss destination and risk saturates at3. A selected optimal dual price is then zero. The cost constraint can still be binding at this threshold, so zero price does not imply slackness in the reverse direction of complementary slackness.

The piecewise-linear risk envelope reflects changing optimal transport patterns. Its slopes and dual prices have a precise finite optimization interpretation, not a claim that an equally numbered KL ball or pointwise adversarial ball describes the same uncertainty.

## Read the flow and dual diagrams

The first diagram draws source rows, destination masses and positive transport flows. Line widths show transported probability; the labels retain integer mass units. The displayed cost is compared with the budget before the primal/dual equality is used as a certificate.

The second diagram plots the dual upper objective against price and marks one selected optimal price. Several prices may minimize it, especially at a kink or plateau. The last graph varies the average-cost budget with losses fixed and compares worst risk with nominal risk. It is not a physical transport-time simulation or a learning trajectory.

## Research extension: Account for a finite destination grid

A finite-support optimum excludes unseen destinations by definition. To relate it to a continuous problem, add a regularity assumption rather than silently extending the certificate. Suppose continuous destinations lie in a compact interval, loss $l$ is $L$-Lipschitz, all source points belong to a finite grid, and a nearest-grid map moves every destination by at most $e$. Write $R_C$ and $R_G$ for continuous and grid worst risks with absolute-distance transport cost.

Every grid plan is a continuous plan. Conversely, round the destinations of a continuous feasible plan. The triangle inequality increases its average transport cost by at most $e$, while its loss decreases by at most $Le$. Thus

$$
R_G(\rho)\le R_C(\rho)\le R_G(\rho+e)+Le.
$$

The enlarged budget is essential; rounding need not preserve the original cost constraint. This is a deterministic approximation bound for a specified loss and domain, not evidence that a future distribution belongs to either ambiguity set. Without a known modulus of continuity, small geometric spacing need not control loss between grid points.

An auxiliary example exposes the issue. Put all source mass at zero on $[-1,1]$ and let $l(y)=\max(0,1-4|y-1/4|)$, so $L=4$. Since $l(y)\le4|y|$ and $l(y)\le1$, continuous risk is exactly $\min(4\rho,1)$: mix zero and the peak $1/4$ to attain it. At $\rho=1/8$, continuous risk is $1/2$:

| Uniform grid spacing $h$ | Grid risk | Continuous risk | Rounding distance $e=h/2$ |
|---|---:|---:|---:|
| 1 | 0 | 1/2 | 1/2 |
| 1/2 | 0 | 1/2 | 1/4 |
| 1/4 | 1/2 | 1/2 | 1/8 |
| 1/8 | 1/2 | 1/2 | 1/16 |

The two coarse grids entirely miss the narrow peak. The bound remains valid, though it is deliberately conservative there. The code solves each auxiliary single-source LP by enumerating feasible one- and two-destination extreme distributions and checks the bound at three budgets. The native three-source exact certificate remains a finite-support result.

```python
from fractions import Fraction as F
from itertools import combinations
def loss(y):return max(F(0),1-4*abs(y-F(1,4)))
def grid_risk(h,rho):
    points=[F(-1)+j*h for j in range(int(2/h)+1)]
    candidates=[loss(y) for y in points if abs(y)<=rho]
    for a,b in combinations(points,2):
        ca,cb=abs(a),abs(b)
        if ca==cb:continue
        weight=(rho-cb)/(ca-cb)
        if 0<=weight<=1:candidates.append(weight*loss(a)+(1-weight)*loss(b))
    return max(candidates)
for h in [F(1),F(1,2),F(1,4),F(1,8)]:
    e=h/2
    for rho in [F(1,8),F(1,4),F(3,8)]:
        continuous=min(4*rho,F(1)); finite=grid_risk(h,rho)
        assert finite<=continuous<=grid_risk(h,rho+e)+4*e
    print(h,grid_risk(h,F(1,8)),F(1,2),e)
```

## Exercises and explained solutions

1. Why can the radius-zero plan not move mass between distinct points? Solution: each such move has positive ground cost, and all masses are nonnegative.
2. Can mass0.1 move distance1 under budget0.1? Solution: yes; the product is0.1. The budget is an average-cost bound.
3. Why does a matching primal and dual value prove optimality? Solution: every feasible risk lies below the dual bound, while the selected feasible plan already attains it.
4. Does matching risk prove a unique plan? Solution: no. A whole face of feasible plans can share the optimal objective.
5. If a dual price is zero, must the cost constraint be slack? Solution: no. Saturation at radius0.7 gives a binding budget with a zero optimal price.
6. If every destination loss increases by2, how does worst risk change? Solution: it increases by2 because every coupling transports total mass1.
7. Can this finite certificate cover an unseen destination with unknown loss? Solution: no. The support and losses were explicit constraints of the problem.

## Connect to the statistical question

[Mohajerin Esfahani and Kuhn](https://arxiv.org/abs/1505.05116) study Wasserstein distributional robustness, reformulations and performance guarantees under stated assumptions. Our direct finite LP demonstrates transport ambiguity and duality; it does not reproduce their whole theory. The next original topic asks how mathematical guarantees connect to practical evaluation, including the separate question of whether the real distribution lies in the modeled uncertainty set.
