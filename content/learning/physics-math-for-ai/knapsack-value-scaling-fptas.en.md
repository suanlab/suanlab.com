## What does an approximation guarantee promise?

An algorithm can return an excellent answer on one dataset and still have poor worst-case behavior. A guarantee states a property for every instance satisfying explicit assumptions. In this unit a maximization algorithm returns a feasible value A with

$$
A\ge(1-\epsilon)\mathrm{OPT},\qquad 0<\epsilon<1.
$$

The fraction is a quality bound, not a probability of success. For minimization a common convention instead bounds A above by (1+epsilon)OPT. Authors also use approximation factors greater than one; check which inequality defines their convention. Additive error OPT−A and multiplicative ratio A/OPT answer different questions. The ratio is unavailable when OPT=0.

The interactive instance has four binary items, fixed weights (2,3,4,5), nonnegative integer values at most20, and capacity0 through14. Epsilon is selected in tenths from0.1 through0.9. These small limits make the computations inspectable. The general approximation proof below is not inferred from the fact that there are only sixteen subsets.

## Why a heuristic needs a separate theorem

Consider a theoretical family outside the panel's fixed weights: capacity M, one item of weight1 and value2, and another of weight M and value M. For M>2, density greedy first takes the small item and cannot fit the large one. Its value2 divided by optimum M tends to zero. A reasonable local rule does not automatically supply a positive constant approximation ratio.

Branch and bound can certify an exact optimum by keeping sound bounds, but may explore many branches. Here we allow a controlled value loss to reduce a dynamic program's score range. Original weights and capacity stay exact; feasibility is never relaxed by rounding weights.

## Choose a scale only from feasible single items

Exclude items heavier than capacity before computing Vmax. Each remaining single item is feasible, so Vmax≤OPT. With n=4 in the panel define

$$
K=\frac{\epsilon V_{\max}}4,\qquad q_i=\left\lfloor\frac{v_i}{K}\right\rfloor.
$$

For a positive eligible maximum and epsilon=e/10, this becomes K=eVmax/40 and q_i=floor(40v_i/(eVmax)). Decisions use integer arithmetic, not rounded plot labels. Excluded entries use computational zero placeholders in the score vector; the diagram marks their scores unavailable, rather than treating them as rounded eligible values. If there is no positive eligible value, every feasible subset has value zero: return the empty subset and leave K and the observed ratio unavailable.

Each eligible value satisfies 0≤v_i−Kq_i<K. A positive value smaller than K disappears from the scaled score. The orange bars display Kq_i in the same units as the original blue values, rather than plotting incomparable units on one length scale. A star marks an item selected by the approximation.

## A dynamic program indexed by exact score

Let D(i,s) be the minimum original weight attaining exact scaled score s using the first i items. Initialize D(0,0)=0; other scores are unreachable. An excluded item copies the previous row. For an eligible item,

$$
D(i,s)=\min\{D(i-1,s),\ w_i+D(i-1,s-q_i)\},
$$

where the second term is absent for a negative score or an unreachable source. Both choices refer to the previous row. Using newly updated states could reuse an item and silently change a binary problem into an unbounded one.

The implementation stores a subset mask with each finite weight. Equal weights use the smaller mask. It does not maximize original value among equal-score solutions. Choose the largest final score whose minimum weight is at most capacity. Minimum weight can be finite but exceed capacity: that is an attainable score which is infeasible for this capacity, not an unreachable state.

The panel shows five columns around the queried score and all five item-prefix rows. A dash means an unreachable allocated cell; a cross means the score lies outside the allocated table. The readout distinguishes these cases. Mask bits1,2,4,8 select items0,1,2,3.

## Prove the quality bound

Let O be an optimal original subset and S the scaled-DP solution. Since the DP optimizes scaled score, its score is at least the score of O. Rounding loses less than K per selected item, and at most four items occur in O. Consequently

$$
v(S)\ge Kq(S)\ge Kq(O)\ge v(O)-4K
=\mathrm{OPT}-\epsilon V_{\max}
\ge(1-\epsilon)\mathrm{OPT}.
$$

The last step uses the eligible-item assumption. The intermediate additive allowance epsilon Vmax is sometimes tighter than epsilon OPT. This argument is independent of the reference enumeration. Testing small instances can catch an implementation error, but it is not a replacement for the universal inequality.

## Why this is an FPTAS

For general n choose K=epsilon Vmax/n. Every eligible score is at most n/epsilon, so total score Q is at most n squared divided by epsilon. A dense n-row score DP takes O(nQ)=O(n cubed/epsilon) arithmetic operations. The panel retains all rows for inspection and counts their allocated cells; a value-only implementation can retain two rows with O(Q) table storage.

A PTAS is polynomial in input size for each fixed epsilon. An FPTAS is polynomial jointly in input size and1/epsilon. Polynomial dependence on a raw numeric capacity or value is only pseudo-polynomial because writing a large number takes logarithmically many bits. Scaled scores avoid dependence on the original value magnitude in the number of DP cells; exact arithmetic still has an input-bit cost. Dense cell count, finite reachable state count and wall-clock time are different measurements.

## Worked experiments: coarse and fine scales

With values (0,0,3,19), capacity9 and epsilon0.7, Vmax=19 and K=133/40. Scores are (0,0,0,5). The scaled DP chooses item3 alone: original value19, weight5, mask8. The separate exact optimum selects items2 and3, value22, mask12. The observed ratio19/22 is about0.864; the worst-case fraction is0.3. The loss3 is below the additive allowance13.3.

At epsilon0.1, K=19/40 and scores become (0,0,6,40). The DP now returns mask12 with value22 and score46. Dense table cells increase from5×6=30 to5×47=235. This example attains ratio1, but the theorem still promises only0.9 across all allowed inputs. A tighter theorem does not specify the precise output on each individual change of epsilon.

The plot compares FPTAS values with (1−epsilon) times an independently enumerated OPT. Such a numerical theorem line is available here because there are sixteen subsets. The approximation algorithm never consults that reference, and a large real application normally does not know OPT in advance.

## Research extension: a guarantee on learned values is not the true-value guarantee

Suppose nonnegative predicted values $\widehat v_i$ are optimized, while true values satisfy $|v_i-\widehat v_i|\le\delta$ on every eligible item. Assume every feasible set has at most $k$ items and the returned set satisfies $\widehat v(S)\ge(1-\epsilon)\widehat{\mathrm{OPT}}$. Comparing with the true optimum set $O$ gives

$$v(S)\ge(1-\epsilon)v(O)-(2-\epsilon)k\delta.$$

Indeed, subtract at most $k\delta$ from the selected predicted value, compare predicted optima, and subtract another $(1-\epsilon)k\delta$ from the predicted value of $O$. A deterministic bound on every item is essential; average prediction error does not imply it. Weights and feasibility remain exact throughout.

For true values $(6,6,5,5)$, weights $(2,3,4,5)$ and capacity five, $k=2$ and $\mathrm{OPT}=12$. At $\epsilon=1/4$:

| Uniform item error $\delta$ | Conservative true-value bound |
|---|---:|
| 0 | 9 |
| 1/2 | 29/4 |
| 1 | 11/2 |

```python
from fractions import Fraction as F
from itertools import product
true=[6,6,5,5]; weights=[2,3,4,5]; eps=F(1,4)
sets=[x for x in product([0,1],repeat=4) if sum(w*z for w,z in zip(weights,x))<=5]
value=lambda v,x:sum(a*z for a,z in zip(v,x))
assert max(value(true,x) for x in sets)==12
for delta in [F(0),F(1,2),F(1)]:
    lower=9-F(7,2)*delta; count=0
    for signs in product([-1,1],repeat=4):
        predicted=[v+s*delta for v,s in zip(true,signs)]
        best=max(value(predicted,x) for x in sets)
        for x in sets:
            if value(predicted,x)>=(1-eps)*best:
                assert value(true,x)>=lower;count+=1
    print(delta,lower,count)
```

The enumeration checks every feasible subset meeting the predicted approximation inequality for extremal error patterns. This is an auxiliary robustness theorem, not a new score predictor in the native panel. It also separates algorithmic scaling error from statistical score error; decreasing epsilon alone does not remove the latter.

## Exercise and solution: an invalid scale

Exercise: use values (1,0,0,20), capacity2 and epsilon0.9. Why must the value20 item be excluded before scaling?

Solution: it weighs5 and cannot fit. Eligible Vmax is1, K=9/40, and the sole positive feasible item gets score4. The DP returns value1. If the infeasible value20 were used instead, K would be4.5, the feasible value1 would scale to zero, and minimum-weight tie breaking would choose the empty subset. Value0 would violate the promised0.1×OPT=0.1. The failure is in the assumptions used to construct the scale.

Additional check: set all original values to zero. The empty subset is optimal, K is unavailable and A/OPT is0/0, so the display must not invent ratio1. Query row0 at a positive allocated score in a nonzero example to see an unreachable cell, then query score160 in the coarse example to see an outside-table cell.

## Research connections and sources

Learned scores, empirical ratios and guarantees should be reported separately in AI-assisted optimization. Changing the objective, allowing negative values, rounding constraints or changing the recurrence requires a new proof. This deterministic small example makes no claim about approximation schemes for arbitrary nonconvex learning objectives.

For the general definitions see [MIT6.854 approximation notes](https://courses.csail.mit.edu/6.854/21/Notes/n20-approx.html). [Katherine Lai's MIT18.434 knapsack notes](https://math.mit.edu/~goemans/18434S06/knapsack-katherine.pdf) and [MIT6.046J problem9 solutions](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/ff14ef9418b742bf99cfa9f52c6accd7_MIT6_046JS15_pset9sols.pdf) develop value-indexed DP and scaling. The four-item cases, exact parameterization and query diagnostics here are constructed for this textbook.
