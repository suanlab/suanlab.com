## Search with a reason to stop

Finding a good allocation and proving that nothing better exists are different tasks. A neural model might suggest a useful candidate, but its score alone is not an upper bound on every alternative. Branch and bound keeps a feasible solution and bounds on the parts of the search that remain. A search tree then becomes an explanation of what was explored, what was excluded and what is still uncertain.

This lab maximizes the value of three binary items with fixed weights (2,3,4):

$$
\max_{x\in\{0,1\}^3}v_0x_0+v_1x_1+v_2x_2,
\qquad 2x_0+3x_1+4x_2\le C.
$$

Values are integers from zero to nine; capacity is an integer from zero to nine. The empty allocation is always feasible. Infeasibility of a particular child therefore cannot mean that this entire example is infeasible. The goal is the optimal value and one optimizer, not enumeration of all tied optimizers.

## A witnessed lower bound and a relaxed upper bound

The incumbent is an actual binary allocation satisfying capacity. If its value is L and the optimal value is OPT, then L≤OPT. Initially the lab floors the root LP coordinates: keep coordinates equal to one and discard the fractional item. Positive weights and nonnegative values make this a feasible candidate, although it can be poor.

At a node, some coordinates are fixed to zero or one. Preserve these fixings and relax only the remaining coordinates to the interval [0,1]. Every binary completion remains feasible in the relaxation, so maximizing the relaxation supplies a node upper bound:

$$
\max_{x\in F_N}v^Tx\le\max_{x\in R_N}v^Tx=U_N,
\qquad F_N\subseteq R_N.
$$

The inequality direction depends on maximization. For minimization, feasible solutions supply upper bounds and relaxations supply lower bounds instead. A high relaxed score is not itself an achievable binary solution.

## Solve this LP without hiding a second integer solver

Subtract the load of the fixed-one items. If it already exceeds capacity, the node is infeasible. Otherwise fill free positive-value items in decreasing value-per-weight order. If a lower-density item occupies capacity while a higher-density free item is not full, moving a small amount of load toward the higher-density item cannot decrease value. Repeating this exchange gives the fractional-knapsack optimum, with at most one fractional item. Zero-value free items can remain at zero; this chooses one LP optimum without asserting uniqueness.

Density comparisons use cross products, with item-index tie breaking. Since weights are 2,3,4 and capacity is integral, all LP coordinates and objectives are multiples of 1/12. The implementation stores integer twelfths and compares them exactly. Rounded display decimals do not determine pruning.

## Branch, close or prune

When the chosen LP optimum is integral, it solves that node's integer problem. Otherwise split its fractional variable into zero and one children. Each integer completion belongs to exactly one child. The lab visits the zero child first in deterministic depth-first order.

Before deciding a processed node's status, floor its LP point and update the incumbent if that candidate improves it. A fractional node is bound-pruned when

$$
U_N\le L.
$$

No completion of that node can improve the incumbent. Equality pruning may hide a different allocation with the same best value. An infeasible node is a different kind of closure: it has no relaxation point at all. The tree uses `open`, `split`, `int`, `bound`, and `none` to distinguish these cases. The selected-node table records the incumbent available at closure, not a later, better value.

## The frontier gives a global certificate

Open frontier nodes cover every remaining integer candidate not already closed or safely excluded. Combining their bounds with the incumbent gives

$$
L\le\mathrm{OPT}\le U=\max\bigl(\{L\}\cup\{U_N:N\text{ is open and feasible}\}\bigr).
$$

An infeasible frontier node contributes nothing. With an empty frontier set U=L. The lab's displayed zero-gap certificate uses U=L; it does not implement every possible strengthening, such as rounding an upper bound down for an integer-valued objective. Lower bounds cannot decrease, and global upper bounds cannot increase as this exact search progresses.

LP bounds are computed when nodes are created. Creation count is therefore the count of bound evaluations; processing count is a separate measure, and neither is elapsed time. Root flooring initializes the incumbent before processing. Later node candidates enter only when that node is processed. Earlier prefixes never borrow a future incumbent. The illustrative trace continues draining the frontier after a zero gap, so certification and completed processing are deliberately separate indicators.

## Worked example: a gap that closes

Use values (3,4,5) and capacity4. The root LP allocation is (1,2/3,0), with bound17/3. Flooring gives incumbent (1,0,0), value3. Processing the root branches on x1; the two child bounds are both11/2. At prefix1 the displayed global gap is therefore 11/2−3=5/2, not the root gap8/3.

At prefix4, the search has found (0,0,1), value5. It still has an upper bound11/2 from a pending node. The complete nine-node processing trace ends at L=U=5. Node N6 fixes x0=x1=1, whose weight5 exceeds capacity4; it is infeasible even though the global incumbent is valid. Querying N6 shows no orange LP allocation, while the global blue integer witness remains visible.

## Worked example: pruning a tied branch

Use values (1,1,2), capacity7. The initial witness (1,0,1) has value3; root bound is10/3. After branching on x1, both child bounds are3. N2 fixes x1=1 and has fractional LP allocation (1,1,1/2), value3. Its floored candidate is worse, but its upper bound cannot improve L=3, so the node is bound-pruned. The different integer allocation (0,1,1) also has value3 and lies in that excluded branch. The optimal value and a witness are certified; uniqueness and complete enumeration are not.

## Research extension: exploit the objective lattice before the gap is zero

The native objective uses integer values and binary decisions, so every feasible objective is an integer. If $L$ is a witnessed incumbent and $U$ a sound real upper bound, then

$$L\le\mathrm{OPT}\le\lfloor U\rfloor.$$

Thus $U<L+1$ already certifies $\mathrm{OPT}=L$. An additive gap below one can prove exact optimality even when the real bounds do not meet. For twelfths this is the exact comparison `upper12 < 12*(L+1)`, with no epsilon. An approximate bound rounded in the wrong direction would invalidate it.

For the existing $(3,4,5)$, capacity-four example, the prefix-four witness has value five and global upper bound $11/2$. The strengthened certificate is already conclusive. The native indicator deliberately still uses equality of real bounds; this auxiliary certificate does not rewrite its trace or borrow a future incumbent.

| Processed prefix | $L$ | $U$ | Integer-lattice certificate |
|---|---:|---:|---|
| 0 | 3 | 17/3 | Not yet |
| 1 | 3 | 11/2 | Not yet |
| 4 | 5 | 11/2 | Exact optimum 5 |

```python
from fractions import Fraction as F
from itertools import product
values=[3,4,5]; weights=[2,3,4]
optimum=max(sum(v*z for v,z in zip(values,x)) for x in product([0,1],repeat=3) if sum(w*z for w,z in zip(weights,x))<=4)
assert optimum==5
for prefix,L,U in [(0,3,F(17,3)),(1,3,F(11,2)),(4,5,F(11,2))]:
    assert L<=optimum<=U
    certificate=U<L+1
    if certificate:assert optimum==L
    print(prefix,L,U,certificate)
```

If objective coefficients share a known positive lattice spacing, the next feasible objective level can give a stronger stopping rule. Learned real scores generally do not lie on that lattice. Exact-value certification still does not establish uniqueness or enumerate all optimal decisions.

## Exercise and solution

Exercise: use values (3,4,5), capacity1 and prefix3. Can a zero gap coexist with an unfinished frontier?

Solution: every nonempty binary selection exceeds capacity, so the empty allocation has optimal value zero. By prefix3, the successive zero branches have isolated the all-zero allocation, while the one branches are already known infeasible from their LP bounds. There are four pending nodes, but L=U=0. Processing all seven created nodes drains the frontier and does not improve the value. This is a proof from bounds, not evidence that many nodes must always be explored.

As a second check, use capacity5 with the original values. The root LP is already (1,1,0), value7. Its integral point certifies the optimal value before any split. A request beyond the available processing history clamps to its last state; a requested future node clamps to the last created node rather than inventing a result.

## Connection to AI research

Learned branching or candidate heuristics can change search order and speed while retaining a valid certificate only if bounding and exclusion remain sound. A learned estimate is not automatically a certified bound. This exact tiny example isolates the logic; it makes no performance claim about industrial mixed-integer solvers or arbitrary nonlinear models. Larger instances can require exponentially many nodes and substantially harder relaxations.

[Boyd and Mattingley's Stanford branch-and-bound notes](https://web.stanford.edu/class/ee364b/lectures/bb_notes.pdf) develop the general bound-and-partition viewpoint and Boolean-convex problems. The maximizing three-item instances, exact twelfths and prefix diagnostics here are constructed for this textbook; the notes mainly present minimization conventions.
