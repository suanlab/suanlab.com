## Pairings and certificates answer different questions

Suppose three tasks need different machines, but each task can run only on some machines. A matching chooses permitted pairs without reusing a task or machine. Finding two pairs demonstrates that at least two assignments are possible. It does not prove that three are impossible. A cut or a vertex cover supplies an upper bound, and an equal lower and upper bound certifies optimality.

This distinction is useful in assignment, tracking and resource allocation. Here we maximize the number of pairs, not a weighted quality score. A maximum-cardinality solution need not maximize a separate sum of compatibility scores. The small model makes every residual step and certificate inspectable.

## Encode the bipartite graph

Left vertices are L0,L1,L2 and right vertices are R0,R1,R2. A neighbor mask is a sum of bits: 1 enables R0, 2 enables R1, and 4 enables R2. Masks (3,1,6) mean L0 connects to R0,R1; L1 only to R0; L2 to R1,R2. All edges cross the partition.

Binary variables obey

$$
x_{ij}\in\{0,1\},\qquad \sum_jx_{ij}\le1,\quad\sum_ix_{ij}\le1,
\qquad \max\sum_{(i,j)\in E}x_{ij}.
$$

Missing edges cannot be selected. The interactive graph has nine possible original edges, giving 512 possible input graphs. This finite size supports independent exhaustive verification; it is not the algorithm used to compute the displayed flow.

## Build the flow reduction

Add source s and sink t. Arcs s→Li and Rj→t have capacity one. Every original Li→Rj edge has capacity four. Flow conservation says that internal incoming and outgoing flow agree. Integer outer flows restrict each left and right vertex to at most one selected internal edge, so an integral feasible flow corresponds to a matching with the same value. Conversely, every matching determines such a flow.

Why four? The cut containing only s has capacity three. A minimum cut therefore costs at most three and cannot cross an internal capacity-four arc. This finite bound plays the role sometimes assigned to infinite internal capacities. It is chosen from a known bound, rather than being an arbitrary enormous numerical constant.

## A residual reverse arc repairs an earlier choice

The algorithm starts with zero flow and runs deterministic breadth-first searches. A forward residual arc has unused capacity; a reverse residual arc allows canceling some previously sent flow. The bottleneck of each augmenting path is one because its outer arcs have unit capacity. Send one unit along the entire path: increase forward flow and decrease the corresponding original flow on reverse arcs.

For masks (3,1,6), the first augmentation selects L0-R0. The next shortest path is s-L2-R1-t, yielding two pairs. Now the next path is

$$
s\to L1\to R0\to L0\to R1\to L2\to R2\to t.
$$

It adds L1-R0, cancels L0-R0, adds L0-R1, cancels L2-R1 and adds L2-R2. Three additions minus two cancellations increase matching size by one. The final matching is L0-R1, L1-R0, L2-R2. Reverse residual arcs do not mean negative actual assignments or negative original capacities. The displayed table keeps original-arc flow nonnegative and shows reverse residual capacity separately.

The implementation computes the complete history, then displays the requested prefix. The separately labeled terminal optimum is available for comparison, but it does not certify an earlier displayed prefix. If fewer than three augmentations exist, larger requests show the terminal state and the actual completed count.

## Cuts bound every feasible matching

An s-t cut is a source-side set S containing s and excluding t. Summing flow conservation over S expresses net flow as outgoing minus incoming flow. Outgoing flow cannot exceed outgoing capacity, so any feasible flow value is at most any valid cut capacity.

Let S be all residual vertices reachable from s. Before termination, t is reachable and S is not an s-t cut at all. The readout therefore withholds its cut and cover certificates. Once no augmenting path remains, t is outside S. Every outgoing original cut arc is saturated and every incoming original arc carries zero flow. Hence flow value equals cut capacity, proving both are optimal.

## Derive a minimum vertex cover

A vertex cover K touches every original edge. In any graph, each edge of a matching requires a different cover vertex, giving |matching|<=|K|. At a terminal prefix in this bipartite reduction, define

$$
K=(L\setminus S)\cup(R\cap S).
$$

The minimum cut cannot cross an internal arc because that alone would cost four. Thus no edge joins L inside S to R outside S, meaning every edge touches K. Its size equals the cut capacity: one for each excluded left vertex and included right vertex. We have a matching and a cover of equal size, certifying minimum cover and maximum matching. This is the bipartite matching-cover equality. A triangle outside the model has maximum matching one and minimum cover two, so bipartiteness matters.

## Query a vertex set without assuming it covers

Query bits 0..2 select left vertices; bits 3..5 select right vertices. For any queried set K, even an invalid cover, construct the valid cut S_K={s}∪(L outside K)∪(R inside K). If u(K) counts uncovered edges,

$$
\operatorname{cap}(S_K)=|K|+4u(K).
$$

This remains an upper bound, but |K| alone is a cover bound only if u(K)=0. Selecting all six vertices is a valid cover yet can be far from minimum. Query status, valid cut capacity and its gap above the current flow are reported separately.

For masks (1,1,2), only two pairs can be assigned. At termination reachable vertices are s,L0,L1,R0, giving K={L2,R0}, query mask 4+8=12. Matching, minimum cover and minimum cut all equal two. The left subset {L0,L1} has only neighbor R0, directly explaining why a perfect matching is impossible.

## Read the graphics

The first diagram shows the original flow reduction, with blue current flow and matching, orange-filled certified cover vertices at terminal prefixes, and dashed orange query rings. A queried ring is not a certification. The second diagram lists the next residual path, residual capacities and additions/cancellations. It lists no further path at termination.

The lower plot uses actual completed augmentation counts. Blue follows the feasible matching history and ends when the algorithm terminates. Orange is the queried valid cut bound. The selected blue dot need not meet that bound, especially for a loose queried set. Neither curve is a training-loss trajectory.

## Research extension: a Hall deficit explains every unmatched task

For a left subset $S$, at most $|N(S)|$ of its tasks can be matched. Thus every matching leaves at least $|S|-|N(S)|$ left vertices unmatched. Define $d=\max_{S\subseteq L}(|S|-|N(S)|)$, including the empty set so $d\ge0$. The exact identity is

$$\nu(G)=|L|-d.$$

To prove attainability, use a terminal residual search. Its reachable left and right sets satisfy $N(S_L)=S_R$; every reachable right vertex is matched, and its partner is reachable too. Counting reachable matched pairs leaves exactly the globally unmatched left vertices in $|S_L|-|S_R|$. This realizes the deficit lower bound. Hall's perfect-matching criterion is precisely $d=0$.

| Neighbor masks | Maximum deficit | Maximum matching |
|---|---:|---:|
| (7,7,7) | 0 | 3 |
| (1,1,2) | 1 | 2 |
| (0,0,0) | 3 | 0 |

![Counts of all 512 labeled three-by-three bipartite graphs grouped by maximum Hall deficit; deficit zero means a perfect matching exists.](/assets/learning/combinatorial-hall-deficit-en.svg)

```python
from itertools import product
from collections import Counter
def matching(masks,i=0,used=0):
    if i==3:return 0
    return max([matching(masks,i+1,used)]+[1+matching(masks,i+1,used|(1<<j)) for j in range(3) if masks[i]&(1<<j) and not used&(1<<j)])
histogram=Counter()
for masks in product(range(8),repeat=3):
    deficit=0
    for subset in range(8):
        neighbors=0
        for i in range(3):
            if subset&(1<<i):neighbors|=masks[i]
        deficit=max(deficit,subset.bit_count()-neighbors.bit_count())
    assert matching(masks)==3-deficit
    histogram[deficit]+=1
assert dict(histogram)=={0:247,1:231,2:33,3:1}
print(sorted(histogram.items()))
```

The code compares two independent finite calculations on all 512 graphs: recursive matching and exhaustive subset deficits. The histogram counts equally listed labeled graphs; it is not an empirical distribution of real assignment systems. **Exercise:** find the deficient set for (1,1,2), then explain why simply adding an edge outside that set need not repair its bottleneck.

## Exercises and solutions

Exercise: on the complete three-by-three graph at prefix zero with query mask zero, what are the queried cut capacity and the reachable-cut certificate?

Solution: nine edges are uncovered, so the query-derived valid cut has capacity 4*9=36. The sink is residual-reachable, so the residual reachable set is not a valid cut and its certificate is unavailable. An upper bound can be valid and very loose.

Exercise: use masks (3,1,0) at prefix one. What happens next? Solution: the path s-L1-R0-L0-R1-t cancels L0-R0 and produces two pairs. The isolated L2 cannot be matched. This smaller example needs one cancellation, whereas the default three-pair example needs two in its final augmentation.

Exercise: does a deterministic returned matching prove uniqueness? Solution: no. The complete graph has several perfect matchings and several minimum covers. The certificate proves the optimal value; the algorithm's neighbor order selects a representative.

## Further study

[MIT's Network Flows and Matching notes](https://ocw.mit.edu/courses/18-200-principles-of-discrete-applied-mathematics-spring-2024/mit18_200_s24_lec11-12.pdf) develop the flow/matching and cover connection. [Karger's flow-computation notes](https://courses.csail.mit.edu/6.854/21/Notes/n07-flow_computation.html) discuss residual optimality. The capacity-four model, prefix diagnostics and worked traces here are explicit textbook constructions checked against independent finite enumeration. General weighted matching and nonbipartite algorithms require additional machinery.
