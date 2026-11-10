## From graph search to an optimality certificate

Find the largest feasible flow from s to t. Arcs have capacities, and inflow equals outflow at internal nodes. This chapter uses four nodes and five directed arcs to execute BFS, bottleneck updates, residual reachability and cut calculations. The small graph also permits an independent exhaustive check of its four possible s-t cuts.

Upper edge labels are current flow/capacity. Orange nodes are reachable from s in the residual graph. If t belongs to that set, it is not yet an s-t cut certificate. Blue shows actual flow values after augmentations; the orange reference line is the exhaustively computed minimum cut capacity.

## Feasibility and residual capacity

$$
0\le f_{ij}\le c_{ij},\qquad
\sum_j f_{ij}-\sum_j f_{ji}=0\quad(i\ne s,t).
$$

The implementation stores skew-symmetric net flow F, with Fji=-Fij. Forward residual capacity is Cij-Fij. A reverse residual arc represents the ability to cancel previously sent flow. Removing reverse arcs would prevent correction of earlier choices and define a different algorithm. The drawing displays original arcs, while residual search also includes reverse directions.

BFS traverses only positive residual arcs. A found path is augmented by its minimum residual capacity, adding delta forward and subtracting it backward. Integral capacities produce integral flow in this example. General termination and Edmonds–Karp complexity require a separate argument based on shortest residual paths.

## Why a cut is an upper bound

Let S contain s but exclude t. Sum conservation equations over internal nodes of S. Internal arcs cancel, leaving net outward flow:

$$
|f|=\sum_{i\in S,j\notin S}f_{ij}-\sum_{i\notin S,j\in S}f_{ij}
\le\sum_{i\in S,j\notin S}c_{ij}=c(S,\bar S).
$$

If residual search cannot reach t, every outward arc from reachable S is saturated and no positive flow crosses back into S. The inequality becomes equality, certifying both maximum flow and minimum cut. Reaching an execution cap does not establish that condition.

## Worked example and exercise

Set capacities s→a=3, s→b=2, a→b=1, a→t=2 and b→t=3. BFS sends two along s-a-t, two along s-b-t, and finally one along s-a-b-t. Total flow is five. The source-outgoing cut also has capacity 3+2=5, so a larger feasible value is impossible.

Exercise: allow only one augmentation. Is the result maximum? Solution: current value is two and a residual route such as s-b-t remains. The optimality flag must be zero and the reachable-cut certificate unavailable. Distinguish a feasible intermediate flow from an optimum. With all capacities zero, flow zero and cut zero already match.

## Certificates for AI optimization

For matching, assignment and routing, objective values should be accompanied by feasibility checks and matching bounds when available. A learned optimizer's candidate can still be checked against original capacity and conservation constraints. Exhaustive cuts serve as an independent oracle here, not an efficient algorithm for arbitrary large graphs. Separate verifiable certificates from merely attractive scores.

Reference: [MIT graph algorithm materials](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
