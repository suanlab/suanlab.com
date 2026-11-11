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

## A primal–dual gap remains useful before the algorithm finishes

Any feasible flow gives a lower bound L on maximum flow; any valid s–t cut gives an upper bound U. Therefore the remaining optimum improvement is at most U−L. Equality certifies optimality without trusting a solver's success flag. A residual-reachable set containing t is not such a cut. A separately chosen source-only cut is always valid in this network.

For all five capacities equal to one, force the first non-shortest path s–a–b–t in an auxiliary Ford–Fulkerson run. The next path s–b–a–t uses reverse residual arc b–a and cancels the old a–b flow. The live BFS would initially choose shorter paths; this constructed ordering isolates why reverse arcs are necessary.

| Stage | Feasible flow L | Source cut U | Gap |
|---|---:|---:|---:|
| Initial | 0 | 2 | 2 |
| First forced path | 1 | 2 | 1 |
| Reverse correction | 2 | 2 | 0 |

```python
C=[[0]*4 for _ in range(4)];F=[[0]*4 for _ in range(4)]
arcs=[(0,1),(0,2),(1,2),(1,3),(2,3)]
for i,j in arcs:C[i][j]=1
def augment(path):
    assert all(C[i][j]-F[i][j]>=1 for i,j in zip(path,path[1:]))
    for i,j in zip(path,path[1:]):F[i][j]+=1;F[j][i]-=1
    assert all(0<=F[i][j]<=C[i][j] for i,j in arcs)
    assert sum(F[1])==sum(F[2])==0
print(0,2,2)
for path in ((0,1,2,3),(0,2,1,3)):
    augment(path);value=sum(F[0]);print(value,2,2-value)
assert F[1][2]==0 and sum(F[0])==2
```

For general Edmonds–Karp, BFS source distances never decrease: a newly created reverse arc cannot create a shorter route under the old distance labels. If an arc becomes saturated again after its reverse was used, its tail distance has increased by at least two. Each directed arc is thus critical O(V) times. Every augmentation saturates an arc, giving O(VE) augmentations and O(VE²) time with O(E) BFS work. This assumes adjacency-based traversal and unit-cost capacity arithmetic; encoded-number bit costs require additional accounting. The four-node exhaustive cut oracle is separate from that general complexity proof.

## Worked example and exercise

Set capacities s→a=3, s→b=2, a→b=1, a→t=2 and b→t=3. BFS sends two along s-a-t, two along s-b-t, and finally one along s-a-b-t. Total flow is five. The source-outgoing cut also has capacity 3+2=5, so a larger feasible value is impossible.

Exercise: allow only one augmentation. Is the result maximum? Solution: current value is two and a residual route such as s-b-t remains. The optimality flag must be zero and the reachable-cut certificate unavailable. Distinguish a feasible intermediate flow from an optimum. With all capacities zero, flow zero and cut zero already match.

## Certificates for AI optimization

For matching, assignment and routing, objective values should be accompanied by feasibility checks and matching bounds when available. A learned optimizer's candidate can still be checked against original capacity and conservation constraints. Exhaustive cuts serve as an independent oracle here, not an efficient algorithm for arbitrary large graphs. Separate verifiable certificates from merely attractive scores.

Reference: [MIT graph algorithm materials](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
