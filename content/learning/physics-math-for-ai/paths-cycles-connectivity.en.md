## Separate three questions

Can vertex zero reach vertex two? What is the minimum number of edges? How many walks use exactly two edges? These are distinct questions. A walk can repeat vertices or edges. Here a simple path has no repeated vertices, and a length-zero path from a vertex to itself is allowed.

The lab uses three vertices and unweighted directed edges. Orange highlights one selected shortest path from source s to target t, not every possible path.

## Breadth-first search

BFS visits the source at depth zero, then its neighbors at depth one, then depth two, and so on. Each vertex enters the queue only when first discovered. Its first discovery depth is a shortest distance: a shorter path would have discovered it in an earlier layer.

For 0→1→2, the path from zero to two is (0,1,2), distance two. The reverse is unreachable. The lab displays — for unreachable distance, distinct from zero. When s=t, the empty path (s) always has distance zero.

Weighted costs require a different argument and possibly a different algorithm. Minimizing edge count does not generally minimize total weight.

## Matrix multiplication counts walks

Use Aᵤᵥ=1 exactly when u→v. Then

$$
(A^2)_{st}=\sum_v A_{sv}A_{vt}
$$

counts two-edge walks by their intermediate vertex. Induction gives that (Aᵏ)ₛₜ counts length-k walks, while A⁰=I represents empty walks.

These need not be simple paths. In 0↔1, (A²)₀₀=1 counts the walk 0→1→0, which repeats zero. Do not interpret adjacency powers as generally counting simple paths or shortest paths.

## Reachability uses Boolean operations

Replace addition and multiplication with OR and AND to ask existence rather than count. The table includes reflexive reachability. Allowing vertex k as an intermediate updates

$$
R_{uv}\leftarrow R_{uv}\lor(R_{uk}\land R_{kv}).
$$

Processing all vertices gives transitive closure. Numeric counting and Boolean existence can use similar-looking matrix notation while belonging to different algebras.

## Weak and strong connectivity

A directed graph is weakly connected when forgetting directions makes it connected. Strong connectivity requires every pair to reach one another. Mutual reachability is reflexive, symmetric and transitive, so it partitions vertices into strongly connected components.

The chain 0→1→2 is weakly connected but has three strong components. Adding 2→0 merges them into one. Contracting each strong component gives a DAG: a cycle between contracted components would mean they were mutually reachable and should already have been one component.

## AI interpretation

Dependency checks, knowledge-graph queries and k-hop message-passing neighborhoods use these ideas. A k-layer network does not necessarily process every length-k simple path independently. Specify repeated visits, normalization, direction and self-loops.

Node positions are illustrative; drawn edge length is not a weight. The displayed shortest path minimizes the number of directed edges.

## Exercises and answers

1. For mask five, **(A²)₀₀=1**, through 0→1→0.
2. In the empty graph, **d(0,0)=0**, while **two is unreachable from zero**.
3. Contracting strong components yields **a DAG**, by the mutual-reachability argument.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). Matrix formulas use the stated row-source orientation.
