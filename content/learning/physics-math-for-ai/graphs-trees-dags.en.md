## A graph represents a relation

A graph G=(V,E) consists of vertices and edges. In a directed graph, (u,v) means u→v and differs from its reverse. The lab uses V={0,1,2}, with no self-loops and at most one edge in each direction.

Bits zero through five encode 0→1,0→2,1→0,1→2,2→0,2→1. Mask 9=1+8 gives the chain 0→1→2. Numerical differences between masks do not define a meaningful graph distance.

## Undirected trees

A finite simple undirected tree is connected and acyclic. With n vertices it has n−1 edges. Any connected graph needs at least n−1 edges; removing an edge on a cycle preserves connectivity. Repeated removal leaves a minimally connected tree.

Consequently, connectedness together with n−1 edges characterizes a tree. The edge count alone is insufficient: a triangle plus an isolated vertex has four vertices and three edges, but is not a tree.

The lab’s tree indicator refers to the underlying simple undirected graph, obtained by removing directions and merging reciprocal edges. It does not call the directed graph itself an undirected tree.

## DAGs and topological order

A DAG has no directed cycles. A topological ordering lists vertices so that every edge points from an earlier vertex to a later one.

Every finite DAG has a vertex of indegree zero. Otherwise, repeatedly following incoming edges eventually repeats a vertex, producing a directed cycle. Remove an indegree-zero vertex, then apply the same argument to the remaining DAG to construct an ordering.

Conversely, a topological ordering rules out a directed cycle: positions would have to increase strictly around the cycle and return to their starting value. Thus

$$
G\text{ is a DAG}\iff G\text{ has a topological ordering}.
$$

## Counterexamples distinguish the concepts

The chain 0→1→2 is both a DAG and an underlying tree. Add 0→2: it remains a DAG, but its underlying graph is a triangle. Instead use 0↔1,1→2: the underlying graph is a tree, but reciprocal edges form a directed two-cycle.

A DAG need not have an underlying tree, and an underlying tree need not imply a DAG when reciprocal edges are permitted. Orienting each undirected tree edge in exactly one direction does yield a DAG; the one-direction assumption matters.

## Follow topological sorting

Count indegrees, initialize a queue with indegree-zero vertices, and repeatedly output a queued vertex while removing its outgoing edges. Newly indegree-zero vertices join the queue. If not all vertices can be output, a directed cycle remains.

Several available vertices may give multiple valid orders. The lab displays one, not a claim of uniqueness. A — means no complete topological order exists; an incomplete output prefix is not displayed as though it ordered the whole graph.

## AI computation graphs

A finite feedforward computation expressed as a DAG can run in topological order. Backpropagation follows dependencies in reverse and accumulates contributions at shared nodes. A recurrent module diagram can contain a cycle while its finite-time unrolling is a DAG. State the level of representation.

Node positions and edge lengths encode neither magnitude nor runtime. Read arrows and the edge table, not geometric distance.

## Cost and independent validation

With adjacency lists, Kahn’s algorithm takes O(|V|+|E|) time: calculate indegrees once, enqueue each vertex once and remove each edge once. Repeatedly scanning an entire adjacency matrix has a different cost. Independently validate a result by checking that it contains every vertex exactly once and that position(u)<position(v) for every edge u→v.

Finding one ordering differs from counting all orderings. With only 0→2 and 1→2, both (0,1,2) and (1,0,2) are valid. A single output is not evidence of uniqueness. Distinguish the algorithm’s selected order from comparisons forced by the graph.

## Exercises and answers

1. Mask 11 is **a DAG with an underlying undirected triangle**.
2. Mask 13 has **an underlying tree and a directed cycle**.
3. If the topological ordering is unique, the indegree-zero choice is **unique at each removal step**.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). Exhausting the lab’s three-vertex cases is separate from proving the general theorem.
