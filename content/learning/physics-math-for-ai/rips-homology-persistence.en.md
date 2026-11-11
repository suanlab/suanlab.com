## Build a complex rather than only a neighbor graph

A finite point cloud is not itself a continuous surface. To study its organization across scales we choose a filtration. Here a Vietoris–Rips simplex enters when every pair of its vertices has distance at most ε. Its filtration value is its diameter. We use edge-distance threshold ε directly, not a ball-radius convention with an extra factor of two.

Four rectangle vertices generate vertices, edges, triangles and ultimately a tetrahedron. A two-dimensional drawing of their positions does not limit the abstract complex to dimension two. The shaded structural diagram shows included triangles; the tetrahedron count is reported separately because its planar projection is ambiguous.

## Boundaries turn geometry into linear algebra

Over GF(2), a chain is a subset of simplices, and adding a simplex twice cancels it. The boundary map adds all codimension-one faces. Each face of a face occurs twice, giving ∂²=0. Therefore boundaries are cycles and homology is the quotient

$$
H_k=\ker\partial_k/\operatorname{im}\partial_{k+1},\qquad
\beta_k=n_k-\operatorname{rank}\partial_k-\operatorname{rank}\partial_{k+1}.
$$

The implementation constructs these boundary matrices and performs binary elimination. It does not infer hole counts from an image classifier or a preset list of Betti numbers.

## Follow a unit square through its thresholds

Before ε=1 there are four isolated vertices: β₀=4, β₁=0. At ε=1 all four sides enter, making one component with one unfilled cycle. The edge-boundary rank is three, so β₁=4−3=1. At ε=√2 the diagonals, all four triangles and the tetrahedron enter. The full simplex is contractible and β₁=β₂=0.

If only edges were retained, the complete graph would still contain graph cycles. If triangles were retained but the tetrahedron omitted, the tetrahedral boundary would create a false H₂ feature. The intended full Rips filtration requires the higher-dimensional cofaces.

## Persistence pairs birth and death

Order simplices by filtration value, then dimension so faces precede cofaces, and reduce boundary columns. A zero reduced column births a class; a later pivot pairs it with a death. The positive H₁ interval for the unit square is

$$
[1,\sqrt2),\qquad \text{lifetime}=\sqrt2-1.
$$

The diagram plots birth against death, omits zero-length pairs, and reports the essential H₀ interval separately as infinite. At a death threshold the corresponding finite interval is no longer alive. Equal-value tie order can change zero-length pairings without changing the meaningful persistent module.

## Compute the quotient through exhaustive chains

For a finite chain complex over a field, rank–nullity gives the cycle dimension nₖ−rank∂ₖ. Since ∂²=0, the boundary image lies inside that cycle space, so quotienting subtracts rank∂ₖ₊₁. This derives the displayed Betti formula. Over GF(2) one can independently enumerate all chains: the image has 2ʳ elements when the boundary rank is r. Counting distinct boundary masks therefore obtains ranks without reusing persistence column reduction.

The executable example does precisely that for the square, including all cofaces. At the last threshold there are six edges, four triangles and one tetrahedron. Boundary ranks are 3, 3, 1, giving Betti numbers (1,0,0). Removing the tetrahedron leaves a two-cycle in the abstract boundary surface even though the vertices are drawn in a plane.

| Threshold | Counts n₀,n₁,n₂,n₃ | Betti numbers β₀,β₁,β₂ |
|---|---|---|
| 0.9 | 4,0,0,0 | 4,0,0 |
| 1 | 4,4,0,0 | 1,1,0 |
| √2 | 4,6,4,1 | 1,0,0 |

Enumerating chains costs exponentially many states and is appropriate here only because there are four vertices. Production computations use sparse reduction and control the skeleton dimension. Computing Hₖ generally needs simplices through dimension k+1; using only a graph computes graph homology instead of full Rips homology.

## Reproduce the calculation in Python

```python
from itertools import combinations
from math import dist, log2, sqrt
points = [(0,0),(1,0),(1,1),(0,1)]
for eps in [.9, 1, sqrt(2)]:
    groups = [[s for s in combinations(range(4),d+1)
               if all(dist(points[i],points[j]) <= eps for i,j in combinations(s,2))]
              for d in range(4)]
    ranks = [0]
    for d in range(1,4):
        columns = [sum(1 << groups[d-1].index(face) for face in combinations(s,d)) for s in groups[d]]
        image = {0}
        for col in columns:
            image |= {x ^ col for x in image}
        ranks.append(int(log2(len(image))))
    betti = [len(groups[d])-ranks[d]-ranks[d+1] for d in range(3)]
    print(round(eps,6), list(map(len,groups)), betti)
```

## Exercise and interpretation

Set width one and height two. The hole is born at two and dies at √5, giving lifetime about 0.236. Verify the counts and inspect what is actually filled at the threshold. Scaling every distance by s scales births and deaths by s; Betti patterns occur at rescaled thresholds.

Persistent features describe this metric filtration. Inferring an underlying population manifold requires sampling, noise and reconstruction assumptions beyond four points. A long-lived hole can be useful evidence, but neither its existence nor its scientific cause is automatically certified by the diagram alone.

Further reading: [GUDHI, Rips Complex](https://gudhi.inria.fr/doc/latest/group__rips__complex.html).
