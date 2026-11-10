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

## Exercise and interpretation

Set width one and height two. The hole is born at two and dies at √5, giving lifetime about 0.236. Verify the counts and inspect what is actually filled at the threshold. Scaling every distance by s scales births and deaths by s; Betti patterns occur at rescaled thresholds.

Persistent features describe this metric filtration. Inferring an underlying population manifold requires sampling, noise and reconstruction assumptions beyond four points. A long-lived hole can be useful evidence, but neither its existence nor its scientific cause is automatically certified by the diagram alone.

Further reading: [GUDHI, Rips Complex](https://gudhi.inria.fr/doc/latest/group__rips__complex.html).
