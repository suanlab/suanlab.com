## Covering and separation answer different questions

How many representatives approximate a complicated set? A cover places every target near a representative; a packing selects many mutually separated points. Specify the metric, target set, allowed centers and inequality conventions first.

The lab uses N vertices of a regular unit-radius polygon, with centers restricted to those vertices. Covering uses distance ≤ε, packing uses >ε. Unmarked parts of the continuous circle are not targets.

## Convert cyclic gaps into distances

A gap of k vertices has chord length 2sin(πk/N). If up to r steps on either side lie within ε, one center covers L=min(N,2r+1) vertices. At least ceil(N/L) centers are needed, and centers spaced L steps apart attain the bound.

$$
N_{\rm cover}=\left\lceil\frac N{\min(N,2r+1)}\right\rceil.
$$

A strict packing requires cyclic gaps of at least r+1, giving maximum floor(N/(r+1)), attained by regularly spaced choices. A one-point packing has no minimum pair distance, shown as a dash.

## Twelve vertices

At N=12,ε=.8, one-step distance is about .518 and two-step distance is one. Thus r=1, each center covers three vertices, four centers cover all vertices and six points form a packing. Filled orange dots mark cover centers and rings packing points. Overlap produces concentric marks and is also recorded in the table.

At ε=2, one center covers everything. Strict separation greater than two permits only one packing point. Changing ≤ or > at the boundary changes the problem.

## Why nets enter high-dimensional proofs

General proofs can apply a tail bound at each net point, pay a union-bound cost, then use continuity or Lipschitz control to reach nearby points. Larger nets make that cost larger. The finite polygon's formulas are not covering numbers for a high-dimensional ball: dimensions, norms, sets and center restrictions differ.

## Exercises with explanations

1. How many centers at N=12,ε=.1? **Twelve: each covers only itself.**
2. Must cover and packing counts agree? **No: their objectives differ.**
3. Does the picture certify a cover of the continuous circle? **No: only finite vertices are checked.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
