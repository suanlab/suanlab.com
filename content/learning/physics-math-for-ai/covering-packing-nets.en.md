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

## From finite vertices to a net of a continuous sphere

Now consider a different target from the finite polygon above: the entire Euclidean unit sphere S^(d−1) in R^d, with centers on that sphere. An ε-net N, for 0<ε<1/2, contains a point u within distance ε of every unit vector x.

Choose sphere points mutually farther than ε until none can be added. Open balls of radius ε/2 around them are disjoint and lie inside a ball of radius 1+ε/2. Comparing volumes gives finiteness and the following bound. Maximality makes every sphere point lie within ε of a selected point, so the set is a net.

$$
|\mathcal N|\left(\frac\varepsilon2\right)^d
\leq\left(1+\frac\varepsilon2\right)^d,
\qquad |\mathcal N|\leq(1+2/\varepsilon)^d.
$$

Ambient dimension enters the exponent. This bound does not identify the exact minimum size and is different from the finite polygon formula.

## Control a symmetric matrix through finitely many quadratic forms

For a real symmetric A, ||A||₂=max_{||x||=1}|xᵀAx|. The absolute value captures large negative eigenvalues as well as positive ones. Choose a net point u near a maximizing direction x and expand the difference.

$$
|x^\top Ax-u^\top Au|
\leq |(x-u)^\top Ax|+|u^\top A(x-u)|
\leq2\varepsilon\|A\|_2.
$$

Writing Q_N=max_{u∈N}|uᵀAu| yields

$$
Q_{\mathcal N}\leq\|A\|_2\leq\frac{Q_{\mathcal N}}{1-2\varepsilon}.
$$

The denominator must be positive, requiring ε<1/2. This argument does not authorize applying the same quadratic-form formula to arbitrary nonsymmetric matrices. A common probabilistic strategy first controls a fixed net by a union bound, then uses this deterministic extension to reach the whole sphere.

## Check an explicit continuous-circle net

In two dimensions, n equally spaced circle points cover the entire circle with radius ε_n=2sin(π/(2n)). A farthest point is the arc midpoint between neighboring centers. At n=8, ε_n≈0.39018 satisfies the required range. This is separate from the screen's vertex-only target.

For A=((2,1),(1,−1)), eigenvalues are (1±√13)/2 and the norm is approximately 2.30278. The table gives the finite-net Q and its certified Upper=Q/(1−2ε_n). Decimals are rounded.

| n | Q | Upper |
|---|---|---|
| 8 | 2.00000 | 9.10586 |
| 16 | 2.26777 | 3.73030 |
| 64 | 2.30277 | 2.55343 |

A denser net brings the correction factor closer to one while requiring more evaluations. Treating Q itself as the exact norm underestimates it as two at n=8. The code's dense angle check is an additional numerical check; the continuous covering guarantee comes from the arc-midpoint argument above.

```python
from math import sin, cos, pi, sqrt

A = [[2, 1], [1, -1]]
exact = (1+sqrt(13))/2
for n in [8, 16, 64]:
    radius = 2*sin(pi/(2*n))
    assert radius < .5
    net = [(cos(2*pi*j/n), sin(2*pi*j/n)) for j in range(n)]
    values = [abs(sum(u[i]*A[i][j]*u[j] for i in range(2) for j in range(2)))
              for u in net]
    Q = max(values)
    upper = Q/(1-2*radius)
    assert Q <= exact+1e-12 and exact <= upper
    for k in range(2001):
        theta = 2*pi*k/2001
        nearest = min(sqrt((cos(theta)-u[0])**2+(sin(theta)-u[1])**2) for u in net)
        assert nearest <= radius+1e-12
    print("n, radius, finite maximum, upper, exact norm:", n, radius, Q, upper, exact)
```

## Questions for research use

1. What extends finite checks to a whole sphere? **A covering radius and a bound on function variation.**
2. Can ε=.8 be inserted into this denominator formula? **No: the quadratic-form correction requires ε<1/2.**
3. What does a probabilistic net argument additionally specify? **A fixed net's size, pointwise failure probability and union bound.**

For background, see [Vershynin, §5.2.2, Lemmas 5.2–5.4](https://arxiv.org/html/1011.3027#S2.SS2). The positive-denominator condition is explicit, and the 2×2 example is computed directly.

## Exercises with explanations

1. How many centers at N=12,ε=.1? **Twelve: each covers only itself.**
2. Must cover and packing counts agree? **No: their objectives differ.**
3. Does the picture certify a cover of the continuous circle? **No: only finite vertices are checked.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
