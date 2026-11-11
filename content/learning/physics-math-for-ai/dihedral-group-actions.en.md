## A group is a composition rule for reversible transformations

The square has eight rigid symmetries: four quarter-turn rotations and four reflections. They form D4, where this notation means the order-eight square group. Some books use D8 for the same group, so always check the convention. Closure, associativity, identity and inverses concern composition of transformations, not just a list of visually symmetric pictures.

Let R rotate counterclockwise by π/2 and S reflect across the horizontal axis. Every element has a unique form RᵏSᵇ with k=0,1,2,3 and b=0,1. Matrices act on column vectors, so the rightmost transformation acts first.

$$
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad
S=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
R^4=S^2=I,\quad SRS=R^{-1}.
$$

The last relation explains noncommutativity: a reflection reverses the sense of rotation. In pair notation the product is (k,b)(l,c)=(k+(−1)ᵇl mod 4,b+c mod 2). The implementation evaluates this rule and the actual matrices independently in the numerical tests.

## An action tells us what the group transforms

A left action on X satisfies e·x=x and (gh)·x=g·(h·x). Here X is the physical plane and the action is matrix multiplication. The blue square gives geometric context; the blue dots collect all distinct images of the selected source point. Orange marks the currently selected group element's image.

The same abstract group can act on pixels, vectors, sets or function spaces. Choosing the action is part of the model. Counting eight group elements does not imply that a particular point must have eight different images.

## Orbit and stabilizer count different things

The orbit is the set of reachable images. The stabilizer is the subgroup of transformations fixing the source point. For a finite group,

$$
|Gx|=\frac{|G|}{|G_x|},\qquad
G_x=\{g:g\cdot x=x\}.
$$

To understand the quotient, group elements producing the same image form a coset of the stabilizer. Thus every image is produced by the same number of elements. At a vertex, four distinct images each have two preimages. At the center, all eight elements produce the same point. A generic point such as (0.7,0.3) has no nonidentity stabilizer and therefore eight images.

## Count configurations with Burnside's lemma

![Six distinct symmetry orbits of square-vertex colorings and the eight fixed-coloring counts whose average is six.](/assets/learning/symmetry-burnside-en.svg)

An action can transform whole configurations, not only a single point. Color the four square vertices with two labeled colors. There are sixteen assignments, but rotation and reflection identify some assignments. Count pairs (g,x) with gx=x in two ways. Summing over g counts its fixed configurations; summing over each orbit counts its size times its stabilizer size, which is |G|. Consequently

$$
|X/G|=\frac1{|G|}\sum_{g\in G}|\operatorname{Fix}(g)|.
$$

For a vertex permutation with c cycles, an invariant coloring is constant along each cycle, giving 2ᶜ fixed assignments. The table sums to 48, so there are six orbits. Adjacent and opposite pairs of orange vertices have the same number of orange vertices, yet belong to different orbits. Counting colors is therefore an incomplete invariant. The exact code independently enumerates orbit representatives and checks the average.

| Symmetry type | Number of elements | Fixed colorings per element |
|---|---:|---:|
| Identity | 1 | 16 |
| Quarter turns | 2 | 2 |
| Half turn | 1 | 4 |
| Reflections fixing two vertices | 2 | 8 |
| Reflections fixing no vertex | 2 | 4 |

This is a separate configuration action, while the interactive controls retain the original point action. Data augmentation only covers the orbit induced by the actual transformation law; collapsing distinct configuration orbits can erase task information.

## Reproduce the calculation in Python

```python
from itertools import product
vertices = [(1,1),(-1,1),(-1,-1),(1,-1)]
permutations = []
for reflect in [False, True]:
    for k in range(4):
        permutation = []
        for x,y in vertices:
            if reflect: y = -y
            for _ in range(k): x,y = -y,x
            permutation.append(vertices.index((x,y)))
        permutations.append(permutation)
colorings = list(product([0,1], repeat=4))
def act(p, c):
    out = [0]*4
    for i,j in enumerate(p): out[j] = c[i]
    return tuple(out)
fixed = [sum(act(p,c)==c for c in colorings) for p in permutations]
representatives = {min(act(p,c) for p in permutations) for c in colorings}
assert sum(fixed) == 8*len(representatives) == 48
print(fixed, 'orbits:', len(representatives))
for r in sorted(representatives): print(r)
```

## Exercise and explained answer

Select the edge midpoint (1,0). Predict orbit size four and stabilizer size two: identity and reflection across its horizontal axis fix it. Load the point mode and verify the entire table, not only the selected orange point. Then move to the center and explain why a large stabilizer corresponds to a small orbit.

## Connection to learning

Weight sharing and data augmentation depend on an action and its orbits. Samples related by symmetry can share a prediction rule only when that transformation preserves the intended task. A reflection may preserve object category while changing a handedness label. Before declaring a symmetry, specify what transforms, which labels should change, and whether the proposed transformations really form a group under composition.

Further reading: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
