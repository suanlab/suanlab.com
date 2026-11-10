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

## Exercise and explained answer

Select the edge midpoint (1,0). Predict orbit size four and stabilizer size two: identity and reflection across its horizontal axis fix it. Load the point mode and verify the entire table, not only the selected orange point. Then move to the center and explain why a large stabilizer corresponds to a small orbit.

## Connection to learning

Weight sharing and data augmentation depend on an action and its orbits. Samples related by symmetry can share a prediction rule only when that transformation preserves the intended task. A reflection may preserve object category while changing a handedness label. Before declaring a symmetry, specify what transforms, which labels should change, and whether the proposed transformations really form a group under composition.

Further reading: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
