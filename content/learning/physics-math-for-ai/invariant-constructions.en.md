## What an invariant preserves

Let C4 act on the real plane by quarter-turn rotations. A function F is invariant when every allowed rotation leaves its output unchanged. The point can move while the function value stays fixed. Choose the original feature as x, x² or xy. Evaluate it at all four rotated positions and average those values to construct a new invariant.

$$
F(gv)=F(v)\quad(g\in C_4),\qquad P\phi(v)=\frac14\sum_{k=0}^3\phi(R^kv).
$$

## Why averaging gives invariance

Rotating the input shifts the four summands, returning the final rotation to the first term. The sum therefore stays unchanged. For any finite group, right multiplication permutes its elements and gives the same argument. This average uses real scalars and division by the group order. In a finite field whose characteristic divides that order, the denominator may fail to be invertible.

$$
P\phi(hv)=\frac1{|G|}\sum_{g\in G}\phi(ghv)=P\phi(v).
$$

## Predict and inspect actual values

At v=(1,2), the feature x² gives values 1,4,1,4 and mean 2.5. For arbitrary coordinates, its average is half the squared Euclidean norm. Choosing x or xy instead cancels positive and negative terms to zero. Blue values show individual features and orange values show their common average. Integer rotation indices are not interpolated as continuous time or angles.

$$
P(x^2)=\frac{x^2+y^2}{2},\qquad P(x)=P(xy)=0.
$$

## Information loss and equivariance

The average coordinates are always (0,0), erasing even nonzero inputs. Invariance alone does not make a feature complete or capable of separating all orbits. Distinct C4 orbits on the same circle can share the same averaged x². Removing orientation may suit some classification tasks, whereas location prediction can require outputs that rotate with inputs. The latter is equivariance rather than invariant output.

$$
\text{invariant: }F(gv)=F(v),\qquad\text{equivariant: }F(gv)=gF(v).
$$

## Construct a feature that separates every C4 orbit

The radial average cannot distinguish (1,2) from (2,1), although neither is a quarter-turn of the other. Identify a real point with the complex number z=x+iy. A quarter-turn multiplies z by i, so its fourth power is unchanged. Keep both real components of this fourth power:

$$
A=x^4-6x^2y^2+y^4,\qquad B=4xy(x^2-y^2),\qquad F_4(v)=(A,B).
$$

| Input | Average of x² | Fourth-power feature (A,B) |
|---|---|---|
| (1,2) | 5/2 | (−7,−24) |
| (2,1) | 5/2 | (−7,24) |

This feature separates **all** C4 orbits, not just these points. If z and z′ are nonzero with equal fourth powers, then (z′/z)⁴=1. Its four complex roots are 1,i,−1,−i, so the points differ by a quarter-turn. If z=0, equality of fourth powers forces z′=0 too. The converse follows directly from invariance. Reflection changes the sign of B in general: these are C4 invariants, not D4 invariants. In particular, swapping the two coordinates relates the table's inputs by a square reflection, so a D4-invariant feature should identify them.

## Separating orbits does not mean generating every polynomial

Let s=x²+y². It is invariant too. Although A and B separate orbits, s cannot be a polynomial in A and B. Every nonconstant monomial in A and B has degree a positive multiple of four, whereas s is homogeneous of degree two. More explicitly, restricting a supposed polynomial identity to (x,y)=(t,0) would express t² as a polynomial in t⁴, which is impossible by comparison of polynomial coefficients. As a function, s can still be recovered using a nonpolynomial root:

$$
A^2+B^2=s^4,\qquad s=(A^2+B^2)^{1/4}\quad(s\ge0).
$$

Thus orbit separation and generation of an invariant polynomial ring are different claims. We have proved the former for this feature, without asserting a general invariant-ring algorithm.

## Information can be complete while gradients are poorly scaled

Scaling v by a real t scales F4(v) by t⁴. At the origin every first derivative of F4 vanishes. Along the x-axis, an input displacement ε produces a feature displacement ε⁴, so there is no inverse Lipschitz bound near the origin, even after identifying C4-related inputs. Exact orbit separation therefore does not guarantee stable recovery from noisy features or useful gradients. Adding lower-degree invariant features such as s can improve this particular radial scaling, but prediction performance still depends on the task.

$$
F_4(tv)=t^4F_4(v),\qquad DF_4(0)=0.
$$

## Verify the feature and its finite illustration

This standard-library code uses exact integers and rational scaling. The 25-point square grid is closed under C4; its seven feature classes are exactly its seven orbits. That finite enumeration illustrates the general proof above. The interactive bars on this page still show the selected x, x² or xy average; the quartic feature is the separate executable extension below.

```python
from fractions import Fraction as Q
from itertools import product

def turn(v):
    x, y = v
    return (-y, x)

def orbit(v):
    points = []
    for _ in range(4):
        points.append(v)
        v = turn(v)
    return set(points)

def feature(v):
    x, y = v
    return (x**4 - 6*x*x*y*y + y**4, 4*x*y*(x*x-y*y))

def radial(v):
    x, y = v
    return Q(x*x + y*y) / 2

v, w = (1, 2), (2, 1)
assert radial(v) == radial(w) == Q(5, 2)
assert w not in orbit(v)
assert feature(v) == (-7, -24) and feature(w) == (-7, 24)
groups = {}
for point in product(range(-2, 3), repeat=2):
    key = feature(point)
    groups.setdefault(key, set()).add(point)
    assert all(feature(q) == key for q in orbit(point))
    x, y = point
    assert sum(a*a for a in key) == (x*x+y*y)**4
assert len(groups) == 7
assert all(points == orbit(next(iter(points))) for points in groups.values())
for t in [Q(1, 2), Q(1, 8)]:
    assert feature(tuple(t*a for a in v)) == tuple(t**4*a for a in feature(v))
    print("scale:", t, "quartic feature:", feature(tuple(t*a for a in v)))
print("radial:", radial(v), radial(w), "quartic:", feature(v), feature(w))
print("grid points:", 25, "orbits:", len(groups))
```

## Further questions and research use

1. Does retaining only A separate these two inputs? **No: both have A=−7, and their B values distinguish them.**
2. Does an invariant feature preserve the original orientation? **It preserves an orbit, so orientations related by the chosen group are deliberately identified.**
3. What should a learning experiment report? **The chosen group, the feature's separation claim, behavior near zero and predictive results separately. A finite grid check alone proves no general separation theorem.**

Review [complex coordinates](../real-complex-numbers/) for the fourth-power proof and [algebraic representations](../algebraic-representations-scope/) for outputs that transform with the input.

## Exercises with explanations

1. What is the averaged x² at (2,1)? **Also 2.5.**
2. Does zero average imply zero input? **No: linear features provide a counterexample.**
3. What happens if an invariant is averaged again? **It stays fixed: P²=P.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
