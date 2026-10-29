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

## Exercises with explanations

1. What is the averaged x² at (2,1)? **Also 2.5.**
2. Does zero average imply zero input? **No: linear features provide a counterexample.**
3. What happens if an invariant is averaged again? **It stays fixed: P²=P.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
