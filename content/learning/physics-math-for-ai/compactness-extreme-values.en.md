## A bound need not be attained

A supremum is the least upper bound; a maximum must also belong to the set. For the identity function, [0,1] attains maximum one, but (0,1) does not:

$$
\sup_{x\in(0,1)}x=1,\qquad 1\notin(0,1).
$$

Every x<1 has a larger candidate (x+1)/2 still in the domain. Completeness supports existence of a supremum; compactness supports attainment under continuity.

## Define compactness in the right space

Topologically, every open cover must have a finite subcover. In finite-dimensional Euclidean space, Heine–Borel gives

$$
K\subset\mathbb R^n\text{ compact}\iff K\text{ is closed and bounded}.
$$

Equivalently, every sequence in K has a subsequence converging within K. Do not export the closed-and-bounded characterization to arbitrary infinite-dimensional normed spaces.

## Continuous functions attain extrema

A continuous real function on nonempty compact K is bounded and attains its minimum and maximum. Its image is compact, hence closed and bounded in R. A sequence approaching the image supremum has that limit inside the closed image, proving attainment:

$$
\exists x_{\min},x_{\max}\in K:\quad
f(x_{\min})\le f(x)\le f(x_{\max})\quad\forall x\in K.
$$

Dropping an assumption can invalidate the general theorem. It does not mean every function outside those assumptions lacks extrema: a constant function attains them even on an open interval.

## Uniform continuity and optimization

Continuity on compact K also implies uniform continuity. On all of R the same conclusion fails, as x² illustrates at large inputs.

The extreme value theorem establishes existence, not uniqueness or an algorithm. A constant function has every point as an optimizer. Separate existence, uniqueness, stability and computation when reading a theorem.

## Read the endpoints and AI connection

Open circles exclude endpoints. In open-interval mode, probing x₀=1 gives —, not the supremum substituted as a function value. Domain choice changes attainment even with the same formula.

Continuous loss on a compact parameter set has a minimizer. Unconstrained settings may need alternatives such as coercivity. Existence alone does not prove gradient descent converges or that the minimizer generalizes.

## Exercises and answers

1. On (0,1), the identity has **infimum zero and no minimum**.
2. Compactness plus continuity **does not guarantee a unique optimizer**.
3. A constant function on an open interval **attains both extrema everywhere**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
