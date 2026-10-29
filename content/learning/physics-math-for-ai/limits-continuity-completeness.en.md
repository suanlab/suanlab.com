## A point and its neighborhood are different

A limit describes nearby inputs, separately from the assigned value at the point. The first model uses f(x)=x+1 for x≠1 and assigns f(1)=b. Even when b=0, nearby values approach two. Changing only b does not change that limit.

$$
\lim_{x\to a}f(x)=L\iff
\forall\varepsilon>0\ \exists\delta>0\ \forall x:
0<|x-a|<\delta\implies|f(x)-L|<\varepsilon.
$$

The puncture excludes a itself. Delta may depend on epsilon but must work for every input in the neighborhood. Shrinking a plotted window illustrates the definition; it does not prove the statement for every tolerance.

## Complete an epsilon–delta proof

For x≠1, |f(x)−2|=|x−1|. Given any ε>0, choose δ=ε:

$$
0<|x-1|<\varepsilon\implies |(x+1)-2|<\varepsilon.
$$

This rule works for an arbitrary tolerance. Assigning b=2 gives continuity; any other assignment gives a removable discontinuity. An open circle marks an excluded value and disappears when that value becomes the actual assigned value.

## A jump cannot be repaired at one point

The second model is zero to the left of one and one to the right. Different one-sided limits rule out a two-sided limit, regardless of b. A sufficiently small tolerance cannot place both side values in one neighborhood around a candidate limit.

Continuity requires a defined point value, existence of the relevant limit and equality of the two. At domain endpoints use limits relative to that domain rather than assuming both sides exist.

## Completeness supports existence

Real completeness can be stated as the least-upper-bound property. The rational set {q≥0:q²<2} is bounded above but has no rational supremum; its real supremum is √2.

$$
x_n\uparrow,\quad x_n\le M\quad\Longrightarrow\quad x_n\to\sup_n x_n\quad\text{in }\mathbb R.
$$

If s is the supremum, s−ε is not an upper bound, so some x_N exceeds it. Monotonicity then gives s−ε<xₙ≤s for every n≥N. This proves monotone bounded convergence using completeness, not a finite picture.

## AI interpretation and exercises

Continuity is pointwise control, distinct from uniform continuity over a whole domain. Floating-point representations are not the complete real number field. Separate mathematical limits from rounding behavior.

1. For b=3, the limit at one is **two**.
2. No assigned b repairs the **jump**.
3. For ε=0.01, **δ=0.01** works in the linear example.
4. Monotone but unbounded sequences need not have a finite limit: **xₙ=n** is a counterexample.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
