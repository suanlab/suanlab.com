## A quotient is not its limit

The forward quotient requires h≠0. The derivative is a limit, not a substitution of zero:

$$
f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}h.
$$

For x² the quotient is 2x₀+h. At x₀=0.5,h=0.1 it equals 1.1, while the derivative equals one. A zero step displays — rather than a fabricated quotient.

## Differentiability is stronger than continuity

Differentiability implies continuity because the increment is h times a convergent quotient. The converse fails: |x| is continuous at zero but has left slope minus one and right slope one. The lab omits a nonexistent tangent instead of calling its slope zero.

$$
f(x_0+h)=f(x_0)+f'(x_0)h+o(|h|).
$$

This is a local first-order approximation: its error becomes negligible relative to |h|. It does not assert global linearity or exact predictions for finite changes.

## State the mean value theorem precisely

If f is continuous on [a,b] and differentiable on (a,b), some c in the interior satisfies

$$
f'(c)=\frac{f(b)-f(a)}{b-a}.
$$

Subtract the line joining the endpoints and apply Rolle’s theorem. Rolle’s argument uses attainment of extrema on a compact interval and the zero derivative at an interior differentiable extremum. These assumptions support existence, not merely notation.

## Same average, different hypotheses

Both x² and |x| have average slope zero on [−1,1]. For x² the witness is c=0. For |x|, every existing interior derivative is minus one or one; at zero no derivative exists. There is no derivative equal to the average slope, and the theorem’s differentiability assumption fails.

## Sensitivity bounds and exercises

A uniform derivative bound |f′|≤L throughout an interval gives |f(x)−f(y)|≤L|x−y| by the theorem. A small derivative at one point does not establish a whole-region bound. Name the region when discussing AI loss or input sensitivity.

Finite differences have truncation error for large steps and cancellation or rounding error for very small steps. One successful comparison is not a general differentiability proof.

1. For x² at x₀=1,h=0.01, the quotient is **2.01**.
2. |x| fails the MVT hypothesis of **interior differentiability**.
3. An everywhere-zero derivative on an interval implies **a constant function**, by the MVT.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
