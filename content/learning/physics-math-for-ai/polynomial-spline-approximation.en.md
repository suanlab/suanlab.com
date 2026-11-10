## A representation differs from a chosen approximation rule

On [0,1], x² is already a quadratic, so best degree-two polynomial error is zero. A particular approximation procedure need not choose those optimal coefficients. We compare two concrete rules: the Bernstein operator and uniform-knot linear interpolation.

Bernstein order n names the nominal basis order. Its result on this target simplifies to a quadratic even when n is larger; the operator still uses n+1 sample weights.

## Interpret Bernstein weights as probabilities

For K~Binomial(n,x), Bₙf(x)=E[f(K/n)]. Apply the second-moment identity:

$$
\begin{aligned}
B_nf(x)&=\sum_{k=0}^n f(k/n){n\choose k}x^k(1-x)^{n-k},\\
B_n(x^2)&=x^2+\frac{x(1-x)}n.
\end{aligned}
$$

Error is nonnegative and maximal at the midpoint, with value 1/(4n). Orders four and eight give 1/16 and 1/32. These are errors of this procedure, not best polynomial errors.

## Local linear splines have a different rate

Connect target values at knots xⱼ=j/n. Convexity puts each segment above the target:

$$
S_nf(x)-x^2=(x-x_j)(x_{j+1}-x),\quad x\in[x_j,x_{j+1}].
$$

The midpoint maximum is h²/4=1/(4n²). Four intervals give 1/64; eight give 1/256. Blue is the target and orange the chosen approximation. The dot marks the query. Zero error at knots does not eliminate between-knot error.

## General rates require regularity

Approximation existence and convergence rate are different claims. The linear-interpolation bound h²‖f''‖∞/8 needs a bounded second derivative. Corners and discontinuities need separate analysis. General Bernstein convergence also uses continuity and domain assumptions.

For AI comparisons, distinguish hypothesis space, coefficient-selection algorithm, data and evaluation norm. The same number of stored values can support different global or local constructions.

## Exercises with explanations

1. What happens at n=1? **Both rules give x**, the line through the endpoints.
2. Does nonzero Bernstein error mean x² is outside the degree-four space? **No.** The procedure does not select the best element.
3. What does evaluation only at spline knots miss? **Between-knot error**, maximal at each interval midpoint here.

Further reading: [MIT 18.102, Functional Approximation Background](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
