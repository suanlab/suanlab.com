## Functions need a chosen notion of distance

Functions can be added and scaled, so they form vector spaces. To say an approximation is close, choose a norm. L1 integrates absolute magnitude, L2 takes the square root of integrated squared magnitude, and L∞ measures essential supremum. Lp identifies functions differing only on null sets.

$$
\|f\|_1=\int|f|,\qquad \|f\|_2^2=\int|f|^2.
$$

## Completeness differs from inner-product geometry

A Banach space contains limits of all Cauchy sequences in its norm. A Hilbert space is complete in an inner-product norm. L1 and L∞ are Banach but generally not Hilbert. For real L2 functions, the inner product is integral fg.

Inner-product norms satisfy the parallelogram law. For disjoint unit vectors, its left side minus right side is four in L1, zero in L2 and minus two in L∞. Completeness alone does not supply orthogonal geometry.

$$
\|u+v\|^2+\|u-v\|^2=2\|u\|^2+2\|v\|^2.
$$

## Compute a sequence and its limit

On [0,1], a geometric-series identity gives

$$
\begin{aligned}
f_m(x)&=\sum_{k=0}^{m}\frac{x^k}{2^{k+1}},\\
f(x)-f_m(x)&=\frac{(x/2)^{m+1}}{2-x},\\
\|f-f_m\|_\infty&=2^{-(m+1)},\quad f(x)=\frac1{2-x}.
\end{aligned}
$$

Degrees two and six give uniform errors 1/8 and 1/128. Since the interval has unit measure, L1 and L2 errors are bounded by the uniform error. Limit norms are log 2, 1/√2 and one. Switching norms changes the measurement, not the blue partial polynomial or orange limit curve.

## The wrong space can lose the limit

Polynomials with uniform norm are incomplete: this Cauchy sequence has a nonpolynomial limit. Continuous functions with uniform norm are complete. Continuous functions with L1 norm are not: a continuous ramp rising from zero to one over width 1/m after x=.5 approaches a step in L1 with error 1/(2m), for m≥2. No continuous function equals that step almost everywhere. **Specify both the set and its norm.** The lab's Lp selections refer to complete Lp spaces, while plotted examples have continuous representatives.

For AI approximation, small average loss differs from uniformly small error. State the input measure and evaluation norm.

## Exercises with explanations

1. What is the degree-zero L1 norm? **1/2:** the function is constant 1/2.
2. Does a nonpolynomial limit mean convergence failed? **No.** Convergence holds in a larger complete space, while the polynomial space lacks the limit.
3. Does changing one point alter an L2 norm? **No.** This leads to the point-evaluation problem in the dual-space chapter.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
