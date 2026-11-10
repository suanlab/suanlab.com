## A function limit versus an integral limit

Pointwise convergence alone does not ensure convergence of integrals. Specify both the convergence notion and the integrating measure. Here all integrals use Lebesgue measure on [0,1]. The first sequence x to the nth power tends to zero for x<1 but remains one at x=1. Its limiting function is zero almost everywhere and has integral zero.

$$
\int_0^1x^n\,dx=\frac1{n+1}\longrightarrow0.
$$

## A common bound for dominated convergence

Dominated convergence assumes measurable functions converging almost everywhere and a single integrable g bounding every absolute value. Then L1 distance to the limit tends to zero, and the integrals converge. For x^n, use g=1. Separate integrability of each term, or a different bound for each n, is not enough. The hypotheses concern the whole sequence, beyond the finite terms shown by a slider.

$$
f_n\to f\text{ a.e.},\quad |f_n|\le g\in L^1\implies\int|f_n-f|\to0.
$$

## A concentrating-mass counterexample

The second sequence has height n on the open interval (0,1/n) and zero elsewhere. Every fixed positive x eventually leaves the support, and the value at zero is always zero. The pointwise limit is therefore zero. Yet width times height is always one, so the integral limit is one. An integrable common dominator cannot exist, since it would contradict dominated convergence.

$$
f_n=n\mathbf1_{(0,1/n)},\qquad\lim_n\int f_n=1\ne0=\int\lim_n f_n.
$$

## Monotone convergence and Fatou

For an increasing nonnegative sequence, monotone convergence permits exchanging limits and integrals, including infinite values. The third example is min(n,1/√x) for x>0 and zero at zero. Its integral is 2−1/n, approaching two. Dominated convergence also applies here because 1/√x is integrable. The graph omits the part of the limit curve above height 2n near zero; the omitted part is not zero. Fatou supplies a one-sided bound for general nonnegative sequences.

$$
\int\liminf_n f_n\le\liminf_n\int f_n,\qquad 0\le f_n\uparrow f\implies\int f_n\uparrow\int f.
$$

## Exercises with explanations

1. Is the pointwise limit of x^n zero everywhere? **No: at one it is one.**
2. Does the concentrating example violate Fatou? **No: zero is at most one.**
3. What is the third example’s integral at n=4? **1.75.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
