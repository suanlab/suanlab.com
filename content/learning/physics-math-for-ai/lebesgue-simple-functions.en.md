## Read levels of values rather than only horizontal intervals

A simple function is measurable and takes finitely many values. On disjoint measurable level sets, its integral sums each value times the measure of its set. This connects to rectangular-area intuition without requiring the sets to be intervals. The lab uses Lebesgue measure on [0,1] and f(x)=x², rounding function values downward to discrete levels.

$$
s=\sum_{k=1}^K a_k\mathbf1_{A_k},\qquad\int s\,d\mu=\sum_{k=1}^K a_k\mu(A_k)\quad(a_k\ge0).
$$

## Calculate actual level-set sizes

At m=2 the value spacing is one quarter. Define the approximation using the floor function. The kth level occupies an x interval between the square roots of k/2^m and (k+1)/2^m. These widths differ because the partition is based on function values, not a uniform x grid. At x=0.6 the true value is 0.36 and the approximation is 0.25.

$$
s_m(x)=2^{-m}\lfloor2^m x^2\rfloor,\quad\mu(A_k)=\sqrt{\frac{k+1}{2^m}}-\sqrt{\frac{k}{2^m}}.
$$

## Define the nonnegative Lebesgue integral

For a general nonnegative measurable function, take the supremum of integrals of nonnegative simple functions below it. The value may be infinite. In this example the dyadic approximations increase to x², and monotone convergence gives integrals approaching one third. The finite slider does not prove the general theorem; explicit formulas and an error bound verify this example.

$$
\int f\,d\mu=\sup_{0\le s\le f,\ s\text{ simple}}\int s\,d\mu,\qquad0\le\int_0^1x^2dx-\int_0^1s_mdx\le2^{-m}.
$$

## Endpoints and signed integrals

At x=1 every approximation equals one. For m=0 it is zero everywhere else, so its integral is zero. Filled and open dots distinguish endpoint conventions; the additional singleton level contributes zero measure. For signed functions, integrate positive and negative parts separately. If both integrals are infinite, their difference is undefined. Absolute integrability ensures a finite real expectation.

$$
f=f^+-f^-,\quad |f|=f^++f^-,\qquad\int|f|\,d\mu<\infty\implies\int f\,d\mu\in\mathbb R.
$$

## Exercises with explanations

1. What are the value at x=1 and integral for m=0? **One and zero.**
2. Does refinement decrease the lower approximation? **No: it increases monotonically.**
3. Is an integral simply an average over listed points? **No: it uses the specified measure.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
