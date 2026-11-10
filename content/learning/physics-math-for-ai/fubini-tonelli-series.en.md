## Double integrals require interchange conditions

On product spaces, Tonelli permits iterated integration of nonnegative measurable functions, including an infinite common value. For signed functions, finite absolute integral gives the usual Fubini guarantee. Here both spaces are positive integers with counting measure, hence sigma-finite, and integrals become double sums. Distinguish the displayed finite table from the specified infinite array.

$$
f\ge0\implies\iint f=\int\!\left(\int f\,d\nu\right)d\mu=\int\!\left(\int f\,d\mu\right)d\nu.
$$

## Finite rectangles always agree

In signed mode the diagonal is one, the superdiagonal minus one and other entries zero. The 4×4 truncation leaves its final diagonal entry unmatched, giving total one. Adding a fifth column cancels every row, giving zero. Within either rectangle, summing by rows or columns agrees. This is not a failure of finite addition.

$$
\sum_{i=1}^N\sum_{j=1}^Ma_{ij}=\sum_{j=1}^M\sum_{i=1}^Na_{ij}.
$$

## Infinite iterated sums use different limits

Every infinite row sums to zero, so the sum of row sums is zero. The first column sums to one, while later columns cancel, giving column-sum total one. Both the overall positive and negative sums are infinite, so a signed product integral is undefined. Finite values for the two iterated sums do not replace absolute integrability.

$$
\sum_i\left(\sum_j a_{ij}\right)=0,\quad\sum_j\left(\sum_i a_{ij}\right)=1,\quad\sum_{i,j}|a_{ij}|=\infty.
$$

## What changes after removing negative signs?

Nonnegative mode puts plus one on both diagonals. Both infinite iterated sums and the product integral equal infinity; this is fully consistent with Tonelli. The finite 4×4 total seven is a truncation, not a finite value assigned to the infinite sum. When interchanging expectations across batches, latent variables or time, distinguish nonnegative losses from signed gradients requiring integrability checks.

$$
\iint|f|\,d(\mu\times\nu)<\infty\implies\text{the signed Fubini interchange is valid}.
$$

## Exercises with explanations

1. What is the signed 4×5 total? **Zero.**
2. Why do the infinite iterated sums disagree? **The limit orders differ and absolute summability fails.**
3. May a nonnegative infinite sum equal infinity under Tonelli? **Yes.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
