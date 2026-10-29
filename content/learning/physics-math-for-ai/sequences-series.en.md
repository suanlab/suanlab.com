## Terms and accumulated sums are different sequences

A sequence assigns a value to each integer index. A series asks about adding its terms through partial sums and then taking a limit. A term becoming small is not by itself enough for the accumulated sum to approach a finite value.

We begin with a geometric sequence uₙ=arⁿ for n≥0 and define S₀=0, S_N as the sum of the first N terms. The index N counts terms, so the last included term is u_(N−1), not u_N. This indexing distinction prevents a common off-by-one error.

## Derive the finite geometric sum

Write S_N=a+ar+…+ar^(N−1). Multiplying by r and subtracting cancels the interior terms:

$$
(1-r)S_N=a-ar^N,
\qquad S_N=\frac{a(1-r^N)}{1-r},\quad r\ne1.
$$

For r=1 the sum is aN; the displayed fraction cannot be used by dividing zero by zero. For N=0, the empty sum is zero. The lab accumulates the terms directly so the readout need not subtract nearly equal numbers in this closed form near r=1.

At a=1,r=1/2,N=10, S_N=2(1−2⁻¹⁰)=1.998046875. It is close to two, not equal to two.

## Pass to an infinite limit under conditions

For a≠0, a geometric series converges exactly when |r|<1. Then rᴺ tends to zero and

$$
\sum_{n=0}^{\infty}ar^n=\frac a{1-r}.
$$

The omitted tail after N terms is arᴺ/(1−r), whose absolute value quantifies the approximation. At the reference settings the tail is 1/512. This bound supplies more information than a curve merely looking flat.

If a=0, every term is zero for any finite r and the series is zero, including |r|≥1. State that degenerate case separately rather than applying a slogan with a hidden nonzero assumption.

## Oscillation and divergence

At a=1,r=−1, partial sums alternate 0,1,0,1,… and have no ordinary limit. They are bounded but not convergent. Other summation conventions can assign values to some divergent expressions, but the lab uses the ordinary partial-sum definition and does not silently substitute a different convention.

At r=1.1 and a=1, positive terms grow and partial sums increase without a finite limit. A finite computed prefix still exists. Calculating thirty terms successfully is not evidence that the infinite sum exists.

## A necessary condition is not sufficient

If S_N converges, then u_N=S_(N+1)−S_N tends to zero. The converse is false. In the harmonic series, terms 1/n tend to zero but the sum diverges. Group terms from n=2^(k−1)+1 to 2^k. There are 2^(k−1) terms, each at least 1/2^k, so each block contributes at least 1/2. Arbitrarily many such blocks rule out a finite total.

This is a general argument about all blocks, not an extrapolation from a finite plot. It connects the earlier implication and proof units to convergence claims.

## Read the experiment and its limits

Blue shows S_k for k=0 through N. Orange is the theoretical infinite sum only when it exists. N=0 leaves the single empty-sum point; it does not fabricate an added term. The graph computes only geometric examples, while the harmonic argument is a separate counterexample in the text.

Changing N changes how much of the sequence is inspected. It does not change r, a or the mathematical existence of the infinite limit. Exact formulas and finite-precision computations also differ: large or nearly cancelling terms can cause numerical error even when a formal identity is valid.

## AI and numerical connections

Truncated expansions, discounted returns and iterative approximations require a tail or convergence argument. A discounted sum with bounded rewards has a geometric absolute bound when the discount magnitude is below one. Undiscounted or unbounded cases need additional assumptions.

A finite list of decreasing training losses does not imply the iterates converge, and convergence of scalar values need not imply convergence to the desired model. Specify which sequence is claimed to converge and in what sense before borrowing a geometric-series intuition.

## Exercises and reference

1. Compute the first three terms and sum at a=2,r=1/2. **Answer:** 2,1,1/2; sum 3.5.
2. Find the infinite sum and tail after three terms. **Answer:** 4 and 0.5.
3. Explain why bounded partial sums need not converge using r=−1.
4. Reconstruct the harmonic-series block proof, identifying the quantified arbitrary number of blocks.

For sequences and series see [OpenStax Algebra and Trigonometry 2e](https://openstax.org/details/books/algebra-and-trigonometry-2e). The geometric formulas and harmonic counterexample are worked with their ordinary convergence definitions here.
