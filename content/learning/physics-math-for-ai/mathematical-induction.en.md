## Infinite scope needs a connecting argument

Checking many integers can reveal a pattern but does not link the unchecked cases. Mathematical induction proves an infinite family by establishing a starting case and a rule connecting an arbitrary case to its successor. The word “arbitrary” is the critical part: checking the transition from 4 to 5 is not the same as proving it for every k.

We use ℕ={0,1,2,…}. A claim starting at one needs its own starting point. The lab includes a true sum identity, a false power inequality and a prime-generating pattern that fails only after many successful cases.

## State the induction principle

For a predicate P(n) on nonnegative integers, establish

$$
P(0),\qquad\forall k\ge0:\ P(k)\Rightarrow P(k+1).
$$

Then P(n) holds for every n≥0. The inductive hypothesis is a temporary assumption about an arbitrary k used to prove the conditional step. It is not permission to assume the final conclusion for every n.

Intuitively, a hypothetical least failing integer cannot be zero and cannot follow a successful predecessor. Making that argument relies on the well-ordering property of the natural numbers. Induction is a reasoning principle about this discrete structure, not a general rule for all real parameters.

## Prove the sum formula

Define $S(0)=0$ and $S(n)=1+\cdots+n$ for n≥1. The proposition is

$$
P(n):\quad S(n)=\frac{n(n+1)}2.
$$

Base: at n=0, both sides are zero. Step: assume the formula at an arbitrary integer k≥0. Then

$$
S(k+1)=S(k)+(k+1)
=\frac{k(k+1)}2+(k+1)
=\frac{(k+1)(k+2)}2.
$$

This is exactly P(k+1). The arbitrary step plus the base proves the identity over the entire nonnegative domain.

At the lab’s default maximum n=5, the checked sums are 0,1,3,6,10,15. Those six matching cases verify the finite implementation; the argument above supplies the infinite claim. These are complementary checks rather than interchangeable labels.

## A base case alone is not enough

Consider $2^n>n^2$ for n≥1. At n=1, 2>1 is true. At n=2, 4>4 is false. Thus the base does not guarantee the successor and the proposed universal claim fails.

The inequality does hold for every n≥5, but that is another statement with another base. Start with 32>25. If $2^k>k^2$ for k≥5, then $2^{k+1}>2k^2\ge(k+1)^2$, because $k^2-2k-1>0$ in this range. Repairing the start point must be explicit; one cannot silently alter the original theorem after seeing a counterexample.

A missing base can fail differently: P(n) meaning n=n+1 has a conditional successor argument derived algebraically from P(k), but P(0) is false. A conditional step without an established start proves no actual case.

## A late counterexample defeats a long pattern

The expression n²+n+41 is prime for n=0 through 39. At n=40,

$$
40^2+40+41=1681=41^2.
$$

One composite value refutes “prime for every n≥0.” The lab checks divisibility by trial factors and displays the smallest divisor. For a prime value, that divisor is the value itself; the accompanying boolean records primality.

A long run of successes can motivate a conjecture, but it cannot supply the missing inductive implication. The polynomial example is not an induction proof merely because the values are generated sequentially. The decisive question is whether P(k) logically guarantees P(k+1), not whether both happened to pass in a finite table.

## Strong induction and recursive reasoning

Strong induction assumes all cases from the start through k to prove k+1. It is equivalent in strength to ordinary induction on the natural numbers: one can define a stronger proposition that packages every earlier case.

For example, every integer n≥2 has a prime factorization. Base n=2 is prime. For the step, if n is prime it already is a one-factor product. Otherwise n=ab with 2≤a,b<n. The strong hypothesis supplies factorizations of both smaller factors, whose product factors n. This establishes existence, not the separate uniqueness theorem.

Recursive algorithms need an analogous decrease or well-founded structure. A function calling itself on an equal-sized input is not justified by induction just because it is recursive. Structural induction on expressions or trees follows their construction rules rather than a simple time index.

## Read the finite checker honestly

Each cell labels n and the truth of that particular case. The table provides actual left and right sides, or polynomial value and divisor. Sum and prime checks begin at zero; the power claim begins at one. If the selected upper limit leaves no applicable cases, the empty check is not evidence that the universal claim was proved.

Only the true sum example receives a separate general-proof panel. Moving the search limit changes the finite evidence, not the validity of that already established argument. The false examples continue to carry their stated domains instead of being automatically “repaired.”

## AI and research practice

Induction appears in invariants of iterative algorithms, correctness of recursive programs and reasoning about symbolic structures. A training loss decreasing for the first hundred steps does not prove monotonic decrease forever. To use induction, specify the invariant and prove that every permitted update preserves it under all stated assumptions.

For a learned reasoning system, separate successfully generated proof text from a checked derivation. Testing many tasks demonstrates a finite performance claim. A formal theorem about every allowed expression needs rules, a verifier and coverage of all construction cases.

## Exercises and reference

1. Prove $\sum_{i=1}^n(2i-1)=n^2$ by induction, including n=0.
2. Explain why proving only P(5) does not prove the sum identity for every n.
3. Find the first failing power case. **Answer:** n=2.
4. In the prime example at n=40, identify a factor. **Answer:** 41.
5. State a loop invariant for an accumulating sum and show initialization, preservation and the relation to the final answer.

For induction and recursive arguments, see the [MIT Mathematics for Computer Science readings](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/). The lab uses exact small-integer calculations; it is not a general automatic theorem prover.
