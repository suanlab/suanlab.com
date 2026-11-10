## What is impossible if a finite table is computable?

All 278 commands from the previous lab halt. Enumerating them gives an exact finite-language minimum. Uncomputability instead says that no algorithm always halts and returns exact K_U(x) for every finite string under a fixed **universal** prefix machine U.

Knowing a short description for one object is different from computing every minimum. Individual upper bounds, and some exact values, can still be proved.

## Count short descriptions

Kraft's inequality bounds the number of prefix programs of length at most L by 2ᴸ. Distinct outputs cannot outnumber programs, so not every length-(L+1) string can have such a short description.

$$
\#\{x:K_U(x)\le L\}\le 2^L.
$$

The finite lab covers two outputs at L=3, four at L=5, sixteen at L=8 and all 256 at L=9. Multiple programs may have the same output: there are 22 programs within length eight but only sixteen distinct outputs.

## A diagonal contradiction

Suppose a terminating algorithm A computes K_U exactly. Given n, enumerate strings of length n+1 in lexical order. Use A to select the first xₙ with K_U(xₙ)>n. Counting guarantees existence, so this search halts.

Yet the search procedure and A are fixed. A program producing xₙ needs only a self-delimiting description of n, costing O(log n), plus a constant. For sufficiently large n:

$$
\begin{aligned}
K_U(x_n)&>n,\\
K_U(x_n)&\le O(\log n)+c<n.
\end{aligned}
$$

This contradiction rules out A. The finite coverage plot illustrates counting and duplicate outputs; it does not prove this universal theorem.

## Why more runtime does not settle the question

Running programs sequentially can become stuck on one that never halts. Dovetailing gives candidates alternating simulation steps, eventually finding each terminating program. Discovered descriptions provide improving upper bounds, but generally cannot certify that every shorter unfinished candidate will never halt later.

An AI-generated compressed description can be executed and checked against its target. A search failure or timeout does not establish intrinsic randomness or universal optimality.

## Exercises with explanations

1. Is byte 173 covered at L=8? **No in this finite language.** Its nine-bit literal is shortest here, which is not a universal lower bound.
2. Does uncomputability rule out useful compressors? **No.** It rules out exact optimization for every input, not useful discoveries.
3. Why establish termination in the contradiction? **The alleged short description must be an actual program producing xₙ.** Totality of A and counting ensure that.

Further reading: [Vitányi, How Incomputable Is Kolmogorov Complexity?](https://homepages.cwi.nl/~paulv/papers/incomp.pdf).
