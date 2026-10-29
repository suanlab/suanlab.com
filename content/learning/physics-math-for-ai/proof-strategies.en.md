## A theorem is an obligation, not a pattern

An experiment can suggest that a claim is true. A proof explains why every case in its stated scope must satisfy it. A counterexample explains why a universal claim fails. These are different kinds of evidence and should not be merged into one “verified” badge.

We study a small theorem whose assumptions can be stated completely: for every integer n, if n² is even then n is even. The finite lab checks a bounded interval, while a separate argument proves the theorem for all integers. Keeping both visible makes their roles concrete.

## Definition, assumption and conclusion

A definition stipulates the meaning of a term. An integer is even if n=2k for some integer k; it is odd if n=2k+1 for some integer k. A theorem is a statement requiring justification from definitions and accepted results.

| Part | Reference theorem |
| --- | --- |
| Domain | n∈ℤ |
| Assumption | n² is even |
| Conclusion | n is even |
| Quantification | Every integer n satisfying the assumption |

The integer domain is essential. Over real numbers, defining “even” still refers to integer multiples of two, and a different statement can result. For instance n=√2 has square 2 but is not an even integer. A hidden domain change can defeat an otherwise familiar argument.

## Start with a direct proof in the easier direction

Prove “if n is even, then n² is even.” Assume n=2k with integer k. Then n²=4k²=2(2k²), and 2k² is an integer. This directly supplies the witness required by the definition of evenness.

This proves the converse direction of our reference theorem, not the reference direction itself. Calling it a proof of both directions would repeat the converse error from the implication unit.

A direct argument for the reference direction can use division with remainder: write n=2k+r, r∈{0,1}. Then n² has the same parity as r. The even-square assumption forces r=0. The key step is a justified exhaustive case split, not assuming n was even at the start.

## Prove the contrapositive

The contrapositive of “n² even implies n even” is “n odd implies n² odd.” For arbitrary integer k, let n=2k+1. Expand:

$$
n^2=(2k+1)^2=4k^2+4k+1=2(2k^2+2k)+1.
$$

The quantity 2k²+2k is an integer, so the square is odd by definition. Every odd integer has this form. By equivalence of a conditional and its contrapositive, the original theorem follows for all integers.

The calculation does not depend on a search bound, an example such as n=3, or the number of successful tests. Those examples can check arithmetic, but the arbitrary-integer argument carries the universal scope.

## Reorganize as a contradiction proof

Assume n² is even and the intended conclusion fails. For an integer, failure of evenness means oddness. Write n=2k+1; the same expansion proves n² is odd. An integer cannot be both even and odd, contradicting the assumption.

The contradiction must actually follow from the assumptions. Reaching an unexpected-looking formula or an inconvenient numerical value is not a contradiction by itself. State precisely which established facts cannot simultaneously hold.

The lab lets you switch between these two argument presentations. The theorem does not change with the presentation, and the finite table does not become an infinite proof when a different button is pressed.

## Stronger conclusions need new evidence

The second claim changes the conclusion to “n is divisible by four.” The assumption remains that n² is even. The integer n=−2 has square 4 but is not divisible by four, so it is a counterexample. A stronger conclusion cannot be borrowed from a proof that established only evenness.

The third claim says an odd square implies an even input. The integer −3 refutes it: its square is 9 and its input is odd. A single valid counterexample disproves a universal statement, even if many other sampled cases do not violate it.

A counterexample must satisfy the assumptions and violate the conclusion. An integer with an odd square does not refute the original even-square theorem, because it fails the premise. The implication table makes that distinction explicit.

## Identify circular reasoning and invalid cancellation

Assuming n=2k at the start of the reference proof would assume the conclusion. In a different common error, dividing an equation by an unknown expression requires proving that expression is nonzero. For example x²=x does not imply x=1 without handling x=0; dividing by x discards a legitimate solution.

Each proof step should cite a definition, assumption, algebraic identity or earlier result with its own conditions satisfied. A numerical solver returning one answer supplies none of these obligations automatically.

## Finite testing has a valuable, bounded role

The table checks all integers from −B to B. That is exhaustive for that finite domain. Increasing B can reveal a missed counterexample, but passing any finite interval alone does not prove a statement over all integers.

Tests are especially useful for discovering incorrect signs, missing boundary conditions and overstrong claims. Their logical strength depends on scope. A finite truth-table check can establish a propositional equivalence because the complete valuation space is finite; a finite integer search cannot exhaust ℤ.

## Apply this discipline to AI research

Separate a definition of robustness from a theorem claiming robustness and an experiment measuring attacks. State the perturbation set, model class, loss, quantifiers and assumptions. A successful attack supplies a counterexample to one asserted guarantee; failure to find an attack does not prove that no attack exists.

For a proposed optimization guarantee, inspect every use of smoothness, convexity, bounded noise or nonzero denominators. Numerical evidence can support relevance of a theorem but does not silently replace its proof. Conversely, a proved theorem may apply to a model that does not describe the experimental setting.

## Exercises and reference

1. Prove directly that the sum of two even integers is even.
2. Explain why n=3 is not a counterexample to “n² even implies n even.” **Answer:** its square does not satisfy the premise.
3. Find the missing case in the division x²=x by x. **Answer:** x=0.
4. Choose a research claim and write a counterexample specification: which assumptions must hold, and which conclusion must fail?

For proof methods, see the [MIT Mathematics for Computer Science readings](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/). The reference theorem and finite checker are presented here with explicit integer assumptions.
