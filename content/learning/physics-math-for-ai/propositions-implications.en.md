## Read the claim before calculating

A theorem about a learning algorithm often says: if assumptions hold, then a conclusion follows. Reading this as “the conclusion happened, so the assumptions hold” reverses the claim. Much confusion about convergence, generalization and scientific evidence begins with this small logical mistake.

A proposition is a statement with a truth value in a specified interpretation. “The matrix is invertible” becomes a proposition after the matrix is fixed. “$x>0$” is instead a predicate with a free variable until we supply a value or quantify over a domain. A question, command or vague promise need not be a proposition at all.

We use classical two-valued logic. This is a precise reasoning system, not a claim that uncertain real-world observations always arrive as unambiguous true/false measurements.

## Symbols and scope

| Symbol | Meaning |
| --- | --- |
| $P,Q$ | Propositions with fixed truth values |
| $\neg P$ | Negation |
| $P\land Q$ | Both statements hold |
| $P\lor Q$ | At least one holds, including both |
| $P\Rightarrow Q$ | Material implication |
| $P\Leftrightarrow Q$ | Both implications hold |

Parentheses matter. Negating $P\land Q$ differs from merely negating $P$. The words “or” in this unit are inclusive, so an extra condition is needed to express exactly one alternative.

## Construct implication from the forbidden case

The conditional $P\Rightarrow Q$ rules out one situation: $P$ holds but $Q$ fails. Its truth table is

| P | Q | P⇒Q | Q⇒P |
| --- | --- | --- | --- |
| False | False | True | True |
| False | True | True | False |
| True | False | False | True |
| True | True | True | True |

Equivalently,

$$
P\Rightarrow Q\equiv\neg P\lor Q.
$$

At the reference lab setting $P=1,Q=0$, the original implication is false. At $P=0,Q=1$, it is true while its converse is false. Agreeing at the both-true row does not make two formulas logically equivalent; equivalence requires agreement at every assignment.

## Necessary and sufficient conditions

When $P\Rightarrow Q$, $P$ is a sufficient condition for $Q$, and $Q$ is a necessary condition for $P$. For example, “$n$ is divisible by four” is sufficient for “$n$ is even.” Evenness is necessary for divisibility by four, but not sufficient: $n=2$ is a counterexample to the reverse implication.

The word “necessary” attaches to what must follow whenever the premise holds. It does not mean the premise is a cause of the conclusion. Two conditions are necessary and sufficient for each other only when both implications are established.

For a square real matrix, an inverse implies nonzero determinant, and nonzero determinant implies an inverse. That is a genuine equivalence within stated dimensions and field. Replacing “square matrix” with an arbitrary rectangular map changes the statement and its definitions.

## The contrapositive preserves the claim

The contrapositive is $\neg Q\Rightarrow\neg P$. Expand its implication:

$$
\neg Q\Rightarrow\neg P\equiv Q\lor\neg P\equiv\neg P\lor Q.
$$

It therefore agrees with the original implication. The converse $Q\Rightarrow P$ and inverse $\neg P\Rightarrow\neg Q$ are equivalent to each other, but generally not to the original statement.

This matters when choosing a proof strategy. Proving a contrapositive proves the intended conditional. Proving its converse may prove a different theorem while leaving the original question unresolved.

## Vacuous truth does not provide evidence of a case

If $P$ is false, $P\Rightarrow Q$ is true under this definition. The implication has not been violated. This does not establish that a premise-satisfying case exists or that $Q$ is independently true.

For example, “every selected model that has property P also has property Q” can be true when no selected model has P. Before interpreting a universal evaluation claim, ask whether its premise was ever satisfied. A logical statement about all qualifying cases can be valid while supplying no successful example.

Material implication also does not encode time, intervention or causation. Observing $P$ before $Q$ does not turn a truth table into a causal experiment. Causal claims need their own assumptions and evidence.

## Valid inference and common mistakes

Modus ponens uses $P\Rightarrow Q$ together with $P$ to infer $Q$. Modus tollens uses the same implication and $\neg Q$ to infer $\neg P$. Both follow by ruling out truth-table rows.

Affirming the consequent tries to infer $P$ from $Q$ and $P\Rightarrow Q$. The row $P$ false, $Q$ true refutes that rule. Denying the antecedent tries to infer $\neg Q$ from $\neg P$; the same row refutes it.

In AI, a sufficient convergence theorem may have assumptions involving smoothness, a step-size range and a noise model. Observing convergence does not prove every assumption was met. Conversely, failing one sufficient condition does not prove convergence is impossible: another theorem or mechanism may apply.

## Read the interactive model

The table enumerates all four assignments, not random samples. In the implication grid, row and column labels 0/1 represent P and Q. T/F letters carry truth information in addition to color. Changing the selected P and Q changes the readout; it does not change the underlying logical definition.

The separate converse and contrapositive columns let you inspect the exact rows where claims differ. A colored cell is a finite logical valuation, not a calibrated probability. The next unit adds predicates and quantifiers so that statements can range over objects.

## Exercises and research practice

1. Let P mean “a dataset has no missing values” and Q mean “a training script runs.” Explain why neither implication is automatic.
2. Write the contrapositive of “if a map is injective, two distinct inputs cannot share an output.”
3. Give one assignment refuting the converse rule. **Answer:** P false, Q true.
4. Find a theorem in a paper. Separate every assumption from its conclusion, state whether the condition is sufficient or necessary, and avoid replacing a sufficient theorem with an equivalence.

For additional logic and proof material, see the [MIT Mathematics for Computer Science textbook](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/resources/mit6_042js15_textbook_pdf/). The lab’s truth table and examples are direct constructions for this book.
