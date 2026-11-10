## Finding and checking are different tasks

A language model may propose a proof, a search procedure may fail to find one, and a checker may reject a supplied term. These are different events. This lab uses annotated simply typed lambda terms to make the distinctions visible. It covers atoms P,Q,R and function types, with no dependent types, recursion, arithmetic axioms or implicit classical principles.

A judgement records assumptions, a term and its type:

$$
\Gamma\vdash t:A.
$$

Read it as “under the assumptions in Gamma, term t has type A.” Under the propositions-as-types interpretation, t is evidence for proposition A. An arrow A→B transforms evidence for A into evidence for B. Assumptions remain part of the claim: a proof under f:P→Q and g:Q→R is not an unconditional construction of those functions.

## Three local rules

A variable obtains its type from the context. A lambda introduces an assumed argument, checks its body and constructs a function type. Application requires a function and an argument with exactly the function's domain type:

$$
\frac{x:A\in\Gamma}{\Gamma\vdash x:A}
\qquad
\frac{\Gamma,x:A\vdash t:B}{\Gamma\vdash\lambda x:A.t:A\to B}
\qquad
\frac{\Gamma\vdash f:A\to B\quad\Gamma\vdash u:A}
{\Gamma\vdash f\,u:B}.
$$

Parameter annotations make inference syntax directed. The checker follows the term's structure rather than guessing a theorem. It can infer a type, reject an unbound variable, reject application of an atomic value or detect an argument mismatch. If a premise fails, enclosing nodes fail too. An argument need not be checked after the function is already known not to be a function; gray tree nodes mark such unvisited syntax rather than inventing a diagnostic for it.

The tree displays the supplied candidate, not a fabricated successful search tree. Each checked node carries its inferred type or failure. The selected step's local rule is shown separately. Long node labels are shortened; the table retains full types.

## Binding without accidental name capture

The model stores variables as de Bruijn indices. Index0 is the nearest binding. A lambda puts its annotated parameter at the front of the context, shifting older bindings by one position. Human-readable names are generated only for display.

For context [f:P→Q,g:Q→R], entering lambda x:P creates [x:P,f:P→Q,g:Q→R]. Thus g is index2, f is index1 and x is index0. Confusing the shifted indices changes the term or makes it ill typed. Reusing a displayed variable name is not a licence to ignore its binder.

## What the bounded search actually enumerates

Search tries increasing exact counts of term nodes. A variable, a lambda and an application each cost one; type annotation size is not included. The grammar is beta-normal: a lambda with a normal body, or a context variable applied to normal arguments. It excludes a lambda immediately used as the function of an application, which would form a beta-redex.

For an arrow goal, search can introduce a lambda. It also considers context-variable heads whose successive arrow results reach the goal, distributing the remaining node budget among their arguments. This covers the stated normal grammar. The smallest found count is therefore a minimum within that grammar and the tested limit. A normal-form size bound is not a bound over all arbitrary proof syntax: normalization can change size.

A miss means **no normal term was found within this budget**. It does not mean false, UNSAT, no proof or uninhabited. Every found term is sent to the separate checker and must infer the requested type. The supplied candidate is checked independently of the search limit, so it may be larger than that limit and still be a valid proof.

## Catalogs used by the controls

Arrows associate to the right: P→Q→P means P→(Q→P).

| Goal index | Type |
| --- | --- |
| 0,1,2 | P,Q,R respectively |
| 3 | P→P |
| 4 | P→Q |
| 5 | P→R |
| 6 | (P→Q)→P→Q |
| 7 | P→Q→P |
| 8 | (P→Q)→(Q→R)→P→R |
| 9 | ((P→Q)→P)→P |

Context0 is empty; context1 is [f:P→Q,g:Q→R]; context2 is [p:P,f:P→Q]; context3 is [p:P]; context4 is [f:P→Q]. Lists are nearest first. The initial context is not silently extended with facts.

Candidate0 is identity lambda x:P.x. Candidate1 is open composition lambda x:P.g(f x), expecting context1. Candidate2 applies variable0 to itself. Candidate3 is variable0; candidate4 applies variable1 to variable0. Candidate5 is lambda x:P.lambda y:Q.x. Candidate6 is closed composition lambda f:P→Q.lambda g:Q→R.lambda x:P.g(f x). Candidate7 is identity on functions lambda f:P→Q.f. Changing context can change which candidate is meaningful.

## Worked experiment: a search miss with an actual proof

Choose goal5, context1, budget5 and candidate1. The candidate is

$$
\lambda x:P.\ g\,(f\,x):P\to R.
$$

Its three variable nodes, two applications and one lambda total six. First infer g:Q→R, f:P→Q and x:P. The inner application gives Q, the outer one gives R, and lambda introduction discharges x to produce P→R. The supplied candidate checks and proves the goal even though budget5 search misses. Its assumptions f,g remain in Gamma.

Raise only the budget to six. Search finds a size6 normal term and the checker confirms it. The supplied candidate's inference does not change. In the plot, blue records search success at each independently evaluated budget; orange records whether the fixed supplied candidate proves the root goal. These indicators are neither probabilities nor elapsed time. The sweep previews nine budgets; the displayed search result uses the selected one.

## Worked experiment: well typed but wrong goal

Choose empty context0, goal4 and candidate0. Identity infers P→P, so its local typing rules all pass. The requested theorem is P→Q, which is different. A blue candidate tree is therefore not itself evidence that the requested root goal was proved.

Boolean assignments P=true,Q=false make P→Q false. With bits P=1,Q=2,R=4, the two countermodel masks are1 and5. This semantic refutation is independent of the search miss. Conversely, finding no Boolean countermodel does not construct a proof term. The lab's classical truth table is a necessary semantic check, not a complete constructive proof-search oracle. Goal9 illustrates a classically valid formula for which this bounded search returns no term; the miss alone makes no nonexistence claim.

## Exercise and solution

Exercise: use context1 and candidate2, which applies f to itself. Which premise fails?

Solution: both occurrences of f have type P→Q. Function application requires an argument of type P, so the root reports an argument mismatch. The variables are individually typed; the application is not. No inferred root type exists.

Now use context3=[p:P] and the same candidate. The function position has atomic type P, so application is rejected before checking its argument. There are three syntax nodes but only two checker steps; the remaining argument is gray and unchecked. This is not a claim that the unchecked argument is invalid.

As a final binding exercise, candidate5 has closed type P→Q→P. After two lambdas the desired outer argument is index1, not index0. Switching the index would instead return Q and change the inferred theorem.

## Trust boundaries and research use

A search heuristic can change without changing the checking rules. A generated proof is useful evidence only when its assumptions, elaborated term and requested statement are checked in the intended system. This lab checks only its displayed simply typed fragment; it is not Lean's kernel and does not validate arbitrary dependent-type proofs. Its exact-subgoal counter counts memoized search states, not time or all possible derivations.

For syntax-directed inference and the annotated variable/lambda/application rules, see [Programming Language Foundations in Agda: Inference](https://plfa.github.io/Inference/). For propositions as types and implication as functions, see [Lean's Propositions and Proofs](https://lean-lang.org/theorem_proving_in_lean4/Propositions-and-Proofs/). The bounded normal search, contexts, candidate catalog and diagnostic comparisons here are constructed for this textbook.
