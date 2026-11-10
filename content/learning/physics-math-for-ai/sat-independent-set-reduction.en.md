## Scope of a finite decision problem

A decision problem is a language of yes-instance encodings, not one particular answer. This experiment uses three variables and three clauses. It can exhaustively inspect eight assignments and every selection of nine occurrence vertices. Such a fixed finite check differs from a general procedure deciding termination of every program.

Assignment bit i gives variable x_{i+1}. Vertex bits select literal occurrences in clause order. Repeated equal literals still have distinct occurrence vertices. Orange nodes show the query selection; being highlighted does not automatically make them a valid witness.

## Construct a reduction with two directions

Connect the three occurrences in each clause into a clique. Also connect contradictory x and not-x occurrences. For m=3 clauses,

$$
\varphi\text{ satisfiable}\iff\alpha(G_\varphi)\ge m.
$$

A satisfying assignment selects one true literal per clause. The selected vertices share neither a clause nor contradictory truth values, so they form an independent set. Conversely, a size-three independent set must select exactly one vertex per clause clique. No contradictory selected literals occur, so their variable requirements are consistent. Extend unselected variables arbitrarily, and every clause remains satisfied.

For a general formula, there are 3m vertices and at most O(m²) edges, giving a polynomial-size construction. Distinguish finite witness checks from this general size argument. A reduction A≤pB means that an efficient solver for B can solve A through the transformation; its direction cannot simply be reversed.

## A separate diagonal halting argument

Encode programs and inputs as strings. Suppose a total computable H always correctly decides whether P(x) terminates. Construct

$$
D(P)=\begin{cases}\text{loop forever},&H(P,P)=\text{halts},\\
\text{halt},&H(P,P)=\text{does not halt}.\end{cases}
$$

Apply D to its own encoding. If H predicts termination, D loops; if H predicts nontermination, D halts. Either answer contradicts correctness, so no such total H exists. This general contradiction is not inferred from a finite SAT table or a timeout. Finite-variable exhaustive SAT decisions and general halting undecidability are different statements.

## Worked example and exercise

For the first formula, assigning every variable true gives forward witness mask 81 by selecting the first true occurrence in each clause. Query mask 273 selects another valid independent set and decodes to assignment three. It need not reproduce original assignment seven: different valid witnesses can satisfy the same formula.

Exercise: can clauses (x OR x OR x), (not-x OR not-x OR not-x), and (y OR y OR y) all hold? Solution: the first two contradict each other. The corresponding maximum independent set has size two. Any attempted selection of one occurrence from each clause encounters a contradiction edge between the first two groups.

## Verifiable AI reasoning

When an LLM proposes a SAT answer or a program property, separate checking a finite witness from claiming a general decision algorithm. A satisfying assignment or independent set can be directly verified. A timeout is not an unsatisfiability certificate. Fast certificate checking and fast certificate discovery are also different properties. This experiment executes only its declared finite formulas, not arbitrary submitted programs.

Reference: [MIT computability and reductions course](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/).
