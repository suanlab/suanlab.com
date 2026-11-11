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

## Recover a witness using a decision oracle

Suppose an oracle decides satisfiability for arbitrary input formulas. After an initial yes answer, fix one actual variable at a time. Ask whether the current formula remains satisfiable with that variable false. If yes, retain false; otherwise retain true. The invariant is that at least one satisfying completion remains. After all n variables have been fixed, the assignment itself is a witness. This uses at most n+1 decision queries, including the initial query.

Variable names can first be renumbered by their distinct occurrences, so n means the number of variables, not the largest binary identifier. A fixed value can be encoded by a unit clause, or by repeating its literal three times to stay in 3-CNF. The formula growth and the original reduction remain polynomial in encoded input length. If the decision oracle were polynomial time, this witness search would be polynomial too; the construction does not supply such an oracle.

| Input | Decision queries in this procedure |
|---|---:|
| Three-variable satisfiable example | 4 |
| Unsatisfiable example | 1 |
| General satisfiable n-variable input | At most n+1 |

```python
from itertools import product
def recover(clauses,n):
    calls=0
    def oracle(partial):
        nonlocal calls
        calls+=1
        return any(all(bits[i-1]==b for i,b in partial.items()) and all(any(bits[abs(l)-1]==(l>0) for l in c) for c in clauses) for bits in product((False,True),repeat=n))
    partial={}
    if not oracle(partial):return None,calls
    for variable in range(1,n+1):
        partial[variable]=False
        if not oracle(partial):partial[variable]=True
    assert all(any(partial[abs(l)]==(l>0) for l in c) for c in clauses)
    return partial,calls
sat=[[1,2,3],[-1,2,3],[1,-2,-3]]
unsat=[[1,1,1],[-1,-1,-1],[2,2,2]]
assert recover(sat,3)[1]==4 and recover(unsat,3)==(None,1)
print(recover(sat,3),recover(unsat,3))
```

The small oracle above exhausts assignments, so the executable demonstration is not a polynomial-time SAT algorithm. Finite witness checking, oracle search-to-decision reduction and halting undecidability are separate claims. A halting execution can be recognized by running it, but an arbitrary timeout cannot certify nontermination or unsatisfiability.

## Worked example and exercise

For the first formula, assigning every variable true gives forward witness mask 81 by selecting the first true occurrence in each clause. Query mask 273 selects another valid independent set and decodes to assignment three. It need not reproduce original assignment seven: different valid witnesses can satisfy the same formula.

Exercise: can clauses (x OR x OR x), (not-x OR not-x OR not-x), and (y OR y OR y) all hold? Solution: the first two contradict each other. The corresponding maximum independent set has size two. Any attempted selection of one occurrence from each clause encounters a contradiction edge between the first two groups.

## Verifiable AI reasoning

When an LLM proposes a SAT answer or a program property, separate checking a finite witness from claiming a general decision algorithm. A satisfying assignment or independent set can be directly verified. A timeout is not an unsatisfiability certificate. Fast certificate checking and fast certificate discovery are also different properties. This experiment executes only its declared finite formulas, not arbitrary submitted programs.

Reference: [MIT computability and reductions course](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/).
