## Why Boolean truth may be insufficient

A planner may say that three tasks satisfy their pairwise timing requirements. Those claims must also describe a possible schedule. SAT asks whether a propositional formula has a truth assignment. SMT adds interpretations from a background theory, such as integer arithmetic. A constraint satisfaction problem specifies variables, their domains and allowed combinations; Boolean SAT is one important special case. General CSPs and SMT problems need different algorithms depending on the domain and constraint language.

Here we use three unbounded integer variables x,y,z and three atoms:

$$
P:(x-y\le a),\qquad Q:(y-z\le b),\qquad R:(z-x\le c).
$$

Bounds a,b,c are integers between −2 and 2. The formula is three input clauses joined by AND. A clause is an OR of some of P,Q,R,¬P,¬Q,¬R. The eight Boolean assignments are finite; the integer domain is not restricted to eight or to a plotted window. The lab is an exact small instance of integer difference logic, not a general-purpose SMT solver.

## Read the two kinds of masks

A clause mask uses weights 1,2,4,8,16,32 for P,Q,R,¬P,¬Q,¬R. Thus mask9 is P∨¬P, a tautology; mask0 is an empty OR and is false. An assignment mask uses only weights 1,2,4: a set bit means that atom is true. Assignment7 makes all three atoms true. The clause matrix shows which literals are present and their truth under the selected assignment. AND the three row results to evaluate the CNF.

The blue plot marks Boolean models. Orange marks assignments that satisfy both the CNF and the integer theory. The SMT-feasible count is the number of Boolean assignments admitting an integer witness. It does not count all integer models, which can be infinite. Each vertical value is a discrete indicator, not a probability or training trajectory.

## Negation depends on the integer domain

For integers the complement of a non-strict inequality is

$$
\neg(u-v\le k)\iff u-v\ge k+1\iff v-u\le-k-1.
$$

For example, false P with a=0 requires y−x≤−1. Using y−x≤0 would wrongly allow equality, making P and its negation overlap. For real variables the strict inequality cannot be replaced with the same integer +1 rule. Changing the domain changes the theory checker.

## Turn inequalities into a graph

Encode v−u≤w by an edge u→v of weight w. If potentials d obey every edge, then setting variables to those potentials supplies an integer witness. Bellman–Ford starts every potential at zero, equivalently adding a zero-weight super-source. Relax edges three times for the three-variable graph. A change on the third pass certifies a reachable negative cycle; otherwise the resulting potentials satisfy all inequalities.

Why is a negative cycle impossible for a feasible assignment? Sum its inequalities: the variable terms telescope, giving

$$
0\le\sum_{e\in\mathrm{cycle}}w_e<0.
$$

The contradiction is a certificate, not a timeout. When feasible, subtract d_x from all potentials so that x=0. Differences are unchanged; this is translation normalization, not a finite search bound. A witness need not be unique.

## Three worked experiments

With a=0,b=0,c=−1 and clauses P,Q,R, the sole Boolean model is assignment7. Its inequalities sum to 0≤−1. No Boolean assignment admits an SMT witness. The selected cycle has weight −1, and its conflict yields ¬P∨¬Q∨¬R, mask56, under these fixed bounds.

Change c to zero. Now (0,0,0) satisfies every atom and all three clauses. One Boolean assignment passes each check. This does not mean there is only one integer model: every (t,t,t) with integer t is a solution. Cycle weight and UNSAT cores become unavailable, rather than displaying a fictional zero-weight conflict.

Finally use a=b=c=0 and clauses P,¬P,P∨¬P, selecting assignment1. The first two clauses contradict each other. Both global assignment counts are zero, but the selected theory has witness (0,2,1): P is true while Q and R are false. That witness fails the input clause ¬P and therefore is not a full-formula solution.

## Distinguish three explanations of failure

A negative cycle explains the selected conjunction of theory literals. A learned clause negates that conflicting conjunction; it is valid under the fixed theory interpretation. It does not declare all other assignments impossible. For example, with all three clauses tautological and bounds (0,0,−1), selection7 conflicts but six other Boolean assignments admit SMT witnesses.

A minimum input UNSAT core is instead a smallest subset of the three asserted clauses whose conjunction has no model in the relevant background. The lab enumerates all eight subsets, breaking equal-size ties by subset mask. Boolean and SMT cores can differ. The first experiment has no Boolean core but SMT core {C1,C2,C3}; the third has core {C1,C2} in both checks. The fixed atom meanings are background assumptions, outside this input-clause core. Industrial solver cores are not necessarily minimum cardinality.

## Exercise and solution

Exercise: choose bounds (−1,−1,−1), clauses ¬P,¬Q,¬R and assignment0. Does the integer negation create an inconsistency?

Solution: the constraints become y−x≤0, z−y≤0, x−z≤0. Their cycle sum is zero and (0,0,0) is a witness. Each original difference is zero, so all three original inequalities with bound −1 are false as required. A zero cycle is not a negative cycle. Now make C1 empty: global SAT and SMT both fail, although the selected theory still has its witness; the minimum input core is {C1}.

## Connection to AI verification and larger solvers

A generated schedule or symbolic plan can be checked by encoding its actual requirements. Feasibility means the encoded requirements admit a model; it does not establish that the encoding captures every real-world requirement. A theory witness, counterexample and core answer different diagnostic questions. This lab exhaustively enumerates eight assignments for transparency. Production solvers combine SAT search, propagation and theory reasoning; the lab does not simulate their complete CDCL/DPLL(T) algorithms or performance.

For reference, [SMT-LIB's logic definitions](https://smt-lib.org/logics.shtml) define the integer difference-logic fragment, and the [cvc5 theory tutorial](https://cvc5.github.io/tutorials/beginners/theories.html) explains interpreted theories and logic selection. The graph model, three experiments and exhaustive checks here are constructed for this textbook.

The witness/cycle equivalence is developed in [MIT 6.046J Lecture 11, pages 5–6](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/312f4a419009b58f8147b75975db4347_MIT6_046JS15_lec11.pdf).
