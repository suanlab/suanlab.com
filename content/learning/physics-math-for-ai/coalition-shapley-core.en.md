## Cooperation asks two different questions

Three laboratories can combine datasets to build a useful benchmark. One question is how much value each possible group can create. Another is how the final value should be divided. Even after agreeing on every group's worth, an average-contribution rule and an allocation immune to group departures can recommend different answers.

This unit studies a supplied transferable-utility game. The players are N={0,1,2}. For each subset S, v(S) is the total payoff available to that coalition, and v(empty)=0. A transferable common unit lets coalition members redistribute their total worth among themselves. This is a modeling assumption, not a claim that arbitrary scientific, financial or personal utilities can always be exchanged.

A characteristic function also abstracts away the detailed bargaining process. It assumes that the worth assigned to S is meaningful without specifying strategic reactions of outside players. If outsiders change what S can achieve, a richer model may be necessary. The lab supplies v(S); it does not infer coalition capabilities from observational data.

## Read coalitions, allocations and efficiency separately

A coalition is a subset of players, while an allocation x=(x0,x1,x2) is a vector of payoffs. Its efficiency condition is

$$
x_0+x_1+x_2=v(N).
$$

Efficiency only distributes the grand coalition's total. It does not by itself make the distribution individually acceptable. Individual rationality requires x_i≥v({i}). An efficient allocation satisfying those individual lower bounds is an imputation.

The gray triangle in the figure shows efficient allocations with nonnegative coordinates. When singleton values are positive, that triangle is larger than the imputation set. The diagram deliberately does not label every gray point individually rational.

The controls set x0=v(N)p0/12 and x1=v(N)p1/12, then calculate x2 from efficiency. If p0+p1 exceeds12, x2 is negative. That query remains visible in the surrounding square; it is not silently projected onto the triangle. A constraint failure should be shown rather than repaired behind the reader's back.

## A marginal contribution depends on who arrived earlier

For a joining order, start with the empty coalition. When player i joins a preceding set S, its marginal contribution is

$$
\Delta_i(S)=v(S\cup\{i\})-v(S).
$$

The three marginal contributions telescope: adding the changes along an order gives v(N)−v(empty)=v(N). Different joining orders can assign those changes to different players. A redundant resource may contribute much when arriving before its substitute and nothing when arriving afterward.

These are hypothetical accounting orders, not measured arrival times or an asserted bargaining protocol. The table lists all six permutations of three players. Its columns always identify the same players, so a player arriving first in one row is still compared with that player in every other row.

## Shapley averages every joining order

The Shapley allocation gives equal weight to all orders:

$$
\phi_i=\frac1{6}\sum_{\sigma}\Delta_i(S_{\sigma,i}).
$$

Equivalently, for three players,

$$
\phi_i=\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(2-|S|)!}{3!}
\bigl[v(S\cup\{i\})-v(S)\bigr].
$$

The coefficient counts how many orders place exactly S before i. Averaging the telescoping identities proves efficiency. Players with identical marginal roles receive equal values, and a null player whose contribution is always zero receives zero. Because marginal differences are linear in v, the allocation is additive across games. These properties motivate the rule; they do not turn it into a universal definition of fairness.

[Serrano's cooperative-game survey, §4](https://economics.brown.edu/sites/g/files/dprerj726/files/papers/2007-11_paper.pdf) presents the marginal formula and its axiomatic interpretation. Here every order is enumerated exactly. The curve's intermediate averages depend on the fixed lexicographic enumeration; only the sixth value averages the whole set. It is not a Monte Carlo convergence experiment.

## Coalitional stability defines the core

A coalition can block an allocation if its members collectively receive less than their own worth. With unrestricted transfers inside S, a positive surplus can be shared to make every member better off. Define coalition slack as

$$
\operatorname{slack}_x(S)=\sum_{i\in S}x_i-v(S).
$$

Negative slack means a blocking opportunity. Equality is allowed: a coalition receiving exactly its worth lacks a strictly positive surplus to divide. The core consists of efficient allocations satisfying every coalition constraint:

$$
\operatorname{Core}(v)=\left\{x:
\sum_{i\in N}x_i=v(N),\quad
\sum_{i\in S}x_i\ge v(S)\text{ for all }S\subseteq N
\right\}.
$$

Singleton constraints include individual rationality. Empty-coalition slack is genuinely0. Grand-coalition slack is also0 for an efficient allocation. Neither value should be treated as missing evidence. [Serrano, §3](https://economics.brown.edu/sites/g/files/dprerj726/files/papers/2007-11_paper.pdf) develops this coalition-blocking interpretation. The core is a set; Shapley is one allocation rule. One need not equal the other.

## Worked example: additive independent resources

Let each player contribute an independent amount1,2 or3. Then v(S) is the sum of the included amounts, and v(N)=6. Every order gives marginal contributions(1,2,3), so Shapley is(1,2,3).

Individual rationality requires x0≥1, x1≥2 and x2≥3. Since the three lower bounds already sum to6, efficiency forces equality in all three. The entire core is therefore the single point(1,2,3). It is not merely one selected point from a larger hidden region.

The first guide uses p0=2 and p1=4, producing x0=1, x1=2 and x2=3. Every coalition receives exactly its additive worth and all slacks vanish. Moving either allocation control can preserve efficiency while making another player unwilling to participate.

## Worked example: Shapley outside a nonempty core

Give player0 one left glove and players1 and2 one right glove each. A coalition earns6 if it contains player0 and at least one right-glove player; otherwise it earns0. Thus v({0,1})=v({0,2})=v(N)=6, while v({1,2})=0.

Player0 is pivotal in four of the six orders. Each right-glove player is pivotal in one. Shapley is therefore(4,1,1). This allocation is efficient and individually rational, yet coalition{0,1} receives5 instead of its worth6. Its slack is−1; coalition{0,2} has the same shortfall.

To find the core, combine x0+x1≥6 with total6 to obtain x2≤0. Similarly x0+x2≥6 gives x1≤0. Singleton constraints require both to be nonnegative, so x1=x2=0 and x0=6. The core is the point(6,0,0), while Shapley lies outside it.

Average contribution and stability answer different questions even when stable allocations exist. It would be incorrect to explain the difference only by saying that this game's core is empty.

## Worked example: a core that is actually empty

In the majority game, every coalition of at least two players has worth6 and each singleton has worth0. Symmetry and the marginal calculation give Shapley(2,2,2). But each pair gets4 and can create6 on its own.

No other efficient allocation can fix all pairs simultaneously. The three necessary constraints are

$$
x_0+x_1\ge6,\quad x_0+x_2\ge6,\quad x_1+x_2\ge6.
$$

Adding them requires2(x0+x1+x2)≥18. Efficiency instead gives the left side12. The contradiction proves that the core is empty. This is a proof of infeasibility, not a failure to find a point on the displayed query grid. Shapley remains well defined because all coalition values and marginal orders still exist.

## Regions and segments should not be replaced with one answer

For grand unanimity, only all three together earn6; proper coalitions earn0. The core is the full nonnegative simplex x0+x1+x2=6. Its three vertices are(6,0,0),(0,6,0),(0,0,6), and Shapley(2,2,2) is one interior allocation.

For pair01 unanimity, worth is6 exactly when both players0 and1 are included. The core is the segment x2=0, x0+x1=6 with x0,x1≥0. Player2 is null, and Shapley is(3,3,0). An isolated marker would hide the continuum of stable allocations.

These five families cover a point, a region, an empty set, a nonempty core excluding Shapley, and a segment. Multiplying all worths by a positive scale multiplies allocations and slacks without changing normalized geometry or membership decisions.

## Why the geometry is exact here

Eliminate x2 using efficiency. Each of the six proper-coalition constraints becomes a half-plane in(x0,x1). Singleton lower bounds and the x2 lower bound make the feasible set bounded. If nonempty, its vertices occur at feasible intersections of constraint boundaries, including the endpoints of lower-dimensional sets.

In three players the boundary normals are coordinate directions or the common diagonal direction. Every nonzero two-boundary determinant has magnitude1, so integer worths give integer vertex coordinates. The implementation uses common denominator12 to also represent the query grid and six-order Shapley averages. All membership and slack comparisons use integer numerators. Decimal conversion and angle sorting are used only for drawing an already determined vertex set.

This argument is specific to the small model and bounded exact inputs. A sparse query grid would not prove a general core empty, and exact enumeration here is not a scalability claim for arbitrary numbers of players.

## Connections to AI require a value-function choice

A dataset owner, feature or model component can be treated as a player only after defining what coalition worth means. Predictive accuracy, utility under a missing-feature convention and causal intervention value are different functions. Averaging contributions does not determine which one is scientifically appropriate.

A coalition score based on conditional observations can depend on correlations and on how absent features are represented. An attribution is therefore not automatically a causal effect or a transferable payment entitlement. This lab exposes the supplied v(S) precisely so that the allocation rule is not mistaken for a method that identifies the underlying capabilities.

The next unit studies shared rewards in a Markov team. There the question is which actions improve an evolving system, not how a characteristic-function total should be divided. Both topics involve cooperation, but their primitives and guarantees differ.

## Research extension: balanced collections certify the core

Give nonempty coalitions nonnegative weights $\lambda_S$ so each player is covered with total weight one: $\sum_{S\ni i}\lambda_S=1$. For any core allocation, summing weighted coalition constraints gives

$$\sum_S\lambda_Sv(S)\le\sum_i x_i=v(N).
$$

Balancedness requires this for every such collection. It is also sufficient for a finite transferable-utility game. Minimize $\sum_i x_i$ subject to all nonempty coalition lower bounds, including N, with x free in sign. Its LP dual maximizes the weighted worth under those balance equalities. The primal is feasible by choosing sufficiently large coordinates and bounded below by the grand-coalition constraint. LP duality gives optimum v(N) exactly when all balanced claims are at most v(N), yielding a core allocation.

In majority, each pair is worth $6s$. Put weight1/2 on every pair. Each player is covered once, but the weighted claim is $9s>6s$. If proper-coalition constraints are relaxed by a common nonnegative $\epsilon$, their sum requires $\epsilon\ge2s$. Equal allocation $(2s,2s,2s)$ attains it.

| Scale s | Grand worth | Balanced pair claim | Minimum nonnegative relaxation |
|---|---:|---:|---:|
| 1 | 6 | 9 | 2 |
| 2 | 12 | 18 | 4 |
| 3 | 18 | 27 | 6 |

```python
from fractions import Fraction as F
pairs=[3,5,6]; balanced={S:F(1,2) for S in pairs}
assert all(sum(w for S,w in balanced.items() if S&(1<<i))==1 for i in range(3))
for scale in [1,2,3]:
    worth=[6*scale if S.bit_count()>=2 else 0 for S in range(8)]
    claim=sum(w*worth[S] for S,w in balanced.items()); total=worth[7]
    assert claim==9*scale and claim>total
    epsilon=2*scale; allocation=[2*scale]*3
    assert sum(allocation)==total
    for S in range(1,7):
        assert sum(x for i,x in enumerate(allocation) if S&(1<<i))>=worth[S]-epsilon
    assert F(3*6*scale-2*total,3)==epsilon
    print(scale,total,claim,epsilon)
```

This extends the empty-core diagnosis to quantified approximate stability. We restrict epsilon to be nonnegative; some least-core conventions allow negative values. The native panel still shows the exact core, not a relaxed substitute.

## Exercises and explained solutions

Exercise1: in the glove game, calculate the slack of coalition{0,2} under Shapley and under the core allocation.

Solution: Shapley gives4+1−6=−1; the core allocation gives6+0−6=0. A zero slack is a binding stability constraint, while a negative slack supplies a blocking surplus.

Exercise2: does a negative x2 query violate efficiency? Can a gray-triangle point fail individual rationality?

Solution: x2 is calculated from efficiency, so the total remains v(N). In these families its negative value violates a singleton constraint. Conversely, in the additive game a nonnegative allocation such as(0,3,3) is efficient but gives player0 less than its standalone worth1.

Exercise3: why does the majority contradiction prove more than inspecting169 query points, and why is the unanimity core not a single solution?

Solution: adding the necessary pair inequalities excludes every real efficient allocation, including points between grid values. Grand unanimity imposes only nonnegative coordinates and total6, leaving the entire triangle. The proof and the set description establish the claims independently of plotting resolution.
