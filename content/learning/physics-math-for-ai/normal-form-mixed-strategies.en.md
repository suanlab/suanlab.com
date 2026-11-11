## A game starts with players, actions and preferences

Consider two decision makers. Player1 chooses a row R0 or R1, while Player2 chooses a column C0 or C1. Each acts once without observing the other's current action. A normal-form game specifies their available strategies and their utilities at every joint choice. In this one-decision setting, a pure strategy is just one available action. In a sequential game, a strategy instead has to specify a complete contingent plan, including decisions at information sets that may never be reached.

Each matrix cell contains an ordered pair (u1,u2). The first value belongs to Player1 and the second to Player2. A large first number is not automatically good for Player2. The lab provides coordination, opposed-payoff and social-dilemma examples with fixed, explicitly displayed matrices. It is a model of simultaneous decisions, without private types, signals or a repeated-play process.

The normal-form and mixed-strategy definitions follow the distinctions in [MIT's Representation of Games notes, sections3.3 and3.3.2](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2025/mit14_12f12_chapter3_f12.pdf). Our finite matrices, integer-probability implementation and worked comparisons are constructed for this lab.

## A mixed strategy is a distribution, not a halfway action

Let p be the probability that Player1 chooses R0, and q the probability that Player2 chooses C0. Their mixed strategies are vectors:

$$
x=(p,1-p),\qquad y=(q,1-q),\qquad p,q\in[0,1].
$$

A probability0.5 does not create a new action halfway between R0 and R1. In any actual play the player takes one action. The distribution describes randomization, a prediction or an appropriate population interpretation, depending on the model. An observed frequency alone does not identify which interpretation generated it.

A pure strategy is included at the boundary: p=1 always selects R0, and p=0 always selects R1. The controls use integer percentages, so entering50 means p=0.5. This discrete control grid helps us keep exact probability numerators; the mathematical definition is not restricted to percentages.

## Independence supplies the joint distribution

With independent private randomization, joint probabilities multiply:

$$
P=xy^T=
\begin{pmatrix}
pq&p(1-q)\\
(1-p)q&(1-p)(1-q)
\end{pmatrix}.
$$

The four entries sum to one. Summing across rows gives Player1's action probabilities, and summing down columns gives Player2's. Independence is an additional assumption: the two marginal vectors alone do not determine the joint law. Two players can each have fair marginal choices while their actions always match or always differ.

The first diagram labels the row and column marginals separately from each cell probability. Cell shading encodes joint mass. The outline selects a cell for inspection; it does not report a sampled event. The experiment never fabricates a random realization from a query control.

## Compute each player's expected utility separately

Let A contain Player1's utilities and B Player2's, using the same row/column ordering. Under an independent mixed profile,

$$
U_1(x,y)=x^TAy,\qquad U_2(x,y)=x^TBy.
$$

For example,

$$
U_1=pqA_{00}+p(1-q)A_{01}+(1-p)qA_{10}+(1-p)(1-q)A_{11}.
$$

The expression for Player2 uses B, not A or its transpose by default. The probabilities are shared because both utilities refer to the same joint event, but the payoffs need not be shared. In the general joint-law comparison, compute U_i as the sum of P_ab times that player's cell utility; a bilinear product formula applies only when P factorizes.

An expectation is an average under the specified law. It is not a number each player receives on every play. Repeated independent samples can estimate it under suitable assumptions, but this lab computes the finite expectation directly rather than running a simulation.

## Worked example: fair independent coordination

In the coordination game the diagonal cells have utility pairs(2,2) and(1,1); both off-diagonal cells have(0,0). At p=q=0.5 with independence, each cell has probability0.25. Both expected utilities are

$$
0.25\cdot2+0.25\cdot0+0.25\cdot0+0.25\cdot1=0.75.
$$

One actual play instead produces utility2,0 or1. The contribution diagram shows each weighted cell value in a separate row: the blue mark is Player1's contribution and the lower orange mark is Player2's. Their sums are displayed separately even when they happen to coincide in this example.

Now keep both marginals fair but couple the actions to always match. The diagonal cells each receive mass0.5, making both expectations1.5. If actions always differ, each off-diagonal cell has mass0.5 and both expectations are zero. None of the marginal probabilities changed. The changed joint law explains the changed expectation.

## Derive all feasible couplings of two binary choices

Set s=P(R0,C0). The marginals force the entire table:

$$
P=\begin{pmatrix}s&p-s\\q-s&1-p-q+s\end{pmatrix}.
$$

Every entry must be nonnegative, which yields

$$
\max(0,p+q-1)\le s\le\min(p,q).
$$

Conversely, any s in this interval gives a nonnegative table with the required marginals and total mass one. The comparison slider moves from the lower endpoint to the upper endpoint; it is not a correlation coefficient. Independent mixing is the particular choice s=pq, which lies in the interval. The numerical factorization diagnostic checks that equality exactly on the control grid.

For binary indicators of action0, covariance is s−pq. At fair marginals its extreme values are−0.25 and0.25. A covariance of zero implies independence for this binary pair because the single free cell s then equals pq. That special conclusion should not be generalized to arbitrary random variables.

## Dependence also changes conditional probabilities

When p is positive,

$$
P(C0\mid R0)=\frac{s}{p}.
$$

Under independence this equals q. Under fair perfectly matched actions it equals one, despite the marginal probability q=0.5. Observing an action can therefore reveal information in a dependent joint law.

If p=0, the elementary conditional ratio is undefined. The interface shows an unavailable value rather than zero. This is different from a cell utility: a utility function can assign a value to an action pair even if the current distribution gives it no mass. No conditional value at a zero-probability event is identified by this finite joint table.

## Worked example: asymmetric utilities at a pure profile

In the social-dilemma matrix, R0C0 has payoffs(3,3), R0C1 has(0,5), R1C0 has(5,0), and R1C1 has(1,1). Set p=1 and q=0. All mass is at R0C1, so Player1's expected utility is0 and Player2's is5. Reusing the first payoff for both players would give an incorrect answer.

Query R0C0. Its current probability is zero, but the game's payoff pair(3,3) remains defined. The query has not made that event occur. The conditional P(R0|C0) is unavailable because P(C0)=0. The conditional P(C0|R0) is defined and equals zero because P(R0)=1. Reversing the conditioning changes both the denominator and the question.

## Exact grid arithmetic and the reference plot

With percentage controls P and Q, independent cell00 has numerator PQ over denominator10,000. In comparison mode define L=max(0,P+Q−100), H=min(P,Q), and coupling percentage r. The numerator is100L+(H−L)r. The other numerators follow by integer subtraction. This keeps simplex, marginal and factorization diagnostics exact before values are converted for display. Expected-utility numerators are exact integers too because this lab's cell payoffs are integers.

Display labels are rounded. The table retains integer numerators for audit. The final plot sweeps Player1's probability while holding Player2's fixed, always using an independent profile. A correlated current-law expectation is not silently placed on that independent curve. Current-law markers appear only when the current joint table factorizes, even if a comparison-mode setting happens to recover independence.

## Utility scale and the limits of this experiment

Utilities encode each player's preferences over modeled lotteries. They need not be money, and their units need not be comparable across players. Replacing one player's utility by alpha times that utility plus beta, with alpha positive, transforms every expectation by the same positive affine map. That preserves its ranking of mixed outcomes. A general increasing nonlinear transformation can change rankings of lotteries, even when it preserves rankings of deterministic outcomes.

The experiment computes distributions and expectations. It does not certify a Nash equilibrium, a correlated equilibrium, an incentive-compatible recommendation system or convergence of any learning algorithm. A feasible correlated table need not satisfy the incentive constraints of correlated equilibrium. Those distinctions matter when extending these ingredients to multi-agent learning.

## Research extension: strategic equivalence does not preserve welfare rankings

A broader invariant than a common utility offset is

$$u_i'(a_i,a_{-i})=\alpha_i u_i(a_i,a_{-i})+h_i(a_{-i}),\qquad\alpha_i>0.
$$

Against any fixed opponent mixture, the added term has the same expectation for every own action. Every unilateral utility difference is multiplied by the positive $\alpha_i$. Thus best responses and the complete Nash set are unchanged. A transformation depending on the player's own action does not have this guarantee.

For the coordination game, keep Player2's payoffs unchanged and add three to Player1's payoff whenever Player2 chooses action1. The row matrix changes from $(2,0;0,1)$ to $(2,3;0,4)$. Its action difference remains $3q-1$. Nevertheless the sum of utilities at the two pure equilibria reverses order:

| Profile $(p,q)$ | Original utility sum | Transformed utility sum |
|---|---:|---:|
| (1,1) | 4 | 4 |
| (1/3,1/3) | 4/3 | 10/3 |
| (0,0) | 2 | 5 |

```python
from fractions import Fraction as F
from itertools import product
A=[2,0,0,1]; B=A[:]; transformed=[2,3,0,4]
def utility(M,p,q):
    masses=[p*q,p*(1-q),(1-p)*q,(1-p)*(1-q)]
    return sum(m*x for m,x in zip(masses,M))
for p,q in product([F(0),F(1,3),F(1,2),F(1)],repeat=2):
    for deviation in [F(0),F(1)]:
        assert utility(A,deviation,q)-utility(A,p,q)==utility(transformed,deviation,q)-utility(transformed,p,q)
for p,q,old,new in [(F(1),F(1),F(4),F(4)),(F(1,3),F(1,3),F(4,3),F(10,3)),(F(0),F(0),F(2),F(5))]:
    assert utility(A,p,q)+utility(B,p,q)==old
    assert utility(transformed,p,q)+utility(B,p,q)==new
    print(p,q,old,new)
```

This is a modeling warning for multi-agent benchmarks: identical strategic predictions do not identify a scientifically meaningful welfare scale. Interpersonal comparisons need their own justification. The construction is auxiliary; native payoff matrices remain unchanged. Equilibrium invariance also does not guarantee identical learning trajectories for updates that depend on payoff magnitudes.

## Exercises with explained solutions

Exercise1: p=0.7 and q=0.4. Find the allowed interval for s, the independent joint table and P(C0|R0) under independence.

Solution: s is between0.1 and0.4. Independence gives s=0.28 and the table entries(0.28,0.42,0.12,0.18) in row-major order. The conditional is0.28/0.7=0.4. At the upper coupling endpoint it becomes0.4/0.7=4/7, showing why a marginal alone is insufficient.

Exercise2: in the opposed-payoff game, Player1 receives1 when actions match and−1 otherwise; Player2 receives the negative. Compare fair independent, perfectly matched and perfectly opposed joint laws.

Solution: their expected utility pairs are(0,0),(1,−1) and(−1,1). The sum is always zero in this particular game. Shared marginal probabilities therefore do not imply identical utilities or identical joint outcomes.

Exercise3: compare a lottery giving utility0 or4 equally with a certain utility1.5. The original expectation ranks the lottery higher:2>1.5. After applying square root to those numbers, the lottery has expectation1 while certainty gives sqrt(1.5)>1. This demonstrates why an arbitrary increasing transformation is not a harmless change of expected-utility scale.
