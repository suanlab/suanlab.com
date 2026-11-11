## Three claims that require different evidence

A solver can fail even when a solution exists. A sequence can converge to the fixed point of the wrong map. A tiny update can reflect rounding or a weak contraction rather than an equilibrium. Separate three questions before interpreting a learning curve:

1. Does the stated mathematical problem have a solution?
2. Are solutions exactly the fixed points of the map being used?
3. Does the chosen iteration converge to such a fixed point?

The previous unit studied a quadratic game whose saddle exists while several gradient updates rotate or diverge. This unit moves to finite games and a compact mixed-strategy square. It compares the continuous Nash proof map, a pure best-response selector and two auxiliary maps. Existence and convergence remain separate even after compactness is available.

## The five explicit games

P1 chooses a row and P2 a column. Each has actions0 and1. The controlled probabilities p and q are the probabilities of action0. Mixed actions are independent. Each payoff list is ordered00,01,10,11.

| Family | P1 payoffs A | P2 payoffs B |
| --- | --- | --- |
|0 coordination|2,0,0,1|2,0,0,1|
|1 opposed payoffs|1,−1,−1,1|−1,1,1,−1|
|2 social dilemma|3,0,5,1|3,5,0,1|
|3 P1 indifferent|0,0,0,0|2,0,0,1|
|4 both indifferent|0,0,0,0|0,0,0,0|

The expected payoff of P1 is

$$
u_1(p,q)=pqA_{00}+p(1-q)A_{01}+(1-p)qA_{10}+(1-p)(1-q)A_{11}.
$$

P2 uses B in the same expectation. Increasing P1's expected payoff says nothing automatic about P2's payoff. Each player's unilateral regret is its best pure-action payoff against the current opponent minus its current expected payoff. Pure deviations suffice because an arbitrary mixed deviation is an average of those pure payoffs.

The full Nash set is solved as in the earlier best-response-set unit. The figure draws its complete union of points, line segments or rectangles. “Pieces” counts the rectangles used to represent the union, not topological connected components or the number of equilibria in a continuum. The orange set belongs to the selected MAP and is computed separately.

## Why finite mixed games have the right domain

Each player's probability vectors form a simplex: entries are nonnegative and sum to1. With finitely many nonempty action sets, their product is nonempty, compact and convex in a finite-dimensional Euclidean space. Expected payoffs are finite sums of products of probabilities, hence continuous and linear in one's own mixed strategy.

Brouwer's fixed-point theorem says that a continuous self-map of such a set has a fixed point. All of the conditions matter: the domain must be nonempty, compact and convex, and the map must be continuous and take every point back into that domain. The conclusion is existence. It does not state that repeatedly applying the map converges, nor that the fixed point is unique.

One must also construct the right map. A fixed point of an arbitrary continuous self-map is not automatically an equilibrium of a game.

## Construct the normalized positive-gain map

Let s_i be a player's mixed-strategy vector and let s denote the joint profile. For each pure action a define the positive improvement

$$
g_{i,a}(s)=\max\{0,u_i(a,s_{-i})-u_i(s)\},\qquad
G_i(s)=\sum_a g_{i,a}(s).
$$

The Nash map adds every positive gain to its action probability, then renormalizes:

$$
(T_N(s))_{i,a}=\frac{s_{i,a}+g_{i,a}(s)}{1+G_i(s)}.
$$

The denominator is at least1, entries remain nonnegative and each player's entries sum to1. Positive part and finite expected utilities are continuous, so the map is continuous. It is therefore a self-map of the compact convex strategy domain to which Brouwer applies.

These facts establish that T_N has a fixed point. A further argument is needed to identify that point with a Nash equilibrium.

## Prove fixed-point and Nash equivalence

At a fixed point, multiplying by the denominator yields g_{i,a}=s_{i,a}G_i for every action. Suppose G_i>0. Every action carrying positive strategy probability would then have a strictly positive gain. Its actual payoff difference from the mixed expected payoff is positive. But

$$
\sum_a s_{i,a}\bigl(u_i(a,s_{-i})-u_i(s)\bigr)=0,
$$

because the mixed payoff is precisely this weighted average. The supposed strict positivity on every supported action contradicts the equality. Thus G_i=0 for every player and no profitable unilateral deviation exists.

Conversely, at a Nash equilibrium every positive gain vanishes, so T_N returns the original strategy. Hence fixed points of this particular map are EXACTLY Nash equilibria. Combining this relation with Brouwer proves existence for finite mixed games. It does not prove a convergence rate for iterating T_N.

The standard proof-map framework and theorem background are described in [MIT6.7980 Lecture1](https://www.mit.edu/~6.7980/nfgs_nash.html) and the compact-domain fixed-point argument in [Lecture2](https://www.mit.edu/~6.7980/brouwer.html). The explicit game calculations and contrasting iterations below are the lab's own finite examples.

## A useful improvement identity has a unilateral scope

Hold opponents fixed and change ONLY player i to its Nash-map strategy. Direct substitution gives

$$
u_i((T_N(s))_i,s_{-i})-u_i(s)
=\frac{\sum_a g_{i,a}(s)^2}{1+G_i(s)}.
$$

Thus positive gains produce a strictly profitable unilateral change. If everyone moves at once, opponents are no longer fixed. The identity does not promise that every player's realized payoff increases.

For example, in the opposed game start at (p,q)=(1,1). P1 has payoff1 and no positive gain. P2 has payoff−1 and a gain2 from switching to its other action. The joint Nash map gives(1,1/3). P2 improves by4/3, but P1's realized payoff falls from1 to−1/3 because its opponent changed. The unilateral identity is valid; an all-player monotonic-payoff interpretation is not.

## Exact certificates for a controlled profile

The p and q inputs are twelfths. The lab computes their payoffs and pure deviations using integer numerators over144. It computes the controlled profile's map images and fixed/Nash statuses as reduced rational numbers, without a tolerance.

In coordination at p=q=1/3, both pure payoffs of each player equal2/3. Every positive gain is0. The proof map returns exactly(1/3,1/3). The full Nash set also contains(0,0) and(1,1), so existence does not imply uniqueness.

The later trajectory is deliberately separate: repeatedly forming rational Nash-map outputs can cause rapidly growing denominators, so these iterations use floating arithmetic. A plotted point that appears fixed, or a rounded residual that looks zero, does not become an exact Nash certificate. The exact panel refers to the controlled initial input only.

## A continuous self-map can cycle forever

Select the opposed game and the auxiliary quarter-turn

$$
Q(p,q)=(q,1-p).
$$

It maps the square continuously to itself. Its unique fixed point is(1/2,1/2), since p=q and q=1−p. Yet starting at(1,1) gives

$$
(1,1)\to(1,0)\to(0,0)\to(0,1)\to(1,1)\to\cdots.
$$

Every Brouwer hypothesis holds, but the iteration has period4. A fixed point exists without attracting this orbit. Q is an auxiliary map, not the normalized Nash map. Its fixed point happens to coincide with the opposed game's Nash equilibrium; changing the game can break that connection.

The mean of states0..3 is(1/2,1/2), which is Nash in this particular game, while the terminal corner is not. This calculation is not a general theorem that averaging arbitrary game iterates yields Nash.

## Best-response selection can discard a mixed fixed point

Rule1 selects each player's best PURE action simultaneously. At a tie it chooses action0. In the opposed game, both players are indifferent at(1/2,1/2), so the selected next point is(1,1). A true mixed Nash equilibrium is not fixed by this single-valued tie-breaking rule.

The full best-response correspondence still contains the equilibrium strategy. Selecting one pure response is a different mathematical object. In several families that selector is discontinuous across an indifference boundary, so Brouwer cannot simply be applied to it. In the social-dilemma and fully indifferent families the particular selections are constant and continuous; a selector is not necessarily discontinuous in every game.

Any fixed point of this pure selector is a pure Nash profile, but the reverse implication for arbitrary mixed Nash profiles is false. The model enumerates the selector's complete fixed set independently rather than assuming it equals the full Nash set.

## A contraction may converge to the wrong target

The auxiliary map

$$
C_\lambda(z)=c+\lambda(z-c),\qquad c=(1/2,1/2),\quad0\le\lambda\le1
$$

has Lipschitz constant lambda on the closed square. For lambda<1, this complete metric domain and strict contraction give a unique fixed point c, with

$$
\|z_T-c\|=\lambda^T\|z_0-c\|.
$$

This is a convergence guarantee for C_lambda. It says nothing by itself about Nash. In coordination at c, each current payoff is3/4 and the best unilateral payoff is1. Both regrets are1/4. The auxiliary contraction fixes the point exactly, but the Nash map sends it to(3/5,3/5). The latter's unilateral improvement is1/20 for each player with the opponent fixed.

Starting the auxiliary contraction at a corner genuinely converges to that same non-Nash target. Switching the game to opposed payoffs makes the same center Nash. The algorithm's geometric convergence does not decide which game it solves.

At lambda1, C is the identity: the whole square is fixed. There is no strict contraction or unique-center claim. For lambda<1 the residual-to-center relation is

$$
\|z-c\|=\frac{\|C_\lambda(z)-z\|}{1-\lambda}.
$$

A small displacement near lambda1 can coexist with substantial distance. At lambda1 the division is invalid. Even a correct distance bound to c would not repair c being the wrong game solution.

## What happens when existence assumptions are missing

These are separate mathematical examples, not claims that the finite mixed-game theorem fails:

- Without compactness, a one-player problem maximizing u(x)=x on the open interval(0,1) has no maximizer. For EVERY feasible x, (1+x)/2 is feasible and strictly better. A sequence approaching1 does not attain the supremum.
- Without convexity, the compact domain{0,1} admits the relative-topology continuous self-map T(x)=1−x with no fixed point. Its mixed extension[0,1] admits1/2. In the opposed game the pure strategy profiles contain no Nash equilibrium while the mixed domain does.
- Without continuity, define T(x)=1 for x<1 and T(1)=0 on[0,1]. For x<1 the image differs from x; at1 it also differs. The compact convex domain and self-mapping property do not suffice.
- Without self-mapping, T(x)=x+delta with delta>0 sends the endpoint1 outside[0,1]. Brouwer's hypotheses are not satisfied.

Numerical samples can illustrate these examples. Their universal claims follow from the case arguments, not from a finite grid failing to find a fixed point.

## Read residuals and averages with their units

The lower plot compares maximum unilateral regret at the current strategy and at the prefix mean. Both are in the game's payoff units. Map displacement is a different quantity in probability units; it is reported separately in the table and exact map panel.

A profile is epsilon-Nash when every unilateral regret is at most epsilon. The controlled-profile integer calculation can test that definition directly. A small proof-map displacement requires a quantitative conversion before being used as an epsilon-Nash certificate; merely calling the two residuals “small” does not establish such a conversion.

At T>0 the mean uses states0..T−1 and excludes terminal stateT. At T0 there is no mean or queried transition; the initial state remains defined. The transition query clamps to T−1 and changes only inspection. Lambda affects only the auxiliary contraction. None of these display/query changes alters the other maps' dynamics.

## Research extension: turn a Nash-map displacement into an incentive bound

For this particular two-action Nash map, let d be the difference between pure payoffs, r the probability on the inferior action, and g=|d|r the unilateral gain. Only the better action has positive gain. Direct normalization gives coordinate displacement

$$\delta=\frac{|d|r^2}{1+|d|r},\qquad g^2=|d|\delta(1+g).
$$

If the player's payoff range is at most M, then |d| and g are at most M. Therefore $g\le\sqrt{M(1+M)\delta}$. A sharper bound is the positive root of $g^2-M\delta g-M\delta\le0$. For two players, use the largest coordinate displacement and the relevant range for each player. This is an explicit conversion for the canonical map, not for an arbitrary auxiliary fixed-point iteration.

A square-root rate is natural near a pure equilibrium. With pure payoffs1 and0, g=r while displacement is quadratic in r:

| Inferior mass r | Gain g | Map displacement |
|---|---:|---:|
| 1/10 | 1/10 | 1/110 |
| 1/100 | 1/100 | 1/10100 |
| 1/1000 | 1/1000 | 1/1001000 |

```python
from fractions import Fraction as F
for denominator in [10,100,1000]:
    r=F(1,denominator); p=1-r; gain=r
    image=(p+gain)/(1+gain); displacement=image-p
    assert displacement==r*r/(1+r)
    assert gain*gain==displacement*(1+gain)
    assert gain*gain<=2*displacement
    alpha=F(1,100); damped=(1-alpha)*p+alpha*image
    assert damped-p==alpha*displacement
    assert gain/displacement==denominator+1
    print(r,gain,displacement)
```

No finite uniform linear gain/displacement ratio works as r tends to zero. Damping the map by alpha multiplies displacement by alpha without reducing the current incentive, so a valid conversion must first divide by alpha. These auxiliary fractions extend beyond the native twelfth-grid input; no new convergence guarantee is attached to its floating trajectory.

## Exercises and worked solutions

1. List the coordination game's Nash equilibria. They are the two pure coordination corners(0,0),(1,1) and the mixed point(1/3,1/3). The finite-game existence theorem does not select one or imply uniqueness.
2. At coordination(1/2,1/2), calculate the Nash map. Each positive action0 gain is1/4 and the action1 positive gain is0, so the new action0 probability is(1/2+1/4)/(1+1/4)=3/5. The auxiliary center-fixing map is a different map.
3. At lambda1/2, from(1,1), where does the auxiliary contraction go after eight updates? Each coordinate is1/2+(1/2)^9=257/512. The same iterate can be close to a correct center in one game and a wrong center in another.
4. Does the pure selector leaving a mixed Nash point refute existence? No. It refutes fixedness under that particular selection. The normalized positive-gain map retains the exact mixed fixed point and has the required continuity.

## Use the distinction in AI research

State the solution concept and assumptions first. Then prove that the algorithm's target solves that problem, specify how iterates are generated, and report the measured quantity with a justified stopping rule. Existence, uniqueness, approximation accuracy, convergence rate and computational cost require different evidence.

This lab exhaustively represents its small explicit sets and checks controlled rational profiles. It does not implement a general large-game Nash solver, a Sperner algorithm or a proof that every learning rule converges. The next domains apply the same discipline to privacy, fairness, robustness and physical modeling.
