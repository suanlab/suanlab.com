## A stable choice means stable against which changes?

A Nash equilibrium rules out a profitable unilateral change when everyone else's strategy stays fixed. It does not allow a player to choose a new action for the opponent. It also does not require the players to maximize a common score. These distinctions matter even in a two-player, two-action game.

Write p for Player1's probability of action0 and q for Player2's. The players randomize independently. Matrices A and B contain their separate utilities, using Player1's action as the row and Player2's as the column. The current utilities are the bilinear expectations from the previous unit.

[MIT's Nash Equilibrium notes, sections6.1 and6.3](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2025/mit14_12f12_chapter6_f12.pdf) define mutual best responses and the support condition for mixed equilibria. The examples, exact set construction and deviation controls here are our own finite teaching model.

## Define best responses before checking equilibrium

Against a fixed q, Player1 compares the two pure-action utilities

$$
r_0(q)=qA_{00}+(1-q)A_{01},\qquad
r_1(q)=qA_{10}+(1-q)A_{11}.
$$

Against a fixed p, Player2 compares

$$
c_0(p)=pB_{00}+(1-p)B_{10},\qquad
c_1(p)=pB_{01}+(1-p)B_{11}.
$$

If r0 is larger, Player1's best response has p=1; if r1 is larger it has p=0. When they tie, every mixture is a best response. Player2 has the analogous rule for q. A best response to one specified opponent strategy differs from a dominant strategy, which must be optimal against every opponent strategy.

The set diagram has p on the horizontal axis and q vertically. Solid blue is Player1's best-response correspondence. Dashed orange is Player2's. At indifference these are sets, not a single arbitrarily selected action. Their intersection is the complete Nash set, drawn with thick dark points, segments or region boundaries. The diamond marks the queried profile.

## Why checking pure deviations is enough

Holding q fixed, a new mixture p' gives Player1 utility

$$
U_1(p',q)=p'r_0(q)+(1-p')r_1(q).
$$

This is a convex combination of two numbers. Its maximum is their maximum, so no mixture can improve beyond the best pure deviation. The same reasoning holds for Player2, and for any finite number of pure actions: expected utility is affine in one's own distribution.

This gives a finite test even though the strategy simplex is continuous. It depends on expected utility and independent mixed profiles with the opponent held fixed. It is not a check of the incentives after a private recommendation in a correlated device.

## Unilateral gain is a certificate for this profile

Define each player's maximum unilateral improvement:

$$
g_1(p,q)=\max(r_0(q),r_1(q))-U_1(p,q),\qquad
g_2(p,q)=\max(c_0(p),c_1(p))-U_2(p,q).
$$

The current strategy is itself available, so these gains are nonnegative. The profile is Nash exactly when both vanish. A proposed deviation in the controls is only one possible change; it can lose utility even when another deviation would improve it. Therefore a failed proposal alone is not an equilibrium certificate. The maximum gains compare every pure action and hence every mixed deviation.

The lab queries probabilities n/12. All payoff entries are integers, so current utilities and gains have exact integer numerators over144. A zero gain is tested by a zero numerator, without substituting a floating-point tolerance for equality. Readouts are rounded for readability; the numerator remains available for audit.

## Support must contain only best actions

At equilibrium, every pure action receiving positive probability must be a best response. Suppose a worse action had positive mass. Moving some of that mass to a better action strictly increases expected utility, contradicting equilibrium. Conversely, mixing only among best actions gives the same maximal utility, so it is a best response.

For a player using both actions with positive probability, the two pure utilities must be equal. This is an indifference condition, not a rule that both probabilities are one-half. Each player's mixture often makes the other player indifferent. Solve those equations with the correct payoff matrix and then check probability bounds and support conditions.

## Worked example: three coordination equilibria

For coordination, A=B has entries(2,0,0,1) in row-major order. Player1's pure utilities are2q and1−q. They tie at q=1/3. Player2 is indifferent at p=1/3. Thus

$$
(p,q)=(1/3,1/3)
$$

is a fully mixed equilibrium, with utility2/3 to each player. Changing one player's mixture against the fixed one-third opponent leaves its utility2/3. The first preset uses numerators4 and4 over12 so the query is exact.

There are also pure equilibria(0,0) and(1,1), giving utilities(1,1) and(2,2). The diagram displays all three. The lower-payoff pure equilibrium is still Nash: one player alone cannot switch both actions to obtain the better diagonal outcome. Multiple equilibria also mean that solving the equilibrium conditions does not select a unique prediction.

## Worked example: an equilibrium that is not jointly efficient

In the social dilemma, the payoff pairs are(3,3),(0,5),(5,0),(1,1). At p=q=0 both choose action1 and receive1. Either player switching alone to action0 receives0, so both maximum gains are zero.

If both switch to action0, the pair becomes(3,3). That joint improvement does not invalidate the original Nash equilibrium. At p=q=1, however, either player can switch alone to action1 and obtain5 instead of3. The unilateral gain is2, so joint cooperation here is not Nash. This example separates equilibrium, joint efficiency and the incentive for a single player to change.

The proposal square in the second diagram changes only the selected player. Each player's current dot and two pure-deviation markers use the same utility scale. The two curves below vary each player separately with its own opponent held fixed; they are not a simultaneous joint-change path.

## Worked example: opposed payoffs and no pure equilibrium

In the opposed-payoff family, Player1 receives1 for matching actions and−1 otherwise, while Player2 receives the negative. Every pure profile gives one player an incentive to switch. There is no pure equilibrium, but p=q=1/2 makes both players indifferent. The mixed equilibrium is the single point(1/2,1/2).

The absence of a pure equilibrium is therefore not the absence of equilibrium. For every finite normal-form game there exists a mixed-strategy Nash equilibrium. This is an existence statement about the game, not a claim that a chosen learning algorithm will reach it, that it is unique or that finding it is always computationally easy. The finite examples demonstrate those distinctions; they do not constitute a general existence proof.

## Degeneracy: a whole segment can be equilibrium

In family3, Player1's utility is always zero, while Player2 has the coordination utility matrix. Player1 is indifferent everywhere. Player2 chooses q=0 when p<1/3, q=1 when p>1/3, and any q at p=1/3. The Nash set is the union

$$
\{(p,0):0\le p\le1/3\}\ \cup
\{(1/3,q):0\le q\le1\}\ \cup
\{(p,1):1/3\le p\le1\}.
$$

The third preset queries p=4/12 and q=7/12, a point on the vertical segment. Reporting only(1/3,1/3) would omit infinitely many valid equilibria. In family4 both players are indifferent everywhere, so the entire unit square is equilibrium. The plot shows a region instead of inventing one privileged solution.

## Constructing the continuous equilibrium set exactly

For a two-action player, the pure-utility difference is affine in the opponent probability. Find its rational zero, if any, and partition the corresponding axis into singleton boundaries and open intervals. Partition p with Player2's indifference root and q with Player1's. On each product atom the utility-difference signs are constant; being a boundary or an interior own mixture also remains fixed. Consequently each best-response condition has constant truth on that atom.

Keep atoms satisfying both conditions. Their closures also satisfy them because both unilateral gain functions are continuous. Remove rectangles contained in larger returned ones. The resulting union can represent points, line segments and rectangles. These are continuous sets, not just a search through the n/12 query grid. The coordinate table records exact rational endpoints.

This construction exploits the special structure of a two-player, two-action game with bounded integer payoffs. It is not an algorithm claimed to solve every large multi-player game. Its independent tests compare region membership against direct pure-deviation inequalities, including degenerate games and non-query-grid rational points.

## Exercises and explained solutions

Exercise1: in coordination, query p=q=1/2. Is it equilibrium? Compute a profitable pure deviation and the maximum gain.

Solution: each current utility is0.75. Against q=1/2, action0 gives1 and action1 gives0.5. Player1 gains0.25 by choosing action0 with certainty; Player2 has the same gain. The profile is not equilibrium even though both players use the same mixture. Equal probabilities are not an equilibrium test.

Exercise2: keep the social-dilemma opponent at action1 and propose action0. Does a negative proposed change prove equilibrium at every current strategy?

Solution: no. It only rejects that particular proposal. To certify equilibrium, compare the current utility with the maximum of both pure utilities. A current mixture that gives positive mass to the inferior action is not optimal even if some other proposed mixture is worse.

Exercise3: when Player1 is always indifferent, list the equilibria with p=1/3. When both players are always indifferent, how many equilibria are there?

Solution: in family3, every q in[0,1] is equilibrium at p=1/3. In family4, every(p,q) in the unit square is equilibrium, an uncountable continuum. The rectangle count in the readout counts geometric components of a representation, not the number of equilibrium profiles.
