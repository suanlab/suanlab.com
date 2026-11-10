## Fix the direction of each player's objective

The row player maximizes a payoff F, while the column player minimizes it. Their utilities are F and−F, so the game is zero-sum. With row action0 probability p and column action0 probability q,

$$
F(p,q)=(p,1-p)A(q,1-q)^T,\qquad p,q\in[0,1].
$$

The four entries of A can be positive, zero or negative. A negative row payoff benefits the column player relative to zero. The interpretation follows the utility signs, without requiring actual monetary transfers.

The players' randomizations are independent, and the opponent does not observe the current realized action before acting. An opponent can respond to a known distribution, but that differs from responding after seeing its sampled action. Giving the opponent that extra information changes the game and can remove the advantage of private mixing.

## Pure choices can leave a gap

If both players are restricted to pure actions, the row player can guarantee the largest row minimum, while the column player can cap the payoff at the smallest column maximum:

$$
\underline v_{\rm pure}=\max_i\min_j A_{ij},\qquad
\overline v_{\rm pure}=\min_j\max_i A_{ij}.
$$

The lower value cannot exceed the upper one, but equality need not hold. For A=(1,−1;−1,1), each row contains−1 and each column contains1. The pure guarantees are−1 and1. No pure saddle exists. This is a failure of pure-choice equality, not a counterexample to mixed finite-game minimax.

Mixing changes the feasible strategy set from a few actions to a probability simplex. It is not the same as replacing a discrete action by the arithmetic mean of its physical value. Expected payoffs are linear in the probabilities, using the full payoff matrix.

## Security envelopes for a chosen mixture

A fixed row mixture p produces two expected payoffs, depending on the opponent's pure column. The smaller is its security level. Similarly, a fixed column mixture q faces two pure rows and its upper bound is their maximum:

$$
L(p)=\min\{pA_{00}+(1-p)A_{10},\ pA_{01}+(1-p)A_{11}\},
$$

$$
U(q)=\max\{qA_{00}+(1-q)A_{01},\ qA_{10}+(1-q)A_{11}\}.
$$

Taking mixtures of columns cannot lower the row payoff below its smallest pure-column payoff. Taking mixtures of rows cannot exceed the largest pure-row payoff. Therefore, at every feasible profile,

$$
L(p)\le F(p,q)\le U(q).
$$

The lower envelope is concave and piecewise affine in p. The upper envelope is convex and piecewise affine in q. The blue and orange curves use different mixture variables on the same unit interval. They are not two players' utilities along one shared trajectory.

## Weak bounds and the minimax value

For any feasible p and q, L(p) cannot exceed U(q). Optimizing gives the weak minimax inequality. The finite mixed zero-sum theorem gives equality and attainment:

$$
\max_p L(p)=v=\min_q U(q).
$$

It follows that any current strategies bracket the true game value:

$$
L(p)\le v\le U(q).
$$

Notice the two different statements. Both the current payoff F and the optimal value v lie between the bounds; they need not equal each other. Equality of optimized bounds does not imply equality at every nonoptimal query.

[Goemans's MIT linear-programming notes, section7](https://math.mit.edu/~shor/18.310/lpnotes310.pdf) develop finite zero-sum guarantees and prove minimax from LP duality. The lab's matrices and two-line envelope construction below are independently constructed examples.

## Why linear-programming duality applies

For a general finite payoff matrix and probability vectors x,y, the row player maximizes a guaranteed level ell subject to

$$
A^Tx\ge\ell\mathbf1,\quad \mathbf1^Tx=1,\quad x\ge0.
$$

The column player minimizes u subject to

$$
Ay\le u\mathbf1,\quad \mathbf1^Ty=1,\quad y\ge0.
$$

The level variables are free in sign. These form a primal-dual pair. Both probability simplices are nonempty, and finite payoff entries bound all security levels. LP strong duality identifies their optimal values. Thus the conclusion depends on the actual feasibility, objective and duality structure, rather than the spelling of “min” and “max”.

## Saddle inequalities and the full solution set

With row maximization and column minimization, a saddle profile satisfies

$$
F(p,q^*)\le F(p^*,q^*)\le F(p^*,q)
\quad\text{for all }p,q.
$$

Against q*, changing the row mixture cannot increase F. Against p*, changing the column mixture cannot decrease it. Reversing which variable minimizes reverses the displayed orientation. [Bertsekas's MIT Lecture9](https://ocw.mit.edu/courses/6-253-convex-analysis-and-optimization-spring-2012/20b272ad85dd2b1c0912db39ab30e0c0_MIT6_253S12_lec09.pdf) connects saddle points, minimax equality and attainment with those roles made explicit.

If p* optimizes L and q* optimizes U, the sandwich L(p*)≤F(p*,q*)≤U(q*) forces F=v. The same guarantees establish both saddle inequalities. Conversely, a saddle profile gives optimal security strategies. Consequently the entire zero-sum saddle set is the product of the row-optimal and column-optimal sets. In this two-action model each is a closed interval, possibly a single point.

## A dual gap measures the query's incentives

At a current profile the row player's maximum unilateral improvement is U(q)−F(p,q). The column player's own utility is−F, so its maximum improvement is F(p,q)−L(p). Their sum is

$$
g_{\rm row}+g_{\rm column}=U(q)-L(p)\ge0.
$$

Zero gap means both improvements vanish and certifies a saddle, hence a Nash equilibrium of utilities(A,−A). A positive gap describes the current strategies, while the game can still have an exact minimax value. Current query probabilities use n/12; their gains have integer numerators over144. The exact zero test is separate from rounded display labels.

## Worked example: mixing closes the pure gap

For A=(1,−1;−1,1), the pure bounds are−1 and1. At p=q=1, F=1, L=−1 and U=1. The current dual gap is2: the row player already receives its best payoff against q=1, while the column player can gain2 by switching columns.

At p=q=1/2, both security envelopes equal0. Every pure opponent choice produces expected row payoff0. The mixed value is0 and the gap vanishes. Enter probability numerators6 and6 to verify it exactly. The optimal interval product is the single point(1/2,1/2).

## Worked example: different optimal probabilities

For A=(3,0;1,2), the row security lines are1+2p and2−2p. They meet at p=1/4, with value3/2. The column upper lines are3q and2−q; their intersection is q=1/2 with the same value. The pure lower and upper values are1 and2.

The optimal probabilities differ because they solve different envelope problems. The second preset uses p=3/12 and q=6/12. All four bound/value markers coincide at1.5, and the gap is zero. The curve corners are exact rational breakpoints, not a coarse numerical search.

## Degenerate and off-grid solutions

For A=(2,2;0,0), the row player guarantees2 at p=1. Every column mixture is optimal. The saddle set is a vertical segment, not a uniquely chosen q. A constant matrix makes both optimal intervals[0,1], giving a full square of saddle profiles.

For A=(4,0;0,3), both optimal probabilities are3/7 and the value is12/7. No probability n/12 equals3/7. Nevertheless the solution table and curve corners retain that exact fraction. At p=q=5/12, the bounds are5/3 and7/4, with positive gap1/12. Snapping the optimum to the query grid would falsely hide this distinction.

The solver evaluates endpoints and the intersection of two affine lines when it lies inside the unit interval. A piecewise affine envelope reaches an optimum at such a breakpoint or on a flat interval with breakpoint endpoints. Collecting all optimal endpoints yields the full optimal interval. This exact small-matrix method is not a general large-scale LP implementation.

## When swapping min and max is invalid

The finite mixed theorem uses continuous bilinear payoffs on probability simplices. In broader continuous minimax problems, the domain, attainment and convexity/concavity assumptions must be checked. Compact convex domains alone do not suffice.

Consider F(x,y)=(x−y)^2 on[−1,1]^2, with x maximizing and y minimizing. For each x the minimizing y=x gives0, so max_x min_y F=0. For fixed y, the worst endpoint x gives(1+|y|)^2; its minimum is1 at y=0. Thus min_y max_x F=1. The function is convex rather than concave in the maximizing variable. This counterexample concerns choices of continuous values; randomizing those choices would create a different expected-payoff problem.

An existence or duality theorem also does not establish convergence of a particular ascent/descent method. A zero-sum learning experiment requires its own update rules, stability assumptions and convergence analysis.

## Exercises with explained solutions

Exercise1: for the matching matrix, query p=3/4 and q=1/4. Compute L,U,F and both unilateral gains.

Solution: L=−1/2, U=1/2 and F=(2p−1)(2q−1)=−1/4. Row gain is3/4 and column gain1/4. Their sum1 equals the dual gap. The game value0 lies between the bounds, as does the distinct current payoff−1/4.

Exercise2: in A=(2,2;0,0), does choosing q=5/12 instead of q=0 invalidate optimality at p=1?

Solution: no. Both columns give payoff2 against p=1, so every q belongs to the optimal interval. Reporting one q as unique would be incorrect. Moving p below1 does lower the row guarantee and creates a positive gap.

Exercise3: for A=(4,0;0,3), explain why a near-zero rounded readout is insufficient to certify an exact saddle at a query-grid point.

Solution: the exact optimal probability is3/7, outside that grid. Check the integer gap numerator and exact optimal intervals. A displayed decimal near the optimum is an approximation, while the rational solution and exact zero test describe different claims.
