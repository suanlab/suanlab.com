## Repetition adds histories, not just a longer sum

A repeated game plays a stage game over several dates. At the start of round t, a player can use the actions observed at dates0 through t−1. A strategy is a complete rule for what to do at each possible history, including histories never reached on one particular run. Two rules can generate the same observed actions and still differ in how they would react after a deviation.

In the lab, both current actions are chosen simultaneously. The four P1 rules are always0, always1, clock alternation and copying the previous opponent action. The opponent can always choose0, alternate, switch once at floor(horizon/2), or copy P1's previous action. Copy rules start from0. None reads the opponent's current action before choosing its own.

The stage utilities are the same ordered matrices used earlier. Blue circles show actual P1 actions; orange squares show P2. The solid box selects a round and the dashed box its predecessor. Rows beyond the displayed prefix do not appear.

## A stage equilibrium and a repeated-game equilibrium differ

[MIT's Repeated Games notes](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2025/mit14_12f12_chapter12_f12.pdf) distinguish finite repetition, history-dependent strategies and discounted infinite repetition. In a known finite horizon with observed histories, additive utilities and a unique stage Nash equilibrium, backward induction makes every subgame repeat that equilibrium: the last round has no future incentive, and its continuation value is fixed when reasoning about the preceding round.

For our social dilemma, both action1 is the unique stage equilibrium. This is a statement about subgame-perfect incentives under those assumptions. It is not obtained by merely plotting a cooperative trace. Different horizon, information or continuation assumptions can change the conclusion. The four illustrative policies are rules to evaluate, not policies certified to form a repeated-game equilibrium.

## Define the observed reward sequence

Let a_t and b_t be the actual actions. For Player1, the realized cumulative utility is

$$
S_T=\sum_{t=0}^{T-1}u(a_t,b_t).
$$

Because the matrix and observed opponent actions are known in this teaching model, we can also evaluate two fixed-action replays:

$$
C_T(a)=\sum_{t=0}^{T-1}u(a,b_t),\qquad a\in\{0,1\}.
$$

The replay keeps the actual opponent record b_t, even when it substitutes a different action in each payoff calculation. It does not regenerate the opponent's reactions. This is a specified hindsight comparator, not necessarily the outcome that would have happened had the learner used that policy from the start.

## Signed external regret and its positive part

For rewards, define the signed external regret against constant actions as

$$
R_T=\max_{a\in\{0,1\}}C_T(a)-S_T.
$$

The comparator is one action held fixed for the whole prefix. It is chosen in hindsight, after seeing the record. The plotted orange value uses the best fixed comparator separately at each prefix; its identity can change as the prefix grows. That orange envelope is not itself an executed online policy.

A changing learner can outperform every fixed action, so this signed difference can be negative. Some conventions use its positive part instead:

$$
R_T^+=\max(0,R_T).
$$

The lab reports both, and reports averages R_T/T and R_T^+/T only when T is positive. A displayed zero average from an empty run would falsely suggest evidence, so empty-prefix averages are unavailable. Later, a no-regret guarantee will concern a particular algorithm, comparator and adversary model over growing horizons. It is not established for these four simple policies.

## A stronger comparator knows each current outcome

The per-round hindsight oracle earns

$$
D_T=\sum_{t=0}^{T-1}\max_a u(a,b_t).
$$

It can choose a different action after knowing each current opponent action. In general it is a stronger benchmark than one fixed action:

$$
D_T\ge\max_a C_T(a),\qquad
D_T-S_T\ge R_T,\qquad D_T-S_T\ge0.
$$

The oracle is not an implemented learner and does not get silently substituted for the fixed comparator. Beating a fixed-action baseline is different from predicting every round's best action. A regret theorem for one comparator does not automatically transfer to the other.

## Worked example: negative signed regret without peeking

Use the opposed-payoff game: P1 receives1 when actions match and−1 otherwise; P2 receives the negative. For four rounds, let both players use clock alternation. Their action sequences are(0,1,0,1), so P1 gets1 each round and S_4=4. Always choosing0 against that record earns0, and always choosing1 also earns0.

Therefore R_4=0−4=−4, its positive part is0, and the dynamic oracle regret is0. There is no lookahead: the two deterministic rules happen to be synchronized by the known round clock. This constructed sequence does not show that alternation predicts arbitrary opponents or is a universal no-regret algorithm.

## Reactive opponents require a different counterfactual

If the opponent depends on the learner's past actions, changing a policy can change future opponent actions. A constant-policy rerun therefore regenerates the entire opponent reaction under that constant learner policy. Define its reward S_T^(a), and compare

$$
R_T^{\rm rerun}=\max_a S_T^{(a)}-S_T.
$$

This can differ from the replay comparator C_T(a). [Arora, Dekel and Tewari](https://arxiv.org/abs/1206.6400) make the distinction explicit: ordinary adaptive-adversary regret keeps the learner's actual preceding history in the comparator, while policy regret changes that history too. Their paper studies a bandit setting; we use the comparator distinction, not a transfer of its algorithmic bounds to this full-information simulation.

Our reruns are available because the simulator's response rule and initial condition are known. One observed action sequence alone does not identify every counterfactual reaction. The interface labels these as different comparison worlds rather than presenting replay as a causal policy-value estimate.

## Worked example: recorded replay versus regenerated reaction

Keep opposed payoffs, let P1 alternate, and let P2 copy P1's preceding action. The first four action pairs are(0,0),(1,0),(0,1),(1,0). P1's rewards are(1,−1,−1,−1), totaling−2.

The recorded opponent actions are(0,0,1,0). A fixed0 replay gets2 and fixed1 gets−2. External regret is2−(−2)=4. A genuine always0 rerun instead induces opponent actions(0,0,0,0), earning4. Always1 induces(0,1,1,1), earning2. The best constant-policy rerun yields regret4−(−2)=6.

The difference is not a numerical inconsistency: the comparator interventions differ. For an exogenous opponent rule that ignores learner actions, both worlds use the same opponent sequence, so the replay and rerun constant-policy rewards agree.

## Finite discounted evaluation is a separate objective

For a displayed prefix with T rounds, finite discounted utility is

$$
V_T(\delta)=\sum_{t=0}^{T-1}\delta^t u_t,\qquad 0\le\delta\le1.
$$

At discount0, the first existing round has weight1 and later rounds weight0. At discount1, the finite sum equals ordinary cumulative utility. Neither case requires dividing by1−delta. The finite discounted average divides by the sum of the existing weights; for an empty prefix both sums are zero and that average is unavailable.

The chart and external regret stay undiscounted when this control changes. Mixing the two objectives would quietly change the comparator. Undiscounted action rewards and regrets are integers here; finite discounted sums use floating-point arithmetic and rounded display values.

## Worked example: cooperative prefix and discounting

In the social dilemma, both copying rules start at action0 and remain there, earning3 each round. Show only the first3 rounds of a planned8-round run. Actual cumulative reward is9. At discount0.8,

$$
V_3=3+0.8\cdot3+0.8^2\cdot3=7.32.
$$

The fixed1 replay would earn15 against the recorded all-zero opponent, producing external regret6. But always1 would change a copying opponent: its three-round rerun gives rewards5,1,1 and total7. Always0 gives9, so rerun-policy regret is0.

A requested query round23 clamps to the last displayed round2. It does not display a future round from the planned horizon. Setting the displayed prefix to zero removes all round cells and selected actions, while retaining the cumulative origin(0,0) as a starting balance rather than a played round.

## Infinite incentives need their own assumptions

As a separate analytic example, consider infinitely repeated social dilemma with perfect public monitoring, common discount delta in[0,1), and grim-trigger strategies: cooperate until any defection, then both defect forever. For our payoffs R=3, temptation5 and punishment1, cooperation gives3/(1−delta). A one-round defection followed by punishment gives5+delta/(1−delta). The incentive condition is

$$
\frac3{1-\delta}\ge5+\frac\delta{1-\delta}
\quad\Longleftrightarrow\quad\delta\ge\frac12.
$$

After punishment starts, both defecting is a stage equilibrium, making the punishment credible. With finite actions, bounded utilities, discount below1 and the stated monitoring, the one-shot-deviation argument checks the relevant histories. At the threshold the player is indifferent. Discount1 does not give a finite infinite-horizon present value.

This analytic grim-trigger calculation is separate from the finite traces: a previous-action copy rule is not grim trigger, and a few cooperative rounds do not certify subgame perfection. The numerical discount threshold also depends on the chosen payoff matrix; do not copy one from a different dilemma example.

## Read the controls and preserve time scope

The block opponent switches at floor(planned horizon/2). Changing the horizon changes that environment's schedule and may change an early action; it is not a future observation by the learner. For the other opponent rules, extending the horizon preserves the existing prefix.

All summary rewards, comparator reruns and curves use only the displayed prefix. The horizontal curve coordinate counts completed rounds, while the history table uses zero-based round labels. Changing a queried round or discount does not change the actual action sequence. These distinctions make prefix experiments reproducible and prevent accidental use of future evidence.

## Research extension: noisy public monitoring changes cooperation incentives

Extend the infinite dilemma analytically, keeping cooperative reward3, deviation reward5 and punishment reward1. In the cooperative public state, a bad public signal occurs with probability $\alpha$ under cooperation and $\beta$ after one unilateral defection. A bad signal starts permanent mutual defection. Signals are public and their conditional probabilities are known; this is not the native previous-action copy rule.

For discount $d<1$, punishment value is $V_P=1/(1-d)$. Cooperation satisfies $V_C=3+d[(1-\alpha)V_C+\alpha V_P]$, hence $V_C-V_P=2/[1-d(1-\alpha)]$. The one-shot incentive condition is

$$2\le d(\beta-\alpha)(V_C-V_P)
\quad\Longleftrightarrow\quad d(1+\beta-2\alpha)\ge1.
$$

Punishment is credible because mutual defection is stage Nash. With the stated public strategy and monitoring model, these continuation inequalities verify the perfect public equilibrium incentives; a cooperative trace alone does not.

| False-alarm $\alpha$ | Detection $\beta$ | Minimum discount |
|---|---:|---:|
| 0 | 1 | 1/2 |
| 1/10 | 1 | 5/9 |
| 1/10 | 2/5 | 5/6 |

```python
from fractions import Fraction as F
for alpha,beta,threshold in [(F(0),F(1),F(1,2)),(F(1,10),F(1),F(5,9)),(F(1,10),F(2,5),F(5,6))]:
    assert threshold==1/(1+beta-2*alpha)
    for discount in [threshold/2,threshold,(1+threshold)/2]:
        punishment=1/(1-discount)
        cooperation=punishment+2/(1-discount*(1-alpha))
        deviation=5+discount*((1-beta)*cooperation+beta*punishment)
        assert cooperation==3+discount*((1-alpha)*cooperation+alpha*punishment)
        assert (cooperation>=deviation)==(discount>=threshold)
    print(alpha,beta,threshold)
assert 1+F(3,10)-2*F(1,5)<1
```

If $\beta\le2\alpha$, no discount below one satisfies this construction's condition. This is failure of this particular trigger scheme, not a theorem that every form of cooperation is impossible. Imperfect monitoring changes the information model, so the native perfect-monitoring calculation and finite traces remain separate.

## Exercises with explained solutions

Exercise1: compute all four rounds of the reactive example, then calculate both fixed replay rewards and both regenerated constant-policy rewards.

Solution: actual pairs are(0,0),(1,0),(0,1),(1,0), with utility−2. Replay rewards are(2,−2), rerun rewards(4,2), external regret4 and rerun-policy regret6. The learner's modified history is exactly what changes the copying opponent.

Exercise2: does zero positive-part external regret imply repeated-game equilibrium, or optimality against the dynamic oracle?

Solution: no. It only says the actual reward is at least the best constant-action replay reward for this prefix. Equilibrium requires incentive conditions over strategies, and the per-round oracle is a different comparator. The four rules here have no asymptotic bound attached.

Exercise3: for three cooperative rounds compare discounts0,0.8 and1. What changes when the prefix is empty?

Solution: utilities are3,7.32 and9, with discount weight sums1,2.44 and3. Empty-prefix totals are0, but selected round/action, average regret and discounted average are unavailable. Zero observations should not be presented as a successful learning result.
