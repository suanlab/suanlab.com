## Cooperation can still require coordination

Two maintenance agents share a system score. If the system is healthy, they collect useful output; if damaged, they can attempt repair. They receive the same reward, yet selecting compatible actions can remain difficult. Having a shared objective does not imply that every policy immune to unilateral changes is jointly optimal.

This unit studies a finite-state Markov game with an infinite discounted horizon. “Finite” refers to two states and two actions per player, not a truncated episode. Both players observe the current state, choose actions simultaneously and receive a common bounded reward. The next state depends on the current state and both actions.

This differs from the preceding transferable-utility coalition game. Here a common score is already specified for both players; the task is selecting policies that influence dynamics. No coalition value or payment division is inferred. Values below use that common score per player, not the sum of two identical copies of it.

## State, joint action and transition are separate objects

Let s belong to{0,1}, and let a,b be the players' actions. A reward r(s,a,b) is immediate. A transition P(s'|s,a,b) is a probability distribution over the next state. It is not a probability that an action is chosen.

[MIT's Markov-game notes, §S6.1–S6.2](https://www.mit.edu/~6.7980/stochastic_games.html) distinguish state dynamics, player reward functions, history-dependent policies and stationary Markov policies. We use the stationary fully observed special case: in state s, P1 selects action0 with probability p_s and P2 with probability q_s. Their mixing is independent, so joint masses are

$$
(p_sq_s,\ p_s(1-q_s),\ (1-p_s)q_s,\ (1-p_s)(1-q_s)),
$$

ordered00,01,10,11. The controls specify these probabilities separately at each state. A policy maps states to distributions; it is not one fixed action reused without looking at state.

## Average the model under the policies

For fixed policies, average over joint actions:

$$
r_\pi(s)=\sum_{a,b}p_s(a)q_s(b)r(s,a,b),
\qquad
P_\pi(s,s')=\sum_{a,b}p_s(a)q_s(b)P(s'|s,a,b).
$$

Now the state sequence is a Markov chain. The first figure's arrows show this averaged transition law. Its action table instead shows conditional model entries at one queried state, alongside the current action masses. These quantities should not be collapsed into one probability.

For example, if all four actions have mass1/4 in the repair model, immediate rewards average to(9/4,1/4). Probabilities of next state1 average to(9/32,5/8). In state0, the conditional next-state1 probability of action00 remains1/2, different from the averaged9/32.

## Infinite discounted value is an expectation

For discount0≤gamma<1, define the common value from each starting state by

$$
V^\pi(s)=\mathbb E_\pi\!\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t,b_t)\mid s_0=s\right].
$$

The expectation includes action mixing and state transitions. It is not a sampled trajectory or an immediate reward. We use the unnormalized discounted sum; no factor1−gamma is applied to it.

Conditioning on the first transition yields

$$
V^\pi=r_\pi+\gamma P_\pi V^\pi,
\qquad (I-\gamma P_\pi)V^\pi=r_\pi.
$$

The [MIT Bellman reference](https://gradml.mit.edu/reinforcement/value_bellman/) develops policy evaluation and discounted optimality. Here P_pi is stochastic, so the sup-norm map V↦r_pi+gamma P_pi V is a contraction for gamma<1. Its unique fixed point can also be written as a convergent matrix series. With bounded reward magnitude M, the value magnitude is at most M/(1−gamma).

Gamma0 keeps only the first reward. Gamma1 is excluded from this infinite-horizon lab: a finite episode could permit1, but a continuing positive reward stream would generally diverge. The slider stops at0.95 to preserve a finite model and readable scales.

## A once-forced action has a different Q value

For a joint action that is forced once and followed by the current policies afterward,

$$
Q^\pi(s,a,b)=r(s,a,b)+\gamma\sum_{s'}P(s'|s,a,b)V^\pi(s').
$$

This is not the value of replacing the entire future policy. An action with zero current mass can still have a defined Q because its reward and transition are specified in the model. Zero visitation under one policy is different from absence of model knowledge.

If those quantities were unknown in an actual learning system, estimating an unvisited action's Q would require further evidence or assumptions. The lab explicitly provides a complete small model; it does not claim that observational data alone identify every counterfactual.

## A unilateral best response is an MDP

Hold the opponent's stationary policy fixed. Averaging over that opponent's action gives one controlled MDP for the player considering a deviation. That player can change its action at both states and thereby change all future visits.

A finite discounted fully observed MDP has a deterministic stationary optimal policy. With two states and two actions there are four such own policies. The lab evaluates all four, finds their componentwise optimal value and retains every exact tie. Although the enumeration is small, its justification comes from discounted MDP optimality, not a claim that arbitrary history-dependent strategies were directly listed.

Let V_BR_i be player i's best-response value. The gain is V_BR_i−V_pi at each starting state. Our all-state stationary Nash status requires both players' gain vectors to be exactly zero. This is stronger than checking an initial distribution that never visits one of the states. Under the stated Markov, full-observation and stationary-opponent assumptions, an optimal MDP response also bounds history-dependent deviations.

## A centralized team benchmark is another MDP

A coordinator choosing the joint action has four choices at each state. Its16 deterministic stationary policies form a team benchmark. The same discounted-MDP argument guarantees an optimal policy for all states simultaneously. All exact optimal ties are retained, including cases where either player can perform the useful repair.

The benchmark's value V_team and gap V_team−V_pi use the common score. A team optimum is also immune to unilateral improvements, since a unilateral policy is among possible joint policies. The reverse implication can fail: coordinating both agents may improve outcomes even when neither can improve alone.

The interface evaluates the entered policies and benchmarks them. It is not an algorithm that learns a policy from samples or proves convergence of independent reinforcement learners.

## Worked example: a Nash coordination trap

In the state-preserving family, the system stays in its starting state regardless of action. Common reward matrices are

$$
R_0=\begin{pmatrix}4&0\\0&3\end{pmatrix},
\qquad
R_1=\begin{pmatrix}2&0\\0&1\end{pmatrix}.
$$

Let both players always choose action1 and gamma=0.8. Values are3/(1−0.8)=15 from state0 and1/(1−0.8)=5 from state1. A unilateral change to action0 earns0 instead, so both best-response gains are zero.

Jointly choosing00 at both states yields values20 and10. The current policy is all-state Nash but its team gaps are5 at both states. Shared rewards have not eliminated the coordination problem.

At state0, forcing00 only once gives Q=4+0.8·15=16. That differs from current value15 and from the policy replacement value20. It is a joint one-step deviation, so its improvement does not contradict the absence of a profitable unilateral deviation.

## Worked example: repair changes future opportunity

The second family uses state0 for healthy and state1 for damaged. Action1 is a costly repair/maintenance action. In joint order00,01,10,11, common rewards are

$$
r_0=(4,2,2,1),\qquad r_1=(0,1,1,-1).
$$

Probabilities of next state1 are

$$
P(1\mid0,\cdot)=(1/2,1/4,1/4,1/8),
\qquad P(1\mid1,\cdot)=(1,3/4,3/4,0).
$$

These are custom teaching parameters, not estimated engineering failure rates. Joint repair at a damaged state gives immediate reward−1 but restores the healthy state with probability1. A single repair gives reward1 but succeeds with probability1/4.

Consider00 when healthy and11 when damaged at gamma0.8. Bellman equations become

$$
V_0=4+0.8(\tfrac12V_0+\tfrac12V_1),
\qquad V_1=-1+0.8V_0.
$$

Solving gives V0=90/7 and V1=65/7. The enumeration certifies this as team optimal. In the damaged state, choosing only one repair once and then resuming these policies gives Q=64/7, slightly below65/7. Comparing immediate rewards1 and−1 alone would select the wrong joint action for this continuation value.

## Discounting can reverse the decision

Keep exactly those policies but set gamma0.2. Their values become(195/44,−5/44). The team-optimal values are instead(175/38,55/38), achieved by two tied policies: both choose00 when healthy, and either agent alone repairs when damaged.

Each player's best-response gain vector is(145/836,1305/836), so the original joint-repair policy is no longer Nash or team optimal. A smaller weight on future healthy operation changes the incentive to incur an immediate cost.

The plotted curves vary discount while holding the entered policies fixed. They are not envelopes of the optimal policies at each discount. The exact policy table concerns the current discount only; changing gamma can change which rows are optimal.

## Why exact comparisons matter

All inputs are bounded integer percentages and model transition entries are rational eighths. The implementation carries rational numerators and denominators through the two-state Bellman solve, candidate comparisons, gaps and Q values. Bellman residuals are exactly zero in that arithmetic. Only display and plotting convert to floating point.

Selecting the policy with largest sum of state values is accompanied by an exact check that it dominates every candidate at each state. This is justified by the existence of a componentwise MDP optimum. Tie detection uses the full exact value vector rather than a rounding threshold or only one initial state.

Tests separately recompute values from the equation for V0−V1, inspect Bellman optimality equations, and compare a long discounted rollout with a geometric tail bound. These checks address the evaluation and benchmark claims, not the convergence of a learning algorithm that was never run.

## Limits and research questions

Partial observation changes which policies can depend on the true state. Correlated joint actions differ from the independent mixed profile entered here. Unknown transition/reward models raise exploration and identification problems. General-sum games give different players different objectives and require equilibrium reasoning beyond maximizing one common score.

A small exact model is useful for checking an implementation, formulating a coordination trap and separating immediate from continuation incentives. It does not by itself establish that a multi-agent training method reaches Nash, reaches the team optimum or transfers to a larger physical system.

## Exercises and explained solutions

Exercise1: compute current value, once-forced Q00 and team-optimal value at state0 in the coordination trap. Why are the three numbers different?

Solution:15 follows action11 forever;16 forces00 once then resumes11;20 follows00 forever. The first change is joint and temporary, while the team benchmark changes both future policies.

Exercise2: at gamma0, what do policy value and Q reduce to? Does the dynamic model necessarily have a unique optimal joint policy?

Solution: V_pi equals the policy-averaged immediate reward and Q equals the conditional immediate reward. In the damaged state either01 or10 earns1, so two optimal joint policies remain tied when healthy-state action00 is used.

Exercise3: why do four own policies and16 joint policies suffice here, and what would fail under hidden state?

Solution: finite discounted full-observation MDP optimality supplies deterministic stationary optima for a fixed opponent and for a joint coordinator. Hidden state may require policies based on beliefs or histories; the current state-indexed enumeration would no longer describe every admissible information structure.
