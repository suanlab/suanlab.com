## A cheap action can be expensive in the long run

Imagine choosing between a cheap attempt that may leave a task unfinished and a reliable action that immediately advances it. The right choice depends on more than the next invoice: repeated unsuccessful attempts also cost something. Dynamic programming makes that future consequence explicit without enumerating every possible history.

The key modeling step is to choose a sufficient state. In a Markov decision process, the current state and action determine the conditional distribution of the next state and the immediate cost. If relevant information is hidden in history, the stated model is incomplete; applying a Bellman update does not repair that omission. Belief states for partial observation are introduced later.

## Read the entire finite model

This lab has decision states S0,S1,S2 and an absorbing terminal state S3. At a decision state s, action 0 costs c0(s), stays at s with probability p, and advances to s+1 with probability 1-p. Its costs at the three states are 1,2,4. Action 1 costs the selected amount c and advances surely. At S3 every future cost and value is zero.

These transition probabilities are known, not estimated from samples. The diagram shows only the selected decision state and its possible outcomes. An edge labeled zero still denotes a zero-probability outcome. The thick action border shows the current greedy choice; it does not indicate that a sampled transition actually occurred.

For a policy, minimize expected discounted cost

$$
V^\pi(s)=\mathbb E_\pi\left[\sum_{t=0}^{\infty}\gamma^t c(S_t,A_t)\mid S_0=s\right],\qquad 0\leq\gamma<1.
$$

Costs are paid before transitioning. A policy selects actions using the state; it is not just one fixed action sequence. Discounting makes bounded costs summable even when an action keeps returning forever. It also changes the objective: a discounted optimum need not minimize undiscounted cost or guarantee reaching the terminal state.

## Conditional expectation produces the Bellman backup

For a proposed future-value vector V, compute both action values:

$$
Q_V(s,0)=c_0(s)+\gamma[pV(s)+(1-p)V(s+1)],
$$

$$
Q_V(s,1)=c+\gamma V(s+1),\qquad (TV)(s)=\min_a Q_V(s,a).
$$

The expectation averages next-state **values**, rather than choosing the most likely next state. The minimization selects an action before the random outcome is observed. Interchanging those operations would give the decision maker information it does not possess.

The principle of optimality says that after a transition, an optimal continuation can be chosen from the resulting sufficient state. Additive cost and the Markov state allow that recursive description. In this finite discounted model an optimal stationary deterministic policy exists, so evaluating all such policies provides a second route to the answer.

## Synchronous value iteration has an explicit clock

Initialize V0 to the selected level on S0,S1,S2 and to zero on S3. Apply V_(k+1)=T(V_k) simultaneously: every new component reads the same old vector. Updating one state in place and then using it for the next state would instead be a different update schedule.

The blue plot is the actual V_N after N backups. At N=0 it remains the supplied initial guess. With V0=0, V_N is the optimum for N stages with zero terminal payoff. With a general V0, the finite-horizon interpretation includes discounted terminal payoff gamma^N*V0(S_N). Neither is automatically the infinite-horizon optimum V*.

With nonnegative costs and V0=0, the iterates rise toward V*. An arbitrary initial guess need not move monotonically. For the second example started instead at level 8, V1(S0)=8.2 initially rises even though the eventual optimum is about 7.023077. Convergence is a contraction statement, not a promise of improvement in one direction at every state.

The action readout is greedy(V_N), obtained from Q_(V_N). It therefore looks one stage farther ahead than V_N. Fixing this greedy rule and following it forever is yet another object: its stationary policy value need not equal V_N. The lab reports that value separately, rather than relabeling an iterate as a policy cost.

## Independently evaluate eight stationary policies

There are two choices at each of three decision states, hence eight deterministic stationary policies. For a fixed policy the Bellman equation is linear. At each state,

$$
V^\pi(s)=
\begin{cases}
\dfrac{c_0(s)+\gamma(1-p)V^\pi(s+1)}{1-\gamma p},&\pi(s)=0,\\
c+\gamma V^\pi(s+1),&\pi(s)=1.
\end{cases}
$$

Start from V(S3)=0 and substitute backward. The denominator is positive on the stated domain. The orange V* is the statewise minimum of the eight evaluated vectors. For a finite discounted MDP, an optimal stationary policy attains these minima together; the lab also checks that a greedy policy for this vector reproduces its value.

This comparison does not use a large value-iteration count as its supposed ground truth. Tests separately solve the full policy linear systems and sum discounted transition distributions to verify the policy evaluator. Exact floating-point Q ties choose action 0; values extremely close to a tie should not be interpreted as evidence of a practically meaningful action advantage.

## Why discounting certifies value error

For probability rows that sum to one, averaging cannot enlarge the maximum component difference. Taking a minimum is also Lipschitz in the maximum action-value difference. Consequently,

$$
\|TV-TW\|_\infty\leq\gamma\|V-W\|_\infty.
$$

This contraction gives a unique fixed point. Define the Bellman residual r=||TV-V||inf. Add and subtract TV, then use contraction against V*:

$$
\|V-V^*\|_\infty\leq r+\gamma\|V-V^*\|_\infty
\quad\Longrightarrow\quad
\|V-V^*\|_\infty\leq\frac{r}{1-\gamma}.
$$

The lab displays actual maximum value error and this residual certificate separately. It can be conservative, especially near gamma=1. It bounds the error of the supplied **value vector**, not directly the loss of its greedy policy. The maximum greedy stationary policy loss is independently computed from that policy's exact value minus V*. Gamma=1 is excluded because this generic contraction argument and denominator no longer apply.

## Worked exercise and solution

Set gamma=0.5,p=0.5,c=3 and V0=0. The first backup gives V1=(1,2,3,0): at S2 the reliable action costs 3 instead of 4. To compute V2(S1), compare action 0's 2+0.5*(0.5*2+0.5*3)=3.25 with action 1's 3+0.5*3=4.5. Thus V2(S1)=3.25. Applying the same old vector to every state gives V2=(1.75,3.25,3,0).

**Solution check.** The infinite discounted optimum is instead V*(S2)=3, V*(S1)=(2+0.25*3)/0.75=11/3, and V*(S0)=(1+0.25*(11/3))/0.75=23/9. Two backups have not reached those values, even though their greedy stationary policy already happens to be optimal. This is a concrete example of value error and policy loss being different quantities.

Now use gamma=0.9,p=0.75,c=3. At S1, the reliable action has optimal cost 3+0.9*3=5.7. Always choosing action 0 there instead costs (2+0.9*0.25*3)/(1-0.9*0.75), about 8.230769. Immediate costs favor action 0, but future repeated returns favor action 1. Follow the iteration table to see when the greedy choice changes.

## Connection to AI learning and planning

This is an exact known-model planner, not temporal-difference learning or Q-learning from sampled transitions. Those methods approximate related conditional expectations using data and add questions about exploration, estimation and optimization. A Bellman residual measured only on a sampled subset does not automatically certify the global error shown here. State representation, objective, transition support and the domain of any error guarantee must stay explicit when moving from this finite reference to learned systems.

## Primary reference

Russ Tedrake, [Dynamic Programming](https://underactuated.mit.edu/dp.html), particularly the discounted-cost and finite-MDP sections, provides the broader planning framework. The finite chain, eight-policy oracle, examples, contraction derivation and diagrams here are original teaching constructions.
