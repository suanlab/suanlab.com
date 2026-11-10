## Acting with a state versus acting with a belief

A fully observed Markov decision process supplies the current state before an action is chosen. Its model specifies states, actions, transition probabilities and costs. A policy can therefore compare actions using the known state and a remaining horizon. The previous Bellman chapter computed this comparison exactly in a finite MDP.

In a partially observable MDP, observations do not reveal the state directly. The controller needs a distribution over possible states, called a belief. A belief is not merely the last observation, a confidence score without a model, or the unknown physical state itself. Under a correct known transition and observation model it summarizes the relevant information in the action–observation history. For a finite-horizon problem, the remaining decision budget must also be supplied to the policy.

## Specify the hidden-state sensing problem

This lab uses a hidden binary variable Z that remains fixed until a terminal guess. The prior belief is b=P(Z=1). Guessing 0 incurs loss one if Z=1, so its expected loss is b. Guessing 1 has expected loss 1-b. Either guess terminates the task; all later costs are zero.

Listening costs c and produces a noisy binary observation Y, with accuracy eta=P(Y=Z). Conditional on Z, repeated sensor noises are independent. Listening changes the information available, not Z itself. At most H further listens are permitted; guessing is always available. H=0 means guess now, rather than avoid making a terminal decision. No discount is used because this is a bounded sensing problem.

For comparison, a fully observed controller that knows Z guesses correctly at zero cost. This is an information advantage, not an algorithm that the partially observed controller can access. The POMDP policy never receives the hidden answer.

## Predict, observe and condition

For a general POMDP, first predict the next-state distribution through the transition model T. Then condition it on the observation model O:

$$
b'(s')=\frac{O(y\mid s',a)\sum_s T(s'\mid s,a)b(s)}
{\sum_j O(y\mid j,a)\sum_s T(j\mid s,a)b(s)}.
$$

The denominator is the observation probability. A posterior requires that probability to be positive. In this lab a listen has the identity transition for Z, giving

$$
m_1=\eta b+(1-\eta)(1-b),\qquad b_1=\frac{\eta b}{m_1},
$$

$$
m_0=(1-\eta)b+\eta(1-b),\qquad b_0=\frac{(1-\eta)b}{m_0}.
$$

If a selected observation has zero mass, its posterior is undefined. The readout shows an unavailable value and the native diagram shows an empty dashed posterior frame. No epsilon creates an artificial posterior. In expectations, that impossible branch contributes zero and is skipped.

Likelihood scaling is used to avoid avoidable underflow in posterior ratios. Constant-preserving expectation calculations also prevent two half-weighted identical tiny risks from vanishing unnecessarily. These are numerical evaluations of the same model, not probability smoothing or arbitrary extra observations.

## Turn information gathering into a Bellman decision

The immediate-guess risk is V0(b)=min(b,1-b). With H positive, compare the two stopping actions with a listen and its contingent continuation:

$$
V_H(b)=\min\left\{b,1-b,
 c+\sum_{y:m_y>0}m_y V_{H-1}(b_y)\right\}.
$$

A policy chooses actions from (H,b), then chooses its next action after seeing the observation. It is not an instruction to listen H times regardless of results. The current first-action code is 0 for guess0, 1 for guess1 and 2 for listen. Exact floating-point ties prefer those actions in that order. A choice extremely close to a tie should not be interpreted as a large practical advantage.

Because stopping is always available, planning cannot cost more than immediate guessing under this model. Increasing the maximum number of listens also cannot increase the optimal cost: the policy can ignore the extra opportunity. This does not mean extra **forced** observations are free or useful.

## Read the visualization without inventing a sampled history

The native bars show the prior and the posterior for the selected hypothetical observation. Blue mass means Z=1 and orange means Z=0. The selected Y is a forced one-observation calculation, even when H=0 or the optimal policy would stop. It does not claim that this observation was actually acquired by the policy.

The lower plot uses belief, not time, on its horizontal axis. Blue is the optimal expected cost for the selected H; orange is immediate-guess error risk. The dot marks the chosen prior. Curves use 201 sampled beliefs joined by segments, so small policy-boundary kinks may lie between plotted samples; selected-belief calculations use the exact finite recursion. The table compares horizons at that same prior.

Expected listen count and terminal error probability are separate readouts. Their combination checks the objective:

$$
V_H(b)=c\,\mathbb E[N_{listen}]+\Pr(\text{terminal guess is wrong})
$$

for the selected optimal policy. A lower total cost can involve more observations and less error; it need not minimize either component independently.

## Why conditional plans produce a concave cost value

Fix a complete conditional plan in advance, including what to do after each observation. Let alpha0 and alpha1 be its expected total costs conditional on the two hidden states. Its prior-averaged cost is affine:

$$
C_\alpha(b)=(1-b)\alpha_0+b\alpha_1.
$$

The stopping plans have vectors (0,1) and (1,0). For a listen with continuation vectors a after Y=0 and d after Y=1, the new vector is

$$
\alpha_0=c+\eta a_0+(1-\eta)d_0,\qquad
\alpha_1=c+(1-\eta)a_1+\eta d_1.
$$

Taking the minimum over the finite set of plans gives a piecewise-linear, **concave lower envelope**. Cost minimization is essential to that sign. Reward maximization instead uses a convex upper envelope; copying its geometric statement here would be incorrect.

For H=0,1,2,3 the unpruned plan counts are 2,6,38,1446, from M_H=2+M_(H-1)². Tests compare this independently enumerated collection with the production Bayesian observation-tree recursion. The small example is exactly enumerable; general POMDPs require much more careful representations and approximations.

## Worked exercise: posterior and one-listen solution

Set b=0.5,eta=0.8,c=0.05,H=1. Before reading the solution, compute the observation probabilities, posterior beliefs and optimal choice between guessing and one listen.

**Solution.** Both observations have probability 0.5. Given Y=1, the posterior is 0.8; given Y=0 it is 0.2. Guessing according to either posterior makes an error with probability 0.2. A listen therefore costs 0.05+0.2=0.25, better than immediate guessing at cost 0.5. The first action is listen.

Now use the same parameters with H=3. A useful optimal plan listens twice, stops if the two results agree, and otherwise listens a third time before guessing. Two observations disagree with probability 2*(0.8)*(0.2)=0.32, so expected count is 2.32. Terminal error is 0.104, giving total cost 0.05*2.32+0.104=0.22.

This also exposes a planning effect: at belief 0.8, one extra listen does not reduce error enough to justify its cost, so V1(0.8)=0.2. Yet two extra listening opportunities give V2(0.8)=0.17. Contingent later information can make a current observation worthwhile even when a one-step information rule would stop.

## Uninformative, perfect and impossible observations

If eta=0.5, either observation leaves the prior unchanged. Paying a positive cost to listen cannot help, regardless of H. If eta=1 and the prior gives both states positive mass, one observation identifies Z perfectly; whether it is worth acquiring still depends on c versus current error risk.

For b=0,eta=1,Y=1, the observation is impossible and has no posterior. The controller already knows Z=0 and can stop at zero cost. An unavailable conditional calculation is not a failure of the value function. These boundary cases distinguish state knowledge, sensor quality, observation support and the value of information.

## Connection to AI research

Belief models connect filtering with decision making: an observation can matter because it changes a future action, not merely because it reduces entropy. Learned world models, active perception and partially observed agents add questions about model error, history representation and approximation. Exact belief sufficiency assumes the stated model; a learned recurrent embedding is not automatically a calibrated Bayesian belief. The current finite reference separates those issues from the arithmetic of a small known POMDP.

## Primary reference

Kaelbling, Littman and Cassandra, [Planning and acting in partially observable stochastic domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf), Artificial Intelligence 101 (1998), supplies the broader belief-MDP and conditional-plan framework. This finite zero/one-loss sensing model, cost-minimization derivation, numerical examples, diagrams and independent checks were constructed for this textbook.
