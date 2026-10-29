## Distinguish outcomes, events and probabilities

An outcome is one possible result of a trial. The sample space contains all outcomes. An event is a subset, while probability is a function assigning numbers to events. The lab uses Ω={0,1,2,3}. Mask three selects bits zero and one, giving A={0,1}; mask six gives B={1,2}. A bitmask selects a set and is not itself a probability.

$$
(\Omega,\mathcal F,P),\qquad \mathcal F=2^\Omega\quad\text{in this finite example}.
$$

## What the axioms require

Probability is nonnegative, assigns one to the whole space and is countably additive on disjoint events. On general spaces, the event collection must be a sigma-algebra closed under complements and countable unions. Every subset is measurable in this finite example, so those closure conditions hold automatically. This finite table does not settle measurability questions on arbitrary infinite spaces.

$$
P(\Omega)=1,\quad P(A)\ge0,\quad P\!\left(\bigcup_i A_i\right)=\sum_iP(A_i)\quad(A_i\text{ disjoint}).
$$

## Normalize weights into probabilities

The default weights (1,2,3,4) sum to ten, giving masses (0.1,0.2,0.3,0.4). Sum masses over an event: P(A)=0.3 and P(B)=0.5. Their intersection {1} has mass 0.2. Adding P(A) and P(B) counts outcome one twice, so subtract its contribution once. Follow the membership columns to verify the calculation.

$$
P(A)=\sum_{\omega\in A}p_\omega,\quad P(A\cup B)=P(A)+P(B)-P(A\cap B)=0.6.
$$

## Zero probability versus an empty event

Weight mode two assigns zero mass to outcomes two and three without deleting them from the sample space. Thus {2} is nonempty but has probability zero. The empty event always has probability zero. The distinction disappears in finite models with strictly positive point masses, but becomes fundamental for continuous variables. A model assigning zero mass to an outcome is different from a logical claim that the outcome cannot occur.

$$
P(A^c)=1-P(A),\qquad P(\varnothing)=0.
$$

## Exercises with explanations

1. What is the default complement probability? **0.7.**
2. What is the union mass of {0,1} and {2,3} in uniform mode? **One.**
3. Must every point have positive mass? **No: the axioms permit zero.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
