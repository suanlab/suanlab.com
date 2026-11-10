## A constant level can hide a circulating flow

Imagine three water tanks connected in a triangle. Every tank can keep the same water level while water circulates around the triangle. Equal inflow and outflow at each tank does not require equal flow in both directions along each pipe. Probability has an analogous distinction: a distribution can stay constant while its stationary transitions carry a directed circulation.

The distinction matters when an AI algorithm is supposed to sample a target distribution. Preserving that target is one question. Respecting detailed balance is another. Reaching the target from a chosen initialization is a third. We will calculate all three questions in a complete three-state example, rather than treating an attractive moving-arrow picture as proof of equilibrium.

You need finite conditional probabilities, matrix multiplication and expectation. [Levin and Peres, sections 1.5–1.6 and 4.3](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf) provide the general definitions of stationarity, reversibility and finite-chain convergence. The integer preparations, computations and elementary derivations below specify this chapter's own finite model.

## Specify the preparation before drawing arrows

There are three named states, 0, 1 and 2. Clockwise means 0→1→2→0. Counterclockwise means the opposite cycle. At each discrete step, the same transition rule applies regardless of the earlier history. The model is therefore a time-homogeneous Markov chain.

Choose nonnegative integer weights f, b and s for clockwise movement, counterclockwise movement and holding. Their sum must be positive. Normalizing weights gives probabilities:

$$
D=f+b+s>0,\qquad p=\frac fD,\quad q=\frac bD,\quad r=\frac sD,\qquad p+q+r=1.
$$

The admitted weights are integers from 0 to 1,000,000. Multiplying all three by the same positive integer preserves the transition probabilities when the scaled inputs remain admitted. It does not make the chain move faster: one transition is still one discrete step. This is different from scaling the rates of a continuous-time process.

Three initial weights specify the initial probabilities after a separate normalization. They need not be equal or strictly positive, but their sum must be positive. An initial weight of zero excludes that state only at initialization; it does not delete the state or forbid arriving there later. The experiment examines 0 through 512 transitions. That is a computational range, not a general mixing-time guarantee.

| Symbol | Meaning | Unit or convention |
| --- | --- | --- |
| f, b, s | Unnormalized transition weights | Exact nonnegative integers |
| p, q, r | Clockwise, counterclockwise, holding probabilities | Dimensionless |
| μ(t) | Distribution after t transitions | Row vector, total one |
| π | A specified stationary reference | Here the uniform distribution |
| t | Number of transitions | Discrete count, not physical seconds |

## Fix the row and column convention

A matrix entry Pij is the probability of arriving at j when the current state is i. Rows describe departures. With our named orientation, the transition matrix is

$$
P=\begin{pmatrix}r&p&q\\q&r&p\\p&q&r\end{pmatrix},\qquad P_{ij}=\Pr(X_{t+1}=j\mid X_t=i).
$$

For example, P01=p and P10=q. Reversing those two entries would reverse the meaning of the clockwise control. Each row sums to one because the three mutually exclusive choices exhaust the next-step possibilities. A diagonal entry is a valid holding probability, not a missing transition.

A useful conceptual picture is `0 → 1 → 2 → 0`, with weight p on each displayed arrow, weight q on each reverse arrow and a holding loop r at each vertex. The three clockwise arrows are different transitions between named states. They are not three copies of one state whose probabilities should be merged.

## Evolve a distribution, not a sampled trajectory

A distribution accounts for all possible next states at once. Under the row-vector convention,

$$
\mu(t+1)=\mu(t)P,\qquad \mu_j(t+1)=r\mu_j(t)+p\mu_{j-1}(t)+q\mu_{j+1}(t),
$$

where state indices are taken modulo three. The second expression is an inflow ledger: probability stays at j, arrives clockwise from j−1, or arrives counterclockwise from j+1. It is also a way to check the orientation of the matrix formula.

Take f=2,b=1,s=1, so p=1/2 and q=r=1/4, and initialize entirely at state 0. Successive distributions are:

| t | Probability of 0 | Probability of 1 | Probability of 2 |
| --- | --- | --- | --- |
| 0 | 1 | 0 | 0 |
| 1 | 1/4 | 1/2 | 1/4 |
| 2 | 5/16 | 5/16 | 6/16 |
| 3 | 22/64 | 21/64 | 21/64 |

The largest component moves between states as the preparation relaxes. Each row sums to one. These rows are exact probability distributions, not frequencies estimated from four or sixteen random trials. A single sampled trajectory would occupy just one state at each time; many independently initialized trajectories would be needed to estimate these probabilities empirically.

## Derive the uniform stationary reference

A stationary distribution satisfies πP=π. Every column of this particular matrix also sums to one, so the uniform row vector is stationary:

$$
\pi=\left(\frac13,\frac13,\frac13\right),\qquad (\pi P)_j=\frac13\sum_iP_{ij}=\frac13.
$$

This calculation proves invariance, including when p and q differ. It says nothing yet about whether this reference is unique or whether an arbitrary initial distribution reaches it.

Uniqueness can be proved directly for this model. A stationary component xj must satisfy

$$
(p+q)x_j=px_{j-1}+qx_{j+1}.
$$

If p+q>0, choose a component with maximal value. The equality expresses that maximum as a convex average of neighbors with nonzero movement weights. Every contributing neighbor must have the same maximum. Following either available movement direction around the three-state cycle reaches every state, so all components are equal. Normalization then gives 1/3 at each state.

If p=q=0, division by p+q would be invalid. In that case r=1 and P is the identity: every initial distribution is stationary. Uniform π remains a useful reference, but selecting it does not establish a unique equilibrium.

## Keep one-way flux and net current separate

In the stationary preparation, define Fij as the joint probability of being at i and then moving to j. Define Jij as the difference between the two directional fluxes:

$$
F_{ij}=\pi_iP_{ij},\qquad J_{ij}=F_{ij}-F_{ji},\qquad J_{ji}=-J_{ij},\quad J_{ii}=0.
$$

For the biased example, the clockwise flux on 0→1 is 1/6 and the reverse flux is 1/12. Their net current is 1/12. Calling the larger number itself “the current” would lose the reverse traffic. Holding has flux r/3=1/12, but its antisymmetric current is zero.

The full net-current matrix is

$$
J=\frac{p-q}{3}\begin{pmatrix}0&1&-1\\-1&0&1\\1&-1&0\end{pmatrix}.
$$

At every vertex, the total off-diagonal outgoing flux is (p+q)/3 and the incoming flux is the same. In our example both are 1/4. The state probability is therefore constant even though each clockwise edge has a nonzero net current. This is the probability version of the circulating-water example.

## Stationarity is a vertex condition; detailed balance is a pair condition

Summing the signed current leaving a vertex gives a local balance equation:

$$
\sum_jJ_{ij}=\pi_i\sum_jP_{ij}-\sum_j\pi_jP_{ji}=\pi_i-(\pi P)_i=0.
$$

This zero divergence allows circulation around a loop. It does not require each summand to vanish. Detailed balance imposes the stronger condition that each state pair has equal directional stationary flux:

$$
\pi_iP_{ij}=\pi_jP_{ji}\ \text{for every }i,j\qquad\Longleftrightarrow\qquad p=q\ \text{in this model}.
$$

Summing detailed balance over partners proves stationarity. The reverse implication fails in the biased three-state example, where every vertex balances but each clockwise current is positive. This is a concrete counterexample, not a disagreement between two numerical tolerances.

At p=q, transitions still occur. Detailed balance does not mean all off-diagonal probabilities vanish or every path stands still. It means the stationary traffic is statistically balanced between each named pair. A chain can have large bidirectional traffic and zero net current.

## Reverse a stationary chain carefully

The statistical time reversal of a chain prepared at its positive stationary reference has transition probabilities

$$
P^{\mathrm{rev}}_{ij}=\frac{\pi_jP_{ji}}{\pi_i}=P_{ji}\qquad\text{for uniform }\pi.
$$

Here reversal simply swaps p and q. It preserves the same uniform stationary distribution and reverses every net current. The original and reversed transition matrices agree exactly when p=q. This concerns the law of stationary paths, not the reversal of velocities in a physical system whose momentum variables have not been included.

A short path makes the difference visible. Conditional on starting at state 0, the three-step clockwise and counterclockwise circuits have probabilities

$$
\Pr(0\to1\to2\to0\mid X_0=0)=p^3,\qquad \Pr(0\to2\to1\to0\mid X_0=0)=q^3.
$$

For p=1/2,q=1/4 these are 1/8 and 1/64. Under stationary initialization both path probabilities acquire the same factor 1/3. Their unequal values expose a preferred orientation even though every single-time marginal is uniform. Single-time histograms can therefore miss a difference in temporal organization.

## Separate reference flows from current preparation flows

A nonstationary initial distribution has its own one-step fluxes. At a selected time t, they are

$$
F_{ij}(t)=\mu_i(t)P_{ij},\qquad \sum_j\bigl(F_{ij}(t)-F_{ji}(t)\bigr)=\mu_i(t)-\mu_i(t+1).
$$

For the biased example at t=0, the net outgoing amounts are 3/4,−1/2,−1/4 at states 0,1,2. State 0 loses probability; the others gain it. Negative net outgoing flow means net incoming flow, not a negative transition probability.

By contrast, the stationary-reference current has zero divergence at every vertex from the start. Both ledgers are valid, but they answer different preparation questions. A picture must specify which distribution weights its arrows. Substituting uniform fluxes into a transient distribution plot would incorrectly show balanced levels while the displayed bars are still changing.

## A unique stationary distribution need not attract every initialization

There are three different return structures when movement is possible:

$$
\text{return period}=\begin{cases}1,&r>0,\\1,&r=0,\ p>0,\ q>0,\\3,&r=0\text{ and exactly one of }p,q\text{ is positive}.\end{cases}
$$

A holding loop permits a return in one step. If both directions are available without holding, returns in two and three steps are possible, so their greatest common divisor is one. A deterministic directed cycle permits returns only at multiples of three. The period belongs to the communication structure, not to the number of frames used to draw an animation.

For the finite irreducible cases, period one gives convergence to the unique stationary distribution from every initial distribution. The directed no-hold case has a unique uniform stationary distribution but period three. Starting at state 0 produces 0,1,2,0 repeatedly; the distribution does not converge. Starting uniformly is stationary even in that periodic case.

The identity case is reducible. Each singleton class has period one, yet a nonuniform preparation stays nonuniform forever. Period one alone is therefore insufficient when the required communication assumption is missing. Distinguish a statement about convergence to some limit from convergence to the specified uniform reference.

## Measure distance without erasing signs

Total variation measures how far two distributions are apart. For our uniform reference,

$$
\|\mu(t)-\pi\|_{\mathrm{TV}}=\frac12\sum_{i=0}^{2}\left|\mu_i(t)-\frac13\right|.
$$

The point mass at state 0 is at distance 2/3. In the biased worked example, distances at t=1,2,3 are 1/6,1/24,1/96. A deterministic directed cycle remains at distance 2/3 from this initialization. The identity preparation with initial weights (1,2,7) remains at distance 11/30.

Individual signed deviations can change sign or move between states. Total variation takes absolute values and is nonincreasing under a common Markov transition when the reference is stationary. Do not confuse oscillating components with an increasing total-variation distance. A zero distance means the distributions are exactly equal; a very small nonzero distance is a different statement.

## Derive an upper bound rather than guessing a mixing rate

The largest total-variation separation between two rows of this cyclic matrix simplifies to

$$
\delta(P)=\max_{i,j}\frac12\sum_k|P_{ik}-P_{jk}|=\max(p,q,r)-\min(p,q,r).
$$

For example, the distance between the first two rows is half of the sum of |r−q|, |p−r| and |q−p|. For three real numbers this sum is twice their maximum minus twice their minimum. Other distinct row pairs give the same result.

To see why it is useful, let M be the total variation of a signed zero-sum difference d=μ−π. If M>0, normalize its positive and negative parts into distributions a and b. Then

$$
d=M(a-b),\qquad dP=M\sum_{i,j}a_i b_j\bigl(P_{i\cdot}-P_{j\cdot}\bigr).
$$

The triangle inequality bounds the total variation of the right-hand side by Mδ(P). If M=0 the bound holds directly. Repeating this argument yields

$$
\|\mu(t)-\pi\|_{\mathrm{TV}}\leq\delta(P)^t\|\mu(0)-\pi\|_{\mathrm{TV}},\qquad \delta(P)^0:=1.
$$

This is an upper bound, not an assertion of equality for every preparation. With weights (7,2,5) and a point mass at 0, δ=5/14. The next distribution is (5/14,7/14,2/14), whose distance is 4/21. The bound is 5/21 and is strictly larger. When p=q=r=1/3, δ=0 and every initialization becomes uniform after one step. When δ=1, the bound alone proves no contraction; inspect the actual transition structure.

## Preserve exact differences when displayed probabilities coincide

All admitted parameters are integers, so normalization and finite matrix powers can be evaluated as exact rational numbers. Fractions keep a signed numerator and a positive denominator. Rational endpoints that are equal certify a value; they do not certify that its decimal display has infinitely many correct digits. A displayed 1/3 is a rounded projection.

A sharp example uses f=s=L, b=L−1 with L=1,000,000 and D=3L−1. If R− denotes the counterclockwise permutation matrix, the transition matrix is a uniform all-entries contribution minus R−/D. A zero-sum deviation is therefore rotated, sign-reversed and divided by D at each step:

$$
\mu(t)-\pi=\left(-\frac1D\right)^t\bigl(\mu(0)-\pi\bigr)R_-^t,\qquad \|\mu(t)-\pi\|_{\mathrm{TV}}=\frac{2}{3D^t}\quad\text{for }\mu(0)=(1,0,0).
$$

At t=64 the total variation is positive but smaller than the nonzero range of ordinary floating-point display. All three probabilities can display the same rounded 1/3 even though their exact deviations are nonzero. The exact model must retain those deviations and report an outside-display value, rather than declare exact stationarity from equal-looking bars.

The natural logarithm of a magnitude is a useful approximate display when a linear projection is unavailable. It is separate from the exact fraction and its sign. Exact zero has no finite logarithmic magnitude. No small plotting constant is substituted for it. The same distinction applies to a tiny signed current: orientation is determined by its exact numerator, not a color threshold.

## Connect the example to AI sampling with its assumptions intact

A proposed MCMC transition is often judged by its invariant distribution. This chapter shows why checking only a one-time histogram is insufficient to understand its paths, and why checking target invariance alone does not establish convergence from a chosen initialization. The next chapter will construct target-preserving transitions by correcting asymmetric proposals.

Detailed balance is one useful construction principle. It is not required by every valid sampling design, and a nonzero current does not by itself rank an algorithm's statistical efficiency. Compare the actual distribution error, communication structure and timescale relevant to the intended task. A fixed three-state calculation also does not certify a learned transition rule, a changing training process or an arbitrary language-model sampler.

The arrows here carry probability per discrete transition. Interpreting them as heat currents requires an energy model, reservoirs and a local detailed-balance relation. Interpreting path asymmetry as physical entropy production requires corresponding physical assumptions. None follows from a circulating probability diagram alone.

## Connect distributions and flows to the experiment

Keep the default forward, backward and holding weights 2,1,1 and initial weights 1,0,0. At N=3 the current law is 22/64,21/64,21/64. Think of bars as water levels and arrows as water moving through pipes during one step. They visualize different quantities.

| Action | Observation to inspect | Question to explain |
| --- | --- | --- |
| Press ‘One transition step’ | Only N increases by one at the same transition and initialization; current and next laws change | How does evolving the entire law differ from following one random particle? |
| Toggle ‘Show stationary-reference flow’ | The uniform reference retains forward flow 1/6 and reverse flow 1/12 | Why can constant levels coexist with net current 1/12? |
| Compare ‘One direction, period three’ with ‘One direction with holding’ | Return period, current law and distance to the uniform reference | Why does existence of a stationary law differ from convergence from every start? |

The flow toggle changes only the displayed reference preparation; it does not change the requested N or initial law. Turning a record-table page does not advance time either. On narrow screens, scroll inside the plot and compare exact rational entries in the flow table rather than arrow thickness. Automatic transitions increment N at fixed transition and initialization. The screen-refresh interval has no physical time unit.

## Predict, calculate and explain

### 1. Read a departure row

For weights (2,1,1), start at state 0. What is the next distribution, and what does the middle entry mean?

**Explanation.** Normalization gives p=1/2,q=r=1/4. Row 0 is (1/4,1/2,1/4). The middle entry is the probability of arriving at state 1, not the probability of departing from state 1. The three entries sum to one.

### 2. Find circulation without changing the levels

Using the uniform reference and the same weights, calculate F01, F10, J01 and the net outgoing stationary current at state 0.

**Explanation.** F01=1/6 and F10=1/12, giving J01=1/12. The opposite neighboring edge contributes −1/12, so the sum leaving state 0 is zero. Off-diagonal incoming and outgoing flux are both 1/4. Holding contributes 1/12 to flux but zero to antisymmetric current.

### 3. Inspect a stationary family

Set f=b=0,s=1 and initial weights (1,2,7). Is the initial distribution stationary? Does it converge to uniform π?

**Explanation.** P is the identity, so (1/10,1/5,7/10) is stationary immediately. Every distribution is stationary. Its total variation from uniform is 11/30 and stays so. Uniform is a stationary reference, not a unique attractor in this preparation.

### 4. Separate invariance from convergence

Set p=1,q=r=0 and initialize at state 0. Where is the distribution at t=8? Does the existence of a unique stationary distribution change that answer?

**Explanation.** Eight modulo three is two, so the distribution is concentrated at state 2. Uniform remains the unique stationary distribution, but the class has period three and point-mass initializations never converge to it. Their total variation remains 2/3. A uniform initialization would instead stay uniform.

### 5. Test whether the contraction bound is exact

Use weights (7,2,5) and initialize at state 0. Compare the actual one-step total variation with the derived bound.

**Explanation.** The next distribution is (5/14,7/14,2/14). Its deviations are (1/42,7/42,−8/42), so total variation is 4/21. The coefficient is 5/14 and the initial distance is 2/3, giving an upper bound of 5/21. The inequality is strict; the coefficient is not a universal exact decay factor.

### 6. Diagnose an equal-looking probability display

Use f=s=1,000,000, b=999,999, a point mass at 0 and t=64. All displayed probabilities look like 1/3. Can you replace the total variation by zero?

**Explanation.** No. With D=2,999,999 the exact total variation is 2/(3D^64), which is positive. Its linear floating projection is outside the nonzero display range. The signed rational deviations still distinguish the preparation from exact uniformity. Exact fractions establish nonzero distance even when the bars coincide visually.

## Continue toward general chains and research questions

The triangle provides a complete finite counterexample to “stationary means detailed balance.” It also separates invariant targets, current preparation, unique stationary solutions, periodicity and quantitative error. Those distinctions will recur in the Hastings-correction and two-state mixing chapters.

Further work includes nonuniform stationary references, transitions with restricted support, general communication classes, quantitative mixing estimates, continuous-time generators and local detailed balance in physical models. Hidden state, coarse graining and time-dependent learned kernels require additional assumptions. The finite example supplies tools for asking those questions precisely; it does not settle the general theory or its research extensions.
