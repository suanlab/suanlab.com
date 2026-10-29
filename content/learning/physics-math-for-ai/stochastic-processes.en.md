## A distribution can evolve in time

A random process describes a collection of time-indexed random variables. We start with a two-state discrete-time Markov chain. State 0 moves to state 1 with probability $a$; state 1 moves to state 0 with probability $b$. The Markov assumption says the current state is sufficient for the specified next-state conditional distribution. It does not say observations in time are independent.

The lab evolves probabilities exactly rather than simulating one random path. This separation makes normalization, stationarity and convergence directly checkable. Individual trajectories can jump even while their distribution is stationary.

## Set the orientation before multiplying

Using row distributions, the transition matrix is

$$
P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix},\qquad0\le a,b\le1.
$$

Each row sums to one. Let $p_t$ denote the probability of state 1, so the distribution row is $(1-p_t,p_t)$. Multiply this row by $P$ and read its second component:

$$
p_{t+1}=a(1-p_t)+(1-b)p_t=a+(1-a-b)p_t.
$$

A column-distribution convention would transpose the matrix. Mixing conventions can preserve plausible-looking numbers while calculating the wrong process.

## Find the stationary distribution

Stationarity solves $\pi_1=a+(1-a-b)\pi_1$. If $a+b>0$,

$$
\pi_1=\frac a{a+b},\qquad\pi_0=\frac b{a+b}.
$$

For $a=0.2,b=0.3$, stationary occupancy is 0.4. Starting from $p_0=1$, the first updates are 0.7, 0.55 and 0.475. Probability approaches 0.4, not zero, because transitions continue in both directions.

At stationarity the two-state flow balances: $\pi_0a=\pi_1b$. This example has no nontrivial circulating probability current. More complex networks can have stationary currents; those require additional states and conditions.

## Prove convergence for this example

Subtract the stationary value:

$$
p_t-\pi_1=(1-a-b)^t(p_0-\pi_1).
$$

The second eigenvalue is $1-a-b$. When $0<a+b<2$, its absolute value is below one and occupancies converge. A negative eigenvalue makes the deviation alternate sign while shrinking.

At $a=b=1$, the stationary distribution still exists and is $(1/2,1/2)$, but a chain starting entirely in state 1 alternates between states forever. Existence and uniqueness of a stationary distribution do not alone establish convergence from every initial state.

If $a=b=0$, every distribution is stationary. The formula becomes $0/0$ and the lab reports no unique stationary occupancy rather than inventing one. These boundary cases are essential for correctly interpreting numerical fixed points.

## Time, dependence and sampling

A stationary marginal distribution does not make consecutive samples independent. Slow transitions yield long correlated runs. For estimating a mean from a trajectory, uncertainty must account for dependence; counting each step as an independent observation can greatly overstate effective information.

Markov-chain Monte Carlo aims to sample a target distribution through transitions. A correct stationary target is one requirement, but mixing, initialization and diagnostic behavior also matter. This two-state calculation does not validate arbitrary high-dimensional sampling algorithms.

## Read the visualization and physical limits

Blue is exact state-1 probability across 30 transitions; orange is the stationary value when uniquely defined. It is not a particle position, a single observed binary trace or a density over continuous space. The table shows the second eigenvalue to connect convergence to the matrix structure.

Do not infer physical temperature or entropy production from a transition matrix alone. A physical stochastic-thermodynamic model needs energies, reservoirs, reversal conventions and units. A pedagogical analogy to learning dynamics does not supply those missing assumptions.

## Exercises and source

1. For $a=0.1,b=0.4$, find stationary occupancy. **Answer:** 0.2.
2. Starting at $p_0=0$, compute two updates for the reference chain. **Answer:** 0.2 and 0.3.
3. Explain the difference between a stationary random trajectory and a trajectory whose state never changes.
4. Design a numerical convergence diagnostic that would detect the period-two preset instead of declaring success from an average near 0.5.

For Markov chains and random-process foundations, see [MIT Introduction to Probability, Random Processes](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/). All boundary conclusions above follow directly from the displayed recurrence.
