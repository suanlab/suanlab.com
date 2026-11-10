## A rate is not a probability

The interactive experiments below compare independently checked finite models. Exact values, certified enclosures and display approximations are distinguished.

Imagine a molecule that can occupy state0 or state1. A clock determines when it changes state. If the clock at state0 rings quickly, the molecule tends to leave0 sooner. The number of rings per second is a rate. It is not the probability of being in state1, and it need not be at most one.

We will evolve the probability law of an ensemble of such molecules. A curve describing this law does not show one molecule's random history. Keeping those two objects separate makes the master equation, its exact solution and an approximate time integrator easier to interpret.

The assumptions are a finite two-state space, constant nonnegative transition rates, the Markov property and a calibrated physical time coordinate. This chapter does not specify energy levels, a heat reservoir or local detailed balance with an environment. Probability flow alone is not a measurement of heat or thermodynamic entropy production. [Levin and Peres, second edition, Section20.1](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf) develops the continuous-time generator and its relation to a Poisson clock. We derive this particular two-state model below.

## Fix the direction, units and initialization

Let alpha be the rate from0 to1 and beta the rate from1 to0. Both have units s^−1. For a small positive h, conditional on starting in0, the probability of a change to1 is alpha h plus terms smaller than first order. A rate of4s^−1 is valid; treating4 itself as a transition probability is not.

```text
                       alpha [1/s]
               state 0 -----------> state 1
                       <-----------
                        beta [1/s]

          initial probabilities: (1-u, u)
          physical time: t [s], not the number of arrows drawn
```

Write the row probability law as mu(t)=(1−p(t),p(t)), with p(0)=u in[0,1]. Specify u independently of the transition rates. Equilibrium is one possible initialization, not a default assumption that erases relaxation.

| Quantity | Meaning | Unit |
| --- | --- | --- |
| alpha, beta | Departure rates for the two directed moves | s^−1 |
| Q | Generator acting on row probability laws | s^−1 |
| t, h | Physical time and an integrator step | s |
| p(t), u | State-one probability and initial probability | dimensionless |
| J(t) | Net probability mass transferred per time | s^−1 |
| N | Number of Euler steps in a fixed time window | dimensionless |

## Build the generator from mass balance

The generator records nonnegative off-diagonal rates and a negative departure rate on each diagonal:

$$
Q=\begin{pmatrix}-\alpha&\alpha\\\beta&-\beta\end{pmatrix},\qquad
\dot\mu(t)=\mu(t)Q.
$$

Each row sums to zero, ensuring that the derivative of total probability is zero. This is not a stochastic matrix: its diagonal entries can be negative and its rows sum to zero rather than one. Multiplying by a short time interval creates an approximation to a transition matrix.

With a column law, the same convention becomes mu_dot_column=Q^T mu_column. Transposing the convention changes notation, not the physical directions. Always name the departure state and arrival state before reading a matrix entry.

State1 gains alpha times the mass in0 and loses beta times the mass in1:

$$
\dot p=\alpha(1-p)-\beta p=\alpha-(\alpha+\beta)p.
$$

The conservation law does not yet prove positivity of an arbitrary numerical approximation. The exact continuous-time dynamics and a discretized recurrence must be checked separately.

## Solve the relaxation without stepping through random jumps

When lambda=alpha+beta is positive, define pi1=alpha/lambda and pi0=beta/lambda. Subtracting the stationary value reduces the differential equation to a decaying mode:

$$
\lambda=\alpha+\beta,\qquad
p(t)=\pi_1+(u-\pi_1)e^{-\lambda t}.
$$

The sum of the two rates controls relaxation of the distribution. An individual departure rate controls a different waiting-time question. For alpha=2s^−1, beta=1s^−1 and u=0, the stationary occupied probability is2/3. At t=1s, p is about0.633475, below its limit. No random trajectory is needed to calculate this law.

At t=0 the exponential is one and the formula returns u. If the initial law is stationary, its deviation is zero at every time. Otherwise the deviation remains nonzero at each finite time because the exponential is strictly positive. Small displayed differences are not exact equality.

## Handle zero rates before dividing

If both rates are zero, Q is zero and every initial law remains unchanged. There is no unique stationary reference. The selected initial law is its own long-time limit, but that does not turn it into the unique equilibrium law of the model.

If alpha=0 and beta>0, state0 is absorbing. The stationary law is unique even though the process is not irreducible. All initial laws converge to state0. Interchanging the rates gives absorption at1. These examples distinguish stationary uniqueness, irreducibility and a particular initialization.

For any nonzero rate sum, the stationary flows satisfy pi0 alpha=pi1 beta. A two-state stationary law cannot sustain a circulating net current around a cycle. To study such currents we will need more states or a richer physical description.

## Derive the transition kernel at a physical time

The exact kernel K_t maps any initial row law to its law at time t. Let q=exp(−lambda t). Solving from each possible initial state gives

$$
K_t=e^{tQ}=
\begin{pmatrix}
\pi_0+\pi_1q&\pi_1(1-q)\\
\pi_0(1-q)&\pi_1+\pi_0q
\end{pmatrix}.
$$

Its entries are nonnegative and its rows sum to one. At zero time it is the identity. An identity kernel at that one instant does not establish uQ=0; stationarity requires invariance for all future times. With both rates zero, use the identity directly instead of the divided expression.

Successive physical intervals obey the semigroup relation K_s K_t=K_(s+t). For this constant generator, differentiating the exponential gives the master equation. The exact probability of being in a state after a finite interval already includes paths that changed state more than once. Replacing it by one departure probability omits those possibilities.

## Separate waiting for the first jump from returning to a state

Starting in0, let tau be the first departure time. Constant hazard alpha gives

$$
\mathbb P(\tau>t\mid X_0=0)=e^{-\alpha t},\qquad
\mathbb E[\tau\mid X_0=0]=1/\alpha\quad(\alpha>0).
$$

At alpha=0 the waiting time is infinite, not zero. Starting in1 uses beta instead. The probability of no departure differs from K_t(0,0), because the latter also counts paths that left and returned.

For equal rates of1s^−1 and t=1s, the no-jump probability from0 is about0.367879. The probability of being back in0 is about0.567668. The difference comes from return paths. A drawing with a molecule sitting at0 at the final instant cannot distinguish these histories.

## Read the instantaneous current with its sign

The two directed flows and the net flow are

$$
F_{01}=\alpha\mu_0,\qquad F_{10}=\beta\mu_1,\qquad
J=F_{01}-F_{10}=\dot p=-\lambda(p-\pi_1).
$$

At stationarity the directed flows can both be positive while their difference is zero. A negative transient current means probability is moving toward0 in the chosen orientation; it is not a negative probability or a negative dissipation rate.

Across a finite window, the net transferred mass is the integral of J. Multiplying the initial current by the entire window is generally only an approximation:

$$
p(t+h)-p(t)=\int_t^{t+h}J(s)\,ds.
$$

Thus an interval's probability change and its starting instantaneous current are related but distinct observables.

## Construct Euler at the same final physical time

Choose N equal steps over a fixed window T, so h=T/N. Forward Euler gives

$$
\mu_{k+1}^{E}=\mu_k^{E}(I+hQ),\qquad
p_{k+1}^{E}=\alpha h+(1-\lambda h)p_k^{E}.
$$

At the grid time kh, its closed form is

$$
p_k^{E}=\pi_1+(u-\pi_1)(1-\lambda h)^k.
$$

Compare this with the exact law at the same kh. Increasing N while fixing T refines the approximation; increasing N while fixing h changes the final physical experiment. These operations must not share an unlabeled animation control.

For the reference alpha=2, beta=1, T=1 and N=8, Euler predicts about0.651145 at the endpoint, compared with the exact0.633475. Both curves can be plausible probabilities while representing measurably different dynamics.

## Positivity and linear stability answer different questions

Euler preserves total mass because every row of I+hQ sums to one. For it to be a stochastic matrix, however, its diagonal entries must also be nonnegative:

$$
h\alpha\le1,\qquad h\beta\le1.
$$

The signed deviation contracts in the linear recurrence when

$$
|1-\lambda h|<1.
$$

These conditions differ. With alpha=4, beta=1 and h=0.35s, the deviation multiplier is−0.75, so the linear mode contracts. But the first Euler row is(−0.4,1.4), which cannot describe probabilities. Clipping it to[0,1] or renormalizing it would create another algorithm and conceal the failure.

A stationary initial law can hide the problem: u=0.8 remains0.8 under this same invalid kernel. One successful trajectory of the recurrence does not certify positivity for every possible initial law.

## A stochastic Euler matrix can still give the wrong dynamics

With alpha=beta=1 and h=1s, Euler swaps the two states on every step. Its deviation multiplier is−1. This is a valid discrete transition matrix with period two, whereas the continuous-time law relaxes smoothly. Mass conservation and nonnegative probabilities do not establish accuracy or mixing of the approximation.

There is another revealing case. With alpha=2, beta=1 and h=1/3s, Euler's deviation multiplier is zero. One Euler step lands exactly on pi1=2/3. The exact finite-time probability from u=0 is instead2(1−exp(−1))/3, about0.421414. An approximation that appears to equilibrate immediately is not necessarily a better model of physical relaxation.

## Preserve small changes and identify what data determine

For equal rates5s^−1 at T=10s from u=0, the exact deviation is−exp(−100)/2. A rounded probability can display0.5 while the net current remains5exp(−100), strictly positive. Computing the current from the signed deviation preserves a quantity that subtracting rounded flows could erase.

The calculation uses exact rational inputs, rational Euler values and directed256-bit exponential intervals. These representations distinguish analytic zeros from small numbers and display rounding. Individual coordinate intervals are not independent random variables. Their boxes do not replace the exact mass-conservation identity.

Scaling both rates by a positive c and dividing every physical time by c preserves the transition kernels. The stationary ratio alone therefore cannot determine the rate scale. Time-resolved measurements can provide additional information, but fitting rates still needs a stated observation model. A learned score or a count of optimizer iterations is not automatically a calibrated physical clock.

## Use the visual experiment

The lab keeps physical time, Euler resolution and initialization separate. Use the named presets, exact inputs and time-record selector:

| Control or comparison | Expected observation | Explanation to give |
| --- | --- | --- |
| Refine N at fixed T | Orange Euler values approach the blue exact law at common grid times | Numerical resolution changes while the physical window stays fixed |
| Compare a stationary start and a concentrated start under an invalid Euler kernel | One can remain valid while the other produces signed values | Positivity is a property of the whole kernel |
| Compare return probability and no-jump survival | Returning includes more histories | A final state does not specify a path |
| Double both rates and halve time | Occupancy agrees at corresponding times; currents double | Dimensionless rate-time products and dimensional flows differ |

The calculation range is rates0..5s^−1, T=0..10s and N=1..128. Rational input numerators and positive denominators are at most one million. These are finite computation limits, not limits of the analytical theory. No sampled particles, heat measurements or entropy-production claims are part of this first lab.

## Six exercises with explanations

### 1. Derive the law from a flow ledger

Use alpha=2, beta=1, u=0 and t=1. Find the generator, stationary occupied probability and exact occupied probability.

**Explanation.** The rows of Q are(−2,2) and(1,−1), each summing to zero. Setting alpha−(alpha+beta)p=0 gives2/3. The initial deviation is−2/3, so the solution is2(1−exp(−3))/3, about0.633475. The value is between the initial probability and its stationary limit. A generator entry2 has units per second and is not itself a probability.

### 2. Identify the missing return paths

At equal unit rates, compare no departure from0 by time one with being in0 at time one. Why are the two probabilities different?

**Explanation.** First-departure survival is exp(−1). The kernel gives K_1(0,0)=(1+exp(−2))/2. Return paths add about0.199788. The relaxation rate is2, while the initial departure rate is1. Conflating these two exponential factors changes the event being calculated.

### 3. Find a stable but unphysical Euler step

For alpha=4, beta=1 and h=0.35, check row sums, positivity and linear contraction. Start at state0.

**Explanation.** The Euler matrix has rows(−0.4,1.4) and(0.35,0.65). Both sum to one. The deviation multiplier−0.75 has magnitude below one, yet the first step produces occupied value1.4. Calling it a probability would be wrong. Linear stability does not impose the sign constraints of a probability kernel.

### 4. Explain periodicity created by discretization

Use equal unit rates and h=1 over four steps from state0. Compare the endpoint with the continuous-time law.

**Explanation.** Euler alternates0,1,0,1,0 in its occupied value, returning to zero after four steps. The exact law at time four is(1−exp(−8))/2, close to0.5. A period-two numerical kernel has replaced smooth continuous-time relaxation. Reducing h at the same final time changes this approximation, rather than changing the physical rate.

### 5. Diagnose a favorable initial law

At alpha=2, beta=1, T=1 and N=1, start with u=2/3. Can the constant numerical law certify a valid Euler kernel?

**Explanation.** The Euler kernel has first row(−1,2), so it fails positivity. Nevertheless pi Q=0 implies pi(I+hQ)=pi, and this particular initialization stays stationary. The exact law is stationary as well. Their agreement on one vector cannot certify the matrix for other initial laws; u=0 exposes the invalid occupied value2 immediately.

### 6. Separate scale identification from rounded equality

Explain why a stationary occupancy ratio does not determine both rates. Then interpret a displayed0.5 in the equal-rate, long-time example.

**Explanation.** Multiplying both rates by c leaves alpha/(alpha+beta) unchanged, while relaxation becomes c times faster. Without a time scale the stationary ratio identifies no unique rate pair. At rates5 and time ten, the displayed probability rounds to0.5 but its deviation is−exp(−100)/2 and its current is5exp(−100). Rounded equality neither identifies the rates nor proves exact finite-time stationarity.

The next master-equation readers will separate Poisson event clocks from physical state changes, then compare time occupation with jump-count occupation when holding rates depend on state. These constructions prepare the probability laws needed before discussing Fokker–Planck equations, response and stochastic thermodynamics.
