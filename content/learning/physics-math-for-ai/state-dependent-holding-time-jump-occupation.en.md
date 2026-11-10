## Count visits or measure time?

The interactive experiments below compare independently checked finite models. Exact values, certified enclosures and display approximations are distinguished.

Three machines hand a job clockwise from0 to1 to2 and back to0. Every completed circuit visits each machine once. Machine0 can nevertheless hold the job much longer than the others. A log written only when a handoff occurs sees equal visit counts. A clock that checks where the job is at an arbitrary time sees unequal occupation.

This example separates the chain at actual jump instants from the continuous-time state law. It matters when an AI system learns from event logs: the rule determining when observations are recorded can change the distribution being estimated. This is a mathematical illustration, rather than a claim that every real log follows a Markov model.

Assume three states, constant nonnegative state-dependent exit rates, a clockwise successor, exponential holding times and the Markov property. Use row laws and time in seconds. We specify no energies or thermal reservoirs. The rate/holding-time construction follows [Levin and Peres, second edition, Section20.1](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf); the occupation and absorption results below are derived for this particular cycle.

## Specify both the destination and the waiting law

At state i, the exit rate is r_i in s^−1. If r_i>0, the holding duration Hi is exponentially distributed and the next actual state is i+1 modulo three. The mean1/r_i is not a deterministic delay.

$$
\Pr(H_i>t)=e^{-r_it},\qquad \mathbb E[H_i]=\frac1{r_i}\quad(r_i>0).
$$

If r_i=0, the particle never leaves that state. Its holding duration is infinite. There is no next actual jump, so the corresponding next-jump probability row is undefined. Inserting a self-jump with probability one would change what is being counted: staying forever is not an infinite sequence of actual jumps.

```text
       Exp(r0)          Exp(r1)          Exp(r2)
   0 ------------> 1 ------------> 2 ------------> 0

   Actual visits: each completed circuit visits every state once.
   Physical time: each visit contributes a random holding duration.
```

The generator records rates. When all rates are positive, the actual-jump kernel J records deterministic successor choices. They are different matrices with different units and purposes.

$$
Q=\begin{pmatrix}-r_0&r_0&0\\0&-r_1&r_1\\r_2&0&-r_2\end{pmatrix},\qquad
J=\begin{pmatrix}0&1&0\\0&0&1\\1&0&0\end{pmatrix},\qquad
\mu(t)=\mu_0e^{tQ}.
$$

## Derive time occupation from stationary flow

First suppose every rate is strictly positive. Stationarity requires equal incoming and outgoing probability at each state. On this directed ring the same stationary current passes through each edge.

$$
\pi_0r_0=\pi_1r_1=\pi_2r_2=c,\qquad
\pi_i=\frac{1/r_i}{\sum_{j=0}^{2}1/r_j},\qquad
c=\frac1{\sum_{j=0}^{2}1/r_j}.
$$

This is global balance, not pairwise detailed balance. Every reverse transition rate is zero, while the clockwise stationary current is positive. The uniform law is stationary for J, but generally not for Q.

For rates1,2,4, the mean holding times are1,1/2,1/4 seconds. Their sum is7/4 seconds. Normalizing them gives time occupation(4/7,2/7,1/7). The directed edge current is4/7 per second and the mean stationary rate of all actual jumps is12/7 per second.

The slowest state holds the largest time share. This is not an extra force toward that state: each completed circuit still visits it just once. Longer residence makes it more likely to be found by an observation made at a typical physical time.

## Derive jump frequencies without confusing convergence

With all rates positive, the sequence of actual jump states cycles deterministically. Over many completed circuits the fraction of visits to each state tends to one third. The stationary visit law nu for J is therefore uniform.

$$
\nu_i=\frac13,\qquad
\pi_i=\frac{\nu_i/r_i}{\sum_j\nu_j/r_j},\qquad
\nu_i=\frac{\pi_i r_i}{\sum_j\pi_jr_j}.
$$

The weighting has two directions. To turn visit frequencies into time shares, multiply each visit frequency by its mean holding duration and normalize. To turn time shares into departure frequencies, multiply by the local exit rate and normalize.

This relation assumes a positive recurrent process with positive exit rates in the states under discussion. Applying1/r_i at an absorbing state would divide by zero and erase the distinction between an infinite jump sequence and a sequence that terminates.

The distribution after exactly k jumps need not converge: starting at0 gives a period-three sequence. Long-run empirical visit frequencies can still be uniform. In contrast, this irreducible finite continuous-time law converges to pi. A limit of distributions and a limit of average counts are separate statements.

## Check initialization before interpreting a flat curve

For rates1,2,4, the initial law(4/7,2/7,1/7) is stationary in physical time. Its state probabilities stay constant, although individual particles keep jumping. A stationary ensemble does not mean that every particle has stopped.

Starting uniformly gives equal jump-style weights, but not a stationary continuous-time law. Its initial derivative is(1,−1/3,−2/3) per second: state0 gains mass because it receives departures from fast state2 and leaves slowly itself.

This derivative is an immediate diagnostic. It uses mu0 Q and does not require waiting for a long simulation. A graph that looks flat at a rounded resolution cannot replace the exact stationarity equation.

| Observation rule | Limiting weights for rates1,2,4 | What it measures |
| --- | --- | --- |
| Actual jump instants, over many visits | 1:1:1 | Visit counts |
| Long physical-time occupation | 4:2:1 | Residence duration |
| A stationary physical-time initialization | 4:2:1 at every time | Ensemble marginals, with continuing jumps |

## Handle one absorbing state before taking limits

If r0=0 and r1,r2>0, state0 is absorbing. Every other state eventually reaches it along the clockwise path. The unique stationary law is concentrated at0, and every initial law converges to it. The actual-jump sequence contains only finitely many jumps almost surely.

From state1, absorption requires waiting at1 and then at2. The mean absorption time is the sum of the two holding means. Starting at the absorber gives mean zero, not infinity: absorption has already occurred, even though the future holding time there is infinite.

$$
\mathbb E_1[T_{\mathrm{abs}}]=\frac1{r_1}+\frac1{r_2},\qquad
\mathbb E_2[T_{\mathrm{abs}}]=\frac1{r_2},\qquad
\mathbb E_0[T_{\mathrm{abs}}]=0.
$$

For r1=2,r2=4, the first mean is3/4 second. This is a mean over random durations; it does not say every particle arrives at time0.75.

## Preserve a family when several states absorb

If r0=r2=0 and r1>0, states0 and2 are absorbing. Starting at1 leads to2 after one actual jump. Starting at0 stays at0. Every probability mixture supported on0 and2 is stationary, so there is no unique stationary law shared by all initializations.

For initial law(w0,w1,w2), the selected long-time limit is(w0,0,w1+w2). For initial weights2,3,5, this is(0.2,0,0.8). It depends on the initial absorption basins rather than an inverse-rate normalization.

More generally, the stationary laws on a cycle with zero exit rates are precisely the probability laws supported on the zero-rate states. Each positive-rate state flows to the first absorbing state reached clockwise. If all rates are zero, every initial law is stationary and no actual jumps occur.

A positive dominating clock can still generate fictitious events in an absorbing state. Those events must not manufacture actual departures or remove the stationary family.

## Use potential events as a computation, not as actual visits

Choose a constant clock rate lambda at least max r_i. Define a potential-event kernel R whose diagonal depends on the current state. A slow state rejects more proposed moves.

$$
R_{ii}=1-\frac{r_i}{\lambda},\qquad R_{i,i+1}=\frac{r_i}{\lambda},\qquad
Q=\lambda(R-I).
$$

This R includes self events, unlike J, which records only actual jumps and may have undefined rows at absorbers. The Poisson-mixture construction remains valid for the physical law. At cutoff M it produces a retained subprobability vector v_M and an omitted Poisson mass tau_M.

$$
v_M=e^{-\lambda t}\sum_{k=0}^{M}\frac{(\lambda t)^k}{k!}\mu_0R^k,\qquad
0\le\mu_j(t)-(v_M)_j\le\tau_M,\qquad
\tau_M=\Pr\{\operatorname{Poisson}(\lambda t)>M\}.
$$

The shared calculation uses exact rational conditional vectors, directed exponential intervals and a positive tail series. It never silently normalizes the retained vector. A small cutoff can give broad component intervals even if an independent stationary or absorbing argument gives an exact law.

Changing lambda while keeping Q,t,mu0 fixed changes the representation and its cutoff cost, rather than the physical law. Cutoff is a numerical coordinate; physical time is a different coordinate. No random sample path is produced by these probability calculations.

## Unequal exit rates change count statistics

At stationarity with all rates positive, the mean number of actual departures per unit time is the occupation-weighted exit rate. On this ring it equals three divided by the mean complete-cycle duration.

$$
\bar r=\sum_i\pi_i r_i=\frac3{\sum_i1/r_i},\qquad
\mathbb E_\pi[N_t]=t\bar r.
$$

This determines an expectation under stationary initialization. It does not prove that Nt is Poisson. With unequal exit rates, successive inter-jump durations have different exponential rates in a cyclic order. A whole-cycle duration is a sum of those exponentials, not generally one exponential.

When every exit rate is the same r, actual inter-jump times are independent exponentials of rate r and the count is Poisson(rt). When all rates vanish, it is the degenerate zero count. For unequal rates there is no single homogeneous Poisson count law valid for every initial state. The calculation deliberately does not display rt for an arbitrary transient initialization.

## What event-only observations cannot identify

A log containing only the successive state names sees the same deterministic clockwise sequence for every positive rate triple. It cannot identify the residence rates. Multiplying all rates by a common constant leaves time occupation ratios unchanged as well, while changing the physical relaxation and turnover scales.

To estimate rates from a fully observed path, one needs time information. If Ni is the number of departures from state i and Ti its total observed residence time, the rate-dependent log likelihood contains the terms below. A final unfinished residence contributes exposure time even though it contributes no departure.

$$
\ell(r)=\sum_i\left(N_i\log r_i-r_iT_i\right)+\text{constant},\qquad
\widehat r_i=\frac{N_i}{T_i}\quad(T_i>0).
$$

The log expression is written for positive rates; a zero-rate boundary is treated by a limit, with0 log0 contributing zero when there were no departures. This is a path-observation calculation, not an estimator implemented by this reader. If Ti=0, that rate is not identified from this formula. If Ni=0 and Ti>0, the maximum over nonnegative rates is at zero, but a finite no-jump observation does not prove that the true rate is exactly zero. Sampling uncertainty and model assumptions remain relevant.

## Use the controls and comparisons

The reader inputs are three rational exit rates0..5/s, a dominating potential clock0..10/s, physical time0..10s, nonnegative initial weights with positive sum, and cutoff0..256. These bounds control computation, rather than course length or the mathematical theory. Invalid clocks and invalid integer text must preserve the last valid preparation.

| Comparison | Keep fixed or change | Read and explain |
| --- | --- | --- |
| Uniform visits versus time occupation | Rates1,2,4; compare two initial laws | Jump frequencies, holding means, pi and initial stationarity |
| One or two absorbing states | Set selected exit rates to zero | Undefined next-jump rows, unique law or stationary family, selected limit |
| Two potential clocks | Hold physical generator and time fixed | Same physical law; different fictitious-event probabilities and cutoff error |
| Change the overall time scale | Divide every rate and the clock by two; double time | Same dimensionless physical query; doubled holding and cycle means |

Read the rate-labeled ring, separate visit/time occupation bars, actual/potential kernel tables and certified component brackets. The physical-time scan changes the ensemble query, never a deterministic countdown of mean waiting times.

## Worked self-checks

### 1. Compute the two occupation laws

For rates1,2,4, find the long-run jump visit law and the physical-time law.

**Explanation.** Every complete actual-jump circuit visits all three states, giving(1/3,1/3,1/3). The mean residence durations1,1/2,1/4 normalize to(4/7,2/7,1/7). Both answers are correct for their respective observation rules; exchanging them changes the meaning of the data.

### 2. Test a seemingly natural initialization

Is the uniform initial law stationary for rates1,2,4? Compute its derivative.

**Explanation.** Multiplying(1/3,1/3,1/3) by Q gives((4−1)/3,(1−2)/3,(2−4)/3)=(1,−1/3,−2/3) per second. The derivative sums to zero, preserving mass, but is not the zero vector. Equal initial weights do not imply a stationary physical-time law.

### 3. Separate holding from absorption time

Let r0=0,r1=2,r2=4. Starting at1, what is the mean time to absorption? What if the initial state is0?

**Explanation.** From1, the particle waits a mean1/2 second, moves to2, waits a mean1/4 second and enters0, giving3/4 second. Starting at0 gives absorption time zero. Its subsequent holding duration is infinite, but that is a different random variable.

### 4. Keep initialization in an absorbing family

Let r0=r2=0,r1=2 and initial weights2,3,5. Find the selected long-time law and the mean absorption time.

**Explanation.** Initial mass0.2 stays at0, mass0.5 stays at2, and mass0.3 from1 reaches2. The limit is(0.2,0,0.8). Only the initial mass at1 waits before absorption, with mean1/2 second, giving an ensemble mean0.3×0.5=0.15 second. No division by a zero exit rate is used.

### 5. Identify a rate scale from observations

Do jump state names or stationary time shares determine the full rate triple?

**Explanation.** Clockwise state names give the same jump sequence for every positive triple. Time shares determine inverse-rate ratios, but multiplying every rate by c preserves those shares. Durations or another calibrated time scale are needed to determine the common multiplier. A finite sample additionally requires statistical uncertainty to be assessed.

### 6. Reject a count interpretation based only on its mean

At stationary rates1,2,4, the expected jump count over one second is12/7. Does that make the count Poisson(12/7)?

**Explanation.** No. Its expectation follows from integrating the stationary departure rate. Actual waiting durations alternate exponential rates1,2,4 rather than sharing a single constant rate. Equality of one moment does not identify a distribution. A constant potential-event clock remains Poisson, but some of its events are self events and it counts a different object.

These master-equation constructions prepare the distinction between density evolution, probability current and physical observations before moving to Fokker–Planck equations and response theory.
