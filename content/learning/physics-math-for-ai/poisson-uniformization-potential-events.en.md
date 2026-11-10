## One clock tick need not change the state

The interactive experiments below compare independently checked finite models. Exact values, certified enclosures and display approximations are distinguished.

Imagine a particle moving between three locations arranged on a ring. A proposed update can send it to a neighbor or leave it where it is. Counting proposed updates and counting actual moves therefore measure different things. Neither count is the physical time coordinate.

This distinction gives a useful construction of a continuous-time Markov chain: run a Poisson clock, then apply a discrete probability kernel at each tick. It also gives a numerical method with a visible error certificate. We sum only finitely many possible tick counts, retain the missing probability mass and bound how much it can change each state probability.

We assume a finite ring, time-independent rates, the Markov property, a row-vector probability convention and a calibrated time in seconds. The model describes a probability law, rather than a sampled particle path. It assigns no energies, thermal bath or physical heat. The construction follows the Poisson-clock representation in [Levin and Peres, second edition, Section20.1](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf); the particular ring, comparison cases and truncation certificate are derived here.

## Specify the generator before choosing a clock

Label states0,1,2 and interpret indices modulo three. The clockwise rate is a and the counterclockwise rate is b. For example, clockwise from2 means a move to0. Rates have units s^−1, may exceed one and must be nonnegative. The initial row law is a normalized vector mu0. Entering weights2,3,5 means initial probabilities0.2,0.3,0.5, rather than transition rates.

```text
             clockwise a
          0 ----> 1 ----> 2 ----> 0
          0 <---- 1 <---- 2 <---- 0
           counterclockwise b

   Q has rate units; Q times a duration is dimensionless.
```

With the row convention, the equation is mu dot=mu Q. Each row of Q sums to zero; its diagonal records the negative total exit rate. There are no negative transition rates hidden in that diagonal.

$$
Q=\begin{pmatrix}
-(a+b)&a&b\\
b&-(a+b)&a\\
a&b&-(a+b)
\end{pmatrix},\qquad \dot\mu(t)=\mu(t)Q,\qquad \mu(t)=\mu_0e^{tQ}.
$$

The three exit rates are equal to r=a+b. This equality will matter when interpreting actual jump counts. State-dependent holding rates require a different argument in the next unit.

## Add potential events without changing the physical law

Choose a clock rate nu at least r. At a clock tick, apply R=I+Q/nu. The off-diagonal entries are nonnegative and the row sums are one. The diagonal1−r/nu is the probability of staying in place at a potential event.

$$
R=I+\frac{Q}{\nu},\qquad R_{ii}=1-\frac{a+b}{\nu},\qquad R_{i,i+1}=\frac a\nu,\qquad R_{i,i-1}=\frac b\nu,\qquad Q=\nu(R-I).
$$

For a=1,b=0,nu=1, each tick moves clockwise. Raising nu to2 changes each tick into a half chance of moving and a half chance of staying. There are twice as many potential ticks per second and half of them cause a move. The physical generator stays exactly the same.

Nu is a representation choice, rather than an extra physical transition rate. Its admissibility condition is essential. If nu<r, the diagonal of R is negative; an algebraic identity still exists but R is no longer a probability kernel, so the Poisson-mixture probability interpretation fails.

When r=nu=0, division by nu is undefined. Handle this case separately: Q=0, use R=I, and there are no events. The exact law remains its initialization. A zero generator with nu>0 also has R=I, but now every potential event is fictitious.

## Condition on the random number of ticks

Let Kt be a Poisson count of mean nu t, independent of the discrete chain with kernel R. Conditional on Kt=k, the state law is mu0 R^k. Averaging these conditional laws recovers the complete continuous-time law.

$$
\Pr(K_t=k)=e^{-\nu t}\frac{(\nu t)^k}{k!},\qquad
\mu(t)=e^{-\nu t}\sum_{k=0}^{\infty}\frac{(\nu t)^k}{k!}\mu_0R^k.
$$

To see why this is the matrix exponential, note that I commutes with R. Thus e^{nu t(R−I)} factors into e^{−nu t} e^{nu t R}. Expanding the latter exponential gives the series. This is not an Euler step with duration1/nu: it is an exact average over a random number of discrete updates at a fixed physical time.

| Object | What is held fixed? | Interpretation |
| --- | --- | --- |
| mu0 R^k | Number k of potential events | Conditional discrete law |
| mu(t) | Physical duration t | Mixture over every possible event count |
| Cutoff M | Finite part of the same mixture | Numerical work and omitted probability, not a different physical time |

A curve indexed by M refines a computation at fixed t. It does not trace the particle through time. A curve indexed by t changes the physical question. A clock comparison changes nu while keeping Q,t,mu0 fixed.

## A periodic embedded chain can still relax in physical time

Take a=1,b=0,nu=1 and start at0. After exactly k potential events, the state is k modulo three. The discrete laws rotate forever, with period three. They do not converge as k grows.

At physical time t, however, the event count is random. The three probabilities sum the Poisson weights in the three residue classes. For this particular one-way ring they can also be written explicitly.

$$
p_j(t)=\frac13\left[1+2e^{-3t/2}\cos\left(\frac{\sqrt3\,t}{2}-\frac{2\pi j}{3}\right)\right],\qquad j=0,1,2.
$$

The oscillatory part is multiplied by an exponentially decreasing amplitude. The continuous-time law therefore converges to the uniform distribution even though R is periodic. This is a property of this irreducible finite generator, not a claim that every reducible continuous-time system reaches one common law.

For general nonzero a+b on this ring, the stationary law is uniform. A stationary directed edge current is(a−b)/3 in units s^−1. Equal occupancies therefore do not imply vanishing probability circulation. Without an environmental model, this current is not itself a heat rate or a thermodynamic entropy-production rate.

## Truncate the event sum and keep the missing mass

Let x=nu t and define c_k=x^k/k!. A cutoff M retains counts0 through M, including both endpoints. All retained coefficients and all conditional laws are nonnegative. The retained vector is a subprobability law.

$$
v_M=e^{-x}\sum_{k=0}^{M}c_k\mu_0R^k,\qquad
s_M=\sum_j(v_M)_j=e^{-x}\sum_{k=0}^{M}c_k,\qquad
\tau_M=\Pr(K_t>M)=1-s_M.
$$

At x>0 and any finite M, the Poisson tail is strictly positive. A displayed s_M=1 can be a rounded value; it does not prove tau_M=0. At x=0, only k=0 has positive weight, and the retained mass is exactly one for every allowed cutoff.

For each component, the omitted conditional vectors have entries between zero and one. Consequently the missing component is nonnegative and no larger than the omitted total mass. The total L1 difference between the complete law and the retained vector equals that mass.

$$
0\le \mu_j(t)-(v_M)_j\le\tau_M,\qquad
\sum_j\left|\mu_j(t)-(v_M)_j\right|=\tau_M.
$$

This certificate concerns a subprobability vector. Calling its L1 loss a total-variation distance between two probability laws would need a separate convention. The computation leaves the vector unnormalized so its missing mass stays visible.

Dividing v_M by s_M, when s_M>0, gives a valid distribution conditional on Kt≤M. It changes the probability question. It can be used as a controlled approximation if the conditioning and its error are declared, but it must not silently replace the unconditional law.

## Bound a tiny tail using positive terms

Directly computing1−s_M can erase a small positive tail when s_M rounds to one. Instead, sum the positive omitted terms. Once L+2>x, successive terms after c_(L+1) have ratios at most x/(L+2), giving a geometric upper bound.

$$
c_{k+1}=c_k\frac{x}{k+1},\qquad
B_L=\frac{c_{L+1}}{1-x/(L+2)},\qquad
\sum_{k=L+1}^{\infty}c_k\le B_L\quad(L+2>x).
$$

Choose L≥M+1, so the lower sum contains at least the first omitted term when x>0. If E− and E+ enclose exp(−x), the following rational endpoints enclose the Poisson tail. All quantities multiplied here are nonnegative.

$$
E_-\sum_{k=M+1}^{L}c_k\ \le\ \tau_M\ \le\ E_+\left(\sum_{k=M+1}^{L}c_k+B_L\right),\qquad
E_-\le e^{-x}\le E_+.
$$

The calculation chooses L until B_L is at most2^−256 times c_(M+1). This controls the unsummed positive series relative to a nonzero omitted term. The complete interval also includes the exponential enclosure's uncertainty; the relative series target alone is not a promise of256 relative bits in every result.

The exponential uses directed256-bit bounds. Bare coefficients, conditional vectors and geometric remainders use exact rational arithmetic. A tail much smaller than a fixed-point quantum can remain positive because it is formed by multiplying an enclosed exponential by an exact tiny rational factor. Rounded numbers and approximate magnitude logarithms are display tools, not the certificate.

## Interpret broad intervals honestly

A small M at large x may retain almost none of the probability mass. The component bounds are still valid, but can be too broad to resolve a probability or its nonzero sign. The reader must show this uncertainty rather than replacing it by zero or a seemingly precise midpoint.

For x=100 and M=0, the retained mass is exp(−100). Nearly all events are omitted. Raising M improves the same physical query; it does not increase elapsed time. Choosing a larger dominating nu can require a larger M even though the exact physical law is unchanged.

A structural identity can provide additional knowledge: with Q=0 the exact law is the initial law, and a uniform initial law remains stationary on an active ring. A generic truncation certificate can be broad even when such a separate exact argument determines the answer. Distinguish limitations of a numerical representation from uncertainty about the specified model.

## Separate potential ticks, actual changes and occupancy

Because this ring has the same exit rate r at every state, a potential tick is accepted as a true move with probability r/nu, independent of the current state. Poisson thinning gives an actual jump count of mean rt. The two expectations agree only when no fictitious events are introduced, including the trivial zero-clock case.

$$
\mathbb E[K_t]=\nu t,\qquad \mathbb E[J_t]=(a+b)t.
$$

Neither count is the probability of occupying state1. A particle can make three clockwise moves and return to its starting point. A holding-time survival probability concerns making no actual jump, while a return probability includes paths with several jumps.

For state-dependent exit rates, the acceptance probability depends on the present state. The actual count generally ceases to be a homogeneous Poisson count with one rate. The next unit will also show why uniform frequencies at jump instants need not imply uniform occupation over physical time.

## Use the experiment controls

The reader admits rates0..5/s in each direction, dominating clock0..10/s, physical time0..10s, three nonnegative initial weights with positive sum, and cutoff0..256. These are interaction and computation limits, rather than limits on the theory or the course's length. Strict integer numerators and positive denominators describe rational quantities. An inadmissible clock must be rejected while preserving the last valid preparation.

| Comparison | Change | Read and explain |
| --- | --- | --- |
| One-way ring, two clocks | Keep Q,t,mu0 fixed; use nu=1 and2 | Conditional period changes; the complete semigroup does not |
| Small versus adequate cutoff | Keep x=100 fixed; compare M=0 and256 | Retained mass, omitted tail and component interval widths |
| Tiny omitted mass | Use x=10^−12 and M=256 | Positive tail certificate despite an unavailable linear display |
| Only fictitious events | Set a=b=0 and nu>0 | Potential count grows; actual jump count stays zero; exact state law is unchanged |

Read the generator ring, conditional-law table, Poisson weights, retained mass and omitted-tail plots, and certified component brackets. Each horizontal coordinate is named. The physical-time scan updates ensemble laws; it does not sample a particle path.

## Worked self-checks

### 1. Construct an admissible potential-event kernel

For a=2,b=1, compare nu=3,nu=6 and nu=2. Which choices give probability kernels?

**Explanation.** At nu=3 the diagonal is zero and the directional probabilities are2/3 and1/3. At nu=6 they are1/2 for staying,1/3 clockwise and1/6 counterclockwise. Both rows sum to one with nonnegative entries and both recover the same Q. At nu=2 the diagonal is−1/2, so the probability interpretation fails.

### 2. Compare periodicity and physical relaxation

For a=1,b=0,nu=1 and initial state0, identify the state after six potential events. Does that determine the law at physical time six?

**Explanation.** Six deterministic clockwise updates return to0. At physical time six, Kt is a Poisson random variable rather than the fixed number six. Its three residue classes all have positive mass, and the exponential formula tends toward one third in each state. An event count and a time duration answer different questions.

### 3. Explain the mass at cutoff zero

Use a=1,b=0,nu=1,t=1,mu0=(1,0,0),M=0. Find the retained vector and its meaning after normalization.

**Explanation.** Only the zero-event term remains, so v0=(exp(−1),0,0), retained mass exp(−1) and tail1−exp(−1). Normalization yields(1,0,0), the state law conditional on no potential events. It is not the unconditional law at time one, which also includes paths ending in states1 and2.

### 4. Derive the component certificate

Why can adding the omitted total mass to every retained component give a valid upper bound even though the three upper bounds need not sum to one?

**Explanation.** Each omitted conditional vector has nonnegative components summing to one. Its contribution to any one state is at most its total weight. Summing over omitted counts bounds every component by tau_M. These are separate marginal bounds; they do not assert that all componentwise maxima can occur together. Intersecting an upper bound with one uses the known probability range, without renormalizing the vector.

### 5. Keep a very small tail distinct from zero

At x=10^−12 and M=256, why is a positive-series tail calculation more informative than subtracting a displayed retained mass from one?

**Explanation.** The first omitted coefficient is x^257/257!, strictly positive but far below binary64's linear display range. Subtraction of a rounded one gives zero. A positive rational coefficient times a positive exponential enclosure retains a positive lower bound, and the geometric remainder controls the unsummed terms. An unavailable decimal display then means a representation limit, not an impossible event.

### 6. Diagnose fictitious events and stationary initialization

With a=b=0,nu=2,t=3, how many actual jumps are expected? With an active ring initialized uniformly, does a stationary exact law imply that every finite Poisson truncation has mass one?

**Explanation.** The potential count has mean six, but R=I and the expected actual count is zero. The exact law remains its initialization. On an active ring, uniform initialization satisfies mu0 Q=0; every conditional vector remains uniform. Yet at x>0 a finite partial vector is s_M times that uniform law and has positive omitted mass. Exact stationarity and complete retention of numerical probability are separate statements.

The next unit replaces equal exit rates by state-dependent holding rates, then compares physical-time occupation with the chain observed only at actual jump instants.
