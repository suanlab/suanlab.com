## A reservoir has a finite budget

Imagine four labeled switches with exactly two on. You can see the first two switches; the other two form a reservoir. If both visible switches are on, both hidden switches must be off. The reservoir cannot supply an unlimited number of excitations, and its possible configurations depend on what you observe in the visible part.

This is a useful starting point for ensembles. An ensemble specifies a probability preparation together with constraints. A microcanonical preparation fixes total energy and assigns equal probability to the allowed microstates in this model. A canonical preparation instead assigns a distribution of subsystem energies, parameterized by an inverse temperature. Calling a hidden block a reservoir does not automatically make the visible block exactly canonical.

Our task is to derive the visible distribution by counting, construct a canonical comparison with the same mean, and measure what that comparison misses. Read the earlier binary-shell and conditional-block chapter first. No physical clock, transition rates or proof of equilibration are assumed here.

## State the system and its constraints

There are n distinguishable subsystem sites and b distinguishable reservoir sites. Every site has energy 0 or the same positive gap ε, measured in joules. Sites do not interact in the energy model. Write N=n+b and fix the integer total excitation count K:

$$
x\in\{0,1\}^{N},\qquad A(x)=\sum_{i=1}^{n}x_i,\qquad \sum_{i=1}^{N}x_i=K,\qquad E_{\rm s}=\varepsilon A,\quad E_{\rm tot}=\varepsilon K.
$$

The uniform preparation is over labeled configurations with this fixed K. It is not uniform over the possible values of A. The reservoir excitation count is K−A; the constraint couples the two counts even though the energy has no interaction term.

The finite calculator accepts nonnegative integer block sizes with 1≤N≤4096 and 0≤K≤N. The empty subsystem and empty reservoir are meaningful boundary cases; an empty total system is outside this calculator's contract. The resource bound is not a physical bound or a thermodynamic-limit theorem. Statistical-mechanical ensemble terminology and the usual reservoir argument are introduced in [Tong, sections 1.2–1.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf); the finite comparisons below follow directly from our specified counting experiment.

## Derive the marginal before approximating it

For a particular visible binary pattern with a excitations, the reservoir must complete it with K−a excitations. There are C(b,K−a) completions. There are C(n,a) distinct visible patterns with that same count:

$$
a_{\min}=\max(0,K-b),\qquad a_{\max}=\min(n,K),\qquad c_a=\binom n a\binom b{K-a},\qquad \sum_{a=0}^{n}c_a=\binom NK.
$$

A binomial coefficient outside its allowed integer range is zero. The last identity partitions all complete fixed-K configurations by the observed value of A. Dividing by the total number of allowed complete configurations gives two related but different probabilities:

$$
p(x_{\rm s})=\frac{\binom b{K-a}}{\binom NK},\qquad P(A=a)=\frac{c_a}{\binom NK},\qquad a=A(x_{\rm s}).
$$

The first is a probability on labeled subsystem patterns. The second is a probability on count readings. Patterns within a given count have equal probability, but patterns at different counts need not. This distinction will matter again when comparing entropies.

## Enumerate the smallest informative example

Take n=2, b=2, K=2. Of the six complete patterns, 0011 gives visible 00, 1100 gives visible 11, and the other four give one visible excitation. Each visible one-excitation pattern has two reservoir completions.

| Visible pattern | Reservoir completions | Probability of this visible pattern |
| --- | --- | ---: |
| 00 | 11 | 1/6 |
| 01 | 01, 10 | 2/6 |
| 10 | 01, 10 | 2/6 |
| 11 | 00 | 1/6 |

The count probabilities are therefore 1/6, 4/6 and 1/6. A count bar at 1 combines two different visible patterns. A diagram should show that aggregation explicitly rather than drawing four equally likely visible patterns.

This preparation can also be understood as conditioning independent bits. Start with any common success probability r strictly between 0 and 1 for all N sites. Every complete pattern with K excitations then has the same product probability. Conditioning on the total count cancels this common factor:

$$
\Pr_r(X=x\mid\textstyle\sum_iX_i=K)=\frac{1}{\binom NK}\quad\text{for }\sum_i x_i=K,\qquad 0<r<1.
$$

Thus a fixed-total preparation is exactly a conditioned product preparation. Removing the conditioning changes the preparation; it does not become an identity just because a block is called a bath.

## Build a matching canonical comparison

Choose ρ=K/N. The comparison prepares the n visible sites independently with success probability ρ. For 0<ρ<1 it has a canonical form with t=βε=ln[(1−ρ)/ρ]. Its pattern and count laws are:

$$
q(x_{\rm s})=\rho^a(1-\rho)^{n-a},\qquad Q(A=a)=\binom n a\rho^a(1-\rho)^{n-a},\qquad \rho=K/N.
$$

This comparison is chosen to match one-site occupancy and mean subsystem energy. It is not fitted to every feature of the finite reservoir. When ρ<1/2, t is positive and corresponds to positive temperature for ε>0. At half filling t=0 is the infinite-temperature limit, not a finite temperature of zero. At ρ>1/2, t is negative: a bounded two-level spectrum admits this mathematical population-inverted preparation. It is not an ordinary positive-temperature kinetic-energy gas.

At ρ=0 or 1, use the corresponding deterministic distribution directly. These are limiting canonical preparations rather than finite β values. Do not evaluate undefined endpoint logarithms or replace true support zeros by small probabilities.

## Derive the mean and the missing correlations

Exchangeability of all N labeled sites gives a mean occupancy ρ at every site. Summing the n visible occupancies gives the same count mean under both laws:

$$
\mathbb E_P[A]=\mathbb E_Q[A]=n\rho,\qquad \mathbb E_P[E_{\rm s}]=\mathbb E_Q[E_{\rm s}]=\varepsilon n\rho.
$$

For distinct sites i and j, their joint occupancy in the fixed-total preparation is K(K−1)/[N(N−1)]. Subtracting ρ² gives a negative covariance when 0<ρ<1. Summing variances and covariances yields, for N≥2:

$$
\operatorname{Cov}_P(X_i,X_j)=-\frac{\rho(1-\rho)}{N-1},\qquad
\operatorname{Var}_P(A)=n\rho(1-\rho)\frac{b}{N-1},\qquad \operatorname{Var}_Q(A)=n\rho(1-\rho).
$$

The correction b/(N−1) records depletion of the finite budget. It equals one for a one-site subsystem, but is smaller for a larger subsystem unless the variance is already zero. Independence under Q is an additional preparation assumption, not a consequence of noninteracting energies. For N=1 the distribution is deterministic and its variance is calculated directly.

## Compare whole distributions, not only averages

For the four-site example, the canonical comparison uses two fair independent bits:

$$
(P_0,P_1,P_2)=(1,4,1)/6,\qquad (Q_0,Q_1,Q_2)=(1,2,1)/4,\qquad \mathbb E[A]=1.
$$

The finite reservoir concentrates more mass near one visible excitation. Its variance is 1/3, while the canonical variance is 1/2. An experiment that checks only average energy cannot see this discrepancy.

A second hand example has n=2, b=4, K=2 and ρ=1/3. The exact count probabilities are (6,8,1)/15, while the matching canonical probabilities are (4,4,1)/9. Both means are 2/3; the variances are 16/45 and 4/9. Here the comparison has the finite positive inverse-temperature parameter t=ln 2.

A useful visualization places the two normalized count histograms side by side, marks their identical mean, and labels the two variances separately. An energy axis multiplies count values by ε; probability bars do not change when only the energy unit is rescaled.

## Inspect the reservoir slope locally

The exact probability ratio between particular visible patterns differing by one excitation depends on the reservoir's remaining budget. For adjacent allowed counts, compare one pattern at a+1 with one at a:

$$
\frac{p(x_{\rm s}^{(a+1)})}{p(x_{\rm s}^{(a)})}=\frac{K-a}{b-K+a+1},\qquad
\frac{P_{a+1}}{P_a}=\frac{n-a}{a+1}\frac{K-a}{b-K+a+1}.
$$

The count ratio includes a visible multiplicity factor. Confusing the two ratios makes a degeneracy effect look like a different Boltzmann weight. In the positive-temperature example the pattern ratios are 2/3 and 1/4. Neither is the constant canonical pattern ratio 1/2 at both steps.

This is why one finite reservoir generally cannot be assigned a single exact exponential weight over every possible energy exchange. Finite differences of reservoir entropy vary with the chosen energy. A large smooth reservoir can make that variation small over an appropriate window; its size and the window must be checked.

## Measure event errors with total variation

Define total variation on the same count space, including exact-zero entries:

$$
\operatorname{TV}(P,Q)=\frac12\sum_{a=0}^{n}|P_a-Q_a|=\max_{B\subseteq\{0,\ldots,n\}}|P(B)-Q(B)|.
$$

For n=2, b=2, K=2, the event A=1 attains the difference 1/6. Total variation answers a concrete question: what is the largest error in the probability of any count-based event? It is not a percentage error in every individual bar; rare events can have large relative discrepancies even when TV is small.

Keep n=2 and half filling while increasing b through even values. With N=b+2, direct substitution gives an exact error sequence:

$$
P_1=\frac{N}{2(N-1)},\qquad P_0=P_2=\frac{N-2}{4(N-1)},\qquad \operatorname{TV}(P,Q)=\frac{1}{2(N-1)}.
$$

| Reservoir sites b | Total sites N | TV | Exact count variance |
| ---: | ---: | ---: | ---: |
| 2 | 4 | 1/6 | 1/3 |
| 6 | 8 | 1/14 | 3/7 |
| 18 | 20 | 1/38 | 9/19 |
| 98 | 100 | 1/198 | 49/99 |

Every row still has mean 1. Specify how K changes when changing N: here K=N/2. Increasing the reservoir while holding a small K fixed is a different experiment whose density tends to zero.

## Keep the direction of relative entropy

Relative entropy compares preparation laws in an ordered direction:

$$
D(P\Vert Q)=\sum_{a:P_a>0}P_a\ln\frac{P_a}{Q_a},\qquad D(Q\Vert P)=\sum_{a:Q_a>0}Q_a\ln\frac{Q_a}{P_a}.
$$

For this matching comparison every allowed P outcome is also allowed under Q, so the forward divergence is finite. The reverse divergence is infinite if Q assigns positive mass to a count that the fixed-total reservoir forbids. This is a support statement, not a numerical overflow.

For n=2, b=0, K=1, P is concentrated at A=1 but Q=(1,2,1)/4. The forward divergence is ln 2 and the reverse divergence is infinite. Smoothing the forbidden counts would change the physical preparation. Likewise, a positive canonical probability below the linear floating-point range remains positive; a logarithmic display must not silently turn it into a forbidden count.

## Understand the usual reservoir expansion

Let a smooth approximation to the reservoir entropy be S_b(E_b)=kB ln Ω_b(E_b). Expand around E_b*=E_tot−E_s*. With ΔE=E_s−E_s*, fixed reservoir parameters and a finite nonzero temperature and heat capacity, the local expression is:

$$
\ln\Omega_b(E_{\rm tot}-E_{\rm s})\approx\frac{S_b(E_b^*)}{k_B}-\frac{\Delta E}{k_BT_b}-\frac{(\Delta E)^2}{2k_BT_b^2C_b}+\cdots.
$$

The linear term produces an exponential energy weight. The quadratic term shows why a reservoir with finite heat capacity changes its slope when it exchanges appreciable energy. For positive C_b, this correction penalizes departures from the expansion point relative to the linear extrapolation. Omitting it requires control of the relevant energy window and higher terms.

The exact finite model uses discrete energy shells. A derivative requires a specified smooth approximation; it is not an exact derivative of a finite table. Do not use this expansion at the half-filled infinite-temperature limit or at an endpoint without separate limiting analysis. The exact count comparison remains well defined there.

## State precisely what becomes close

For a particular ordered subsystem pattern with a ones, the exact marginal can also be written with falling factorials:

$$
p(x_{\rm s})=\frac{(K)_a(N-K)_{n-a}}{(N)_n},\qquad (z)_j=z(z-1)\cdots(z-j+1),\qquad (z)_0=1.
$$

For fixed n, N tending to infinity and K/N tending to ρ, each finite product approaches ρᵃ(1−ρ)ⁿ⁻ᵃ. There are only finitely many subsystem patterns, so their entire distribution, and hence the count distribution, converges in total variation to the corresponding product law. Endpoint densities lead to deterministic limits.

This argument fixes n. If the subsystem grows with the reservoir, correlations can accumulate. For example, at half filling with comparable n and b, the ratio of count variances does not approach one. Local observables, extensive fluctuations and full probability laws can have different approximation requirements. No general ensemble-equivalence theorem for interacting or long-range systems follows from this finite binary argument.

## Connect the preparation question to entropy and AI

The matching product preparation maximizes entropy on subsystem patterns with the specified mean count. Both p and q are uniform within each count, so their conditional pattern laws agree. This gives a useful identity:

$$
H(q)-H(p)=D(p\Vert q)=D(P\Vert Q),\qquad H(q)=n h(\rho),\qquad H(p)=H(P)+\sum_a P_a\ln\binom n a,\quad h(r)=-r\ln r-(1-r)\ln(1-r).
$$

The entropy H(P) of the count reading is not the pattern entropy H(p). The extra multiplicity term restores the labels hidden by the observation. Endpoint terms use their continuous zero limits. The identity explains the maximum-entropy gap without claiming that observed count entropy must always be maximized by the binomial law under a mean-only constraint on counts.

For AI, compare fixed-budget sampling with independent Bernoulli masking. The same expected number of selected items does not imply the same correlations or the same distribution of selected counts. A normalized fixed-budget data pipeline can violate an independence assumption used by a likelihood. The physical calculation supplies a precise example of that modeling issue; it does not establish a temperature or heat cost for an arbitrary learning algorithm.

## Read numerical evidence without changing the preparation

The calculator retains integer configuration counts and rational normalization before projecting probabilities onto a display. A count of zero means the event is impossible under that preparation. A positive count with a very large denominator can instead produce a probability too small for the ordinary numeric field; retain its sign and logarithm. These cases have different consequences for reverse KL.

Logarithms, entropy and divergence are numerical approximations. A direct floating-point sum of positive and negative KL terms can lose a small positive answer through cancellation. An equivalent nonnegative representation, when the target support contains the source support, is:

$$
D(P\Vert Q)=\sum_{a:Q_a>0}Q_a f(P_a/Q_a),\qquad f(z)=z\ln z-z+1,\qquad f(0)=1.
$$

The linear corrections cancel because both laws normalize. This expression also needs care near z=1, where evaluating the formula by naive subtraction loses precision; a local expansion can preserve the small positive contribution. It does not remove support infinities. The numerical implementation uses that positive form and bounded-significand information terms with unrestricted exponent, while keeping the probability laws exact until observation.

For independent checks, the reference calculations use integer combinatorics, rational probabilities and high-precision logarithms; small cases additionally enumerate complete bit configurations. Compare exact counts and moments as well as rounded chart heights. The largest accepted N is an implementation budget, and the absence of a physical clock remains part of the model even if a UI cycles through examples.

## Research extension: identify a finite budget from two readings

One coin can hide a finite budget. If half of a large collection of switches is on, any one inspected switch is fair, regardless of the collection size. A second simultaneous reading asks a different question: did the first on switch reduce the remaining budget? The answer can identify a finite reservoir within a specified preparation model.

Keep a uniform fixed-K shell of N labeled binary sites, N≥2, and observe two distinct sites from one preparation. Repeat the complete preparation independently when collecting data. These are simultaneous site observations, not two successive times or a reservoir relaxation experiment. The existing reader continues to compare theoretical distributions rather than draw these repeated samples.

A matching independent preparation Q has exactly the same one-site law. Even infinitely many independent one-site observations cannot distinguish two candidate shell sizes having the same density p.

$$
p=K/N,\qquad P(X_1=1)=P(X_2=1)=p,\qquad Q(X_1=1)=Q(X_2=1)=p.
$$

Count completions after fixing the first two labeled bits. For example, two ones consume two of the K excitations. These four pattern probabilities sum to one and include genuine forbidden patterns when a budget is too small.

$$
P_{11}=\frac{K(K-1)}{N(N-1)},\quad P_{10}=P_{01}=\frac{K(N-K)}{N(N-1)},\quad P_{00}=\frac{(N-K)(N-K-1)}{N(N-1)}.
$$

Subtract the product of the two marginal means. Negative covariance comes from the fixed total rather than an interaction energy. This derivation requires the declared uniform shell; an unrelated anticorrelated sensor model need not have a reservoir of this size.

$$
c=P_{11}-p^2=-\frac{p(1-p)}{N-1},\qquad 0<p<1.
$$

Conditional prediction makes depletion intuitive: the first observed one leaves K−1 excitations among N−1 sites. Under Q the corresponding conditional probability remains p.

$$
P(X_2=1\mid X_1=1)=\frac{K-1}{N-1}=p-\frac{1-p}{N-1},\qquad K>0.
$$

With the exact joint law and 0<p<1, invert the covariance to recover N, then check the integer K constraint. This is structural identification inside the model, not a claim that a finite dataset determines N exactly.

$$
N=1+\frac{p(1-p)}{-c},\qquad K=Np\in\mathbb Z,\qquad c<0.
$$

Rows index X₁=0,1 and columns X₂=0,1. The same amount moves out of both agreeing patterns and into both disagreeing patterns; this preserves the single-site means.

$$
P-Q=\begin{pmatrix}c&-c\\-c&c\end{pmatrix},\qquad Q=\begin{pmatrix}(1-p)^2&p(1-p)\\p(1-p)&p^2\end{pmatrix}.
$$

The disagreement event attains the total-variation difference. Counting just A=X₁+X₂ also preserves this TV because 01 and 10 have the same sign of discrepancy. Ordered pattern and count probabilities still use different multiplicities.

$$
\operatorname{TV}(P,Q)=\tfrac12\sum_{a,b}|P_{ab}-Q_{ab}|=-2c=\frac{2p(1-p)}{N-1}.
$$

At half filling, every single-site reading is fair but agreeing pairs are depleted. N must be even for an integer K=N/2. For N=4 the pair law is (1,2,2,1)/6 rather than four equal quarters.

$$
p=\tfrac12,\ d=(N-1)^{-1}:\quad P_{00}=P_{11}=\tfrac14(1-d),\quad P_{01}=P_{10}=\tfrac14(1+d).
$$

For half filling this forward KL shrinks quadratically while TV shrinks linearly in d. At N=2 use the continuous 0 log 0 limit. Different discrepancy measures therefore express different sensitivities; a small value does not certify exact independence.

$$
D(P\Vert Q)=\tfrac12[(1-d)\log(1-d)+(1+d)\log(1+d)]=\tfrac12d^2+O(d^4)\quad(d\to0).
$$

Near independence, inversion is ill-conditioned: a small covariance error can produce a large size error. At p=0 or 1 every shell is deterministic, c=0 and the inverse is undefined. A nonnegative empirical covariance cannot simply be inserted and accepted as a physical size.

$$
\left.\frac{\partial N}{\partial c}\right|_p=\frac{p(1-p)}{c^2},\qquad p\in(0,1).
$$

### Read the comparison

Compare ordered pair probabilities at N=4 and N=20 with the same independent limit. The lower panel shows why the signal gets weaker as the total budget grows: TV and the covariance magnitude vanish even though the one-site mean never changes.

![Identical one-site probabilities hide different agreeing and disagreeing pair laws; finite-reservoir dependence weakens with total size.](/assets/learning/ensembles-reservoir-identification.en.svg)

### Reproduce the example

The following exact-fraction calculation counts reservoir completions for four half-filled sizes. N recovered from the exact probabilities is not a finite-sample estimate. The covariance and TV are dimensionless.

| N | p | P11 | Covariance | Pair TV | Recovered N |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.50000000 | 0.16666667 | -0.08333333 | 0.16666667 | 4 |
| 8 | 0.50000000 | 0.21428571 | -0.03571429 | 0.07142857 | 8 |
| 20 | 0.50000000 | 0.23684211 | -0.01315789 | 0.02631579 | 20 |
| 100 | 0.50000000 | 0.24747475 | -0.00252525 | 0.00505051 | 100 |

```python
from fractions import Fraction as F
from math import comb
for N in [4,8,20,100]:
    K = N//2
    P = {(a,b):F(comb(N-2,K-a-b),comb(N,K))
         for a in [0,1] for b in [0,1]}
    p = P[1,0]+P[1,1]
    c = P[1,1]-p*p
    Q = {(a,b):(p if a else 1-p)*(p if b else 1-p) for a,b in P}
    tv = sum(abs(P[key]-Q[key]) for key in P)/2
    recovered = 1+p*(1-p)/(-c)
    print(f"{N} {float(p):.8f} {float(P[1,1]):.8f} "
          f"{float(c):.8f} {float(tv):.8f} {int(recovered)}")
```

### Three research questions with explanations

#### 1. Why is one site insufficient?

**Explanation.** Its Bernoulli law contains only p. For example, N=4,K=2 and N=20,K=10 both give probability one half. Independent repetitions of that same marginal cannot reveal the within-preparation dependence that was never observed.

#### 2. Does nonzero covariance prove an interaction energy?

**Explanation.** No. The energy remains additive over sites. Conditioning on the fixed total creates dependence. An independence assumption and an additive Hamiltonian are different statements.

#### 3. Does the inverse formula give an exact sample-based size?

**Explanation.** No. It identifies the parameter from an exact law under the shell model. Sampling error is amplified when c is near zero, and other preparation mechanisms can have the same pair covariance. Model checks and uncertainty analysis remain necessary.

### Source and research boundary

The ensemble background is [Tong, sections 1.2–1.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf). The pair-law inversion, exact event discrepancy and conditioning argument here are direct calculations for the declared finite shell. They do not identify an arbitrary reservoir from one measured covariance.

## Practice with answers and a verification checklist

Use exact fractions before decimals. A comparison should state its sample space, constraints, matching rule, error observable and approximation conditions.

### Exercise 1: Reconstruct the visible law

For n=2, b=2, K=2, find the probability of visible pattern 01 and the probability A=1. Explain why they differ.

**Answer:** Pattern 01 has reservoir completions 01 and 10, giving 2/6=1/3. Count A=1 also includes visible 10, so its probability is 4/6=2/3. The factor two counts visible labels; it is not an extra reservoir probability.

### Exercise 2: Detect dependence with one conditional

In the same preparation, calculate the probability that the second visible site is on given that the first is on. Compare with the matching canonical preparation.

**Answer:** After fixing the first site on, one excitation remains among three equally treated sites. The conditional probability is 1/3. The marginal probability is 1/2. In the independent canonical preparation both the conditional and marginal probabilities are 1/2. The fixed-total constraint causes dependence without an interaction energy.

### Exercise 3: Test a positive-temperature comparison

For n=2, b=4, K=2, calculate both count distributions, the common mean, the two variances and TV. Give t for the comparison.

**Answer:** P=(6,8,1)/15 and Q=(4,4,1)/9, with common mean 2/3. Their variances are 16/45 and 4/9. The signed differences are (−2,4,−2)/45, so TV=4/45. The comparison uses ρ=1/3 and t=ln 2. Matching this mean does not make either variance or the adjacent exact pattern ratios match.

### Exercise 4: Set a count-event accuracy target

For n=2 at half filling, how large must an even reservoir b be to make TV strictly less than 0.01? State how K is chosen.

**Answer:** The exact requirement 1/[2(b+1)]<1/100 gives b+1>50. The smallest allowed even b is 50, N=52 and K=26. Then TV=1/102. Holding K=2 instead would not test the same half-filled comparison family.

### Exercise 5: Explain an infinite divergence

For n=2, b=0, K=1, find both distributions and the two KL divergences. Is the infinite result a failure of the probability model?

**Answer:** P=(0,1,0), Q=(1,2,1)/4. Forward KL is ln[1/(1/2)]=ln 2. Reverse KL is infinite because Q has positive mass on counts 0 and 2 while P forbids them. Both distributions normalize. The infinity reports incompatible support in that direction, not failed normalization or an instruction to add smoothing.

### Exercise 6: Separate count entropy from pattern entropy

For n=2, b=2, K=2, express H(p), H(q) and their difference using exact logarithms. Check the multiplicity contribution.

**Answer:** The four pattern probabilities are (1,2,2,1)/6. Their entropy is ln 6−(2/3)ln 2, while H(q)=2 ln 2. The count entropy is ln 6−(4/3)ln 2, and adding P(A=1)ln 2 restores H(p). The gap is (8/3)ln 2−ln 6, equal to D(P∥Q). It is not H(q)−H(P), because those two entropies concern different sample spaces.

Before trusting a visualization, verify normalization and the exact mean, check the finite-population variance factor, retain forbidden outcomes as true zeros, and label any out-of-range positive probability. Distinguish independent sample frames from physical time. The later canonical-fluctuation and particle-exchange chapters will change the preparation conditions explicitly; interaction, continuous phase space and the thermodynamic limit require further work.

## Connect the calculation to the experiment

Start with **Two sites and a small reservoir**. Read P(A|K) and Q(A) as different preparation laws over the same horizontal count A. The three bars describe probabilities, not three measured trajectories. Before reading the numerical table, predict which law is more concentrated around A=1; then check the equal mean, unequal variances and TV=1/6.

| Experiment | Keep fixed or change | What to explain |
| --- | --- | --- |
| Larger reservoir at the same density | Keep n=2 and K/(n+b)=1/2; change b and K together | The mean stays 1 while Var P approaches Var Q and TV decreases |
| Fixed excitation without a reservoir | Set b=0 with n=2 and K=1 | P forbids A=0,2; reverse KL is infinite even though forward KL is finite |
| Empty subsystem | Set n=0 while the total system remains nonempty | A=0 is the only outcome; both laws agree with zero variance |

The input fields specify counts n,b,K. They do not specify an energy gap ε. Consequently the displayed moments are count moments, not SI energy moments, and matching the density specifies only the dimensionless canonical parameter t. To convert a count mean to joules, supply a fixed ε and multiply by ε; multiply the variance by ε². A calibrated temperature additionally needs that gap and the inverse-temperature relation, including its endpoint limitations. Do not read the integer K as a temperature control.

Use the paged distribution table to inspect all counts when the figure selects only some labels. Distinguish a true zero from a positive value outside the linear display range. Automatic comparison switches between independent preparations; it supplies neither reservoir dynamics nor a thermalization time.
