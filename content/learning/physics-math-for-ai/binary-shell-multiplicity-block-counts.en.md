## A pattern and a meter reading are different objects

Picture four labeled switches. The pattern 1100 specifies which switches are on. A meter that reports only the number on reads 2. It gives the same reading for 1010 and for four other patterns. We have already met two descriptions: a microstate records the labeled configuration, while a macrostate records the chosen observation.

The distinction depends on the observation. A second meter reporting the number on in the left pair would distinguish 1100 from 0011. There is no universal list of variables that counts as a macrostate for every question. Here we choose a map, count its inverse images, and then specify a probability preparation. Counting alone does not determine what a real apparatus or evolving system samples.

## Define the finite model before assigning probability

There are N distinguishable, noninteracting sites. Each site has two states with energies 0 and ε, where ε>0 is fixed in joules. A microstate is a labeled binary vector x. Our macro observation K counts excitations:

$$
x=(x_1,\ldots,x_N)\in\{0,1\}^N,\qquad K(x)=\sum_{i=1}^N x_i,\qquad E(x)=\varepsilon K(x).
$$

N, the site labels and ε are fixed during a comparison. Because ε is positive, fixing K also fixes an energy shell in this particular model. At ε=0 every pattern would have the same energy, so an energy measurement would no longer identify K. The counting map could still be defined, but its interpretation as a distinct energy shell would change.

The calculator supports integer N from 0 through 4096 and block sizes and selected excitation counts within 0–N. This is a finite computational resource limit, not a definition of the thermodynamic limit. No reservoir temperature, transition rate or clock is supplied. The first reader example uses ε=10⁻²¹ J. An energy written as kB times a temperature specifies an energy scale; it does not by itself specify a thermal bath. The microcanonical preparation and entropy framework are introduced in [Tong, sections 1.2 and 1.3.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf).

## Count a shell by choosing its occupied positions

To obtain a configuration with K excitations, choose the K labeled positions that contain 1. The unchosen positions contain 0. Counting ordered selections first and removing the K! orderings of each subset gives the binomial multiplicity:

$$
\Omega_N(K)=\binom NK=\frac{N!}{K!(N-K)!},\qquad 0\le K\le N.
$$

The factors divide integers exactly. They do not mean the physical sites are indistinguishable: permutations that exchange an occupied and an empty labeled position produce different microstates. The division removes repeated descriptions of the same chosen subset, not the labels of the system.

A recurrence avoids constructing floating factorials. Complementing every bit gives the symmetry, and partitioning all binary vectors by their count gives the total:

$$
\Omega_N(k+1)=\Omega_N(k)\frac{N-k}{k+1},\qquad
\Omega_N(k)=\Omega_N(N-k),\qquad \sum_{k=0}^N\Omega_N(k)=2^N.
$$

For N=0 there is one empty vector, so the total and the unique shell multiplicity are both 1. This convention makes composition and normalization work without inventing an extra empty-system configuration.

## Enumerate four sites before drawing a smooth curve

For N=4, K=2, the six configurations are 1100, 1010, 1001, 0110, 0101 and 0011. Writing them out makes the lost label information visible. The whole count distribution is small enough to inspect:

| Excitations K | Multiplicity | Probability under uniform full-space preparation |
| --- | ---: | ---: |
| 0 | 1 | 1/16 |
| 1 | 4 | 4/16 |
| 2 | 6 | 6/16 |
| 3 | 4 | 4/16 |
| 4 | 1 | 1/16 |

The middle shell is the most likely meter reading under the stated preparation. None of its six individual configurations is more likely than 0000 under that same preparation: each individual vector has probability 1/16. A high bar for a macrostate can result from collecting many equally weighted microstates.

These bars are discrete. Connecting their centers can guide the eye, but it does not introduce fractional excitations or a trajectory moving between bars. A selected pattern and a selected shell are different queries and should have different labels in a visualization.

## Specify which uniform preparation you mean

There are two useful probability experiments. In the first, prepare any of the 2ᴺ vectors with equal probability. In the second, restrict preparation to vectors with a selected count K and make those vectors equally likely:

$$
p_{\mathrm{full}}(x)=2^{-N},\qquad
p_{{\rm shell},K}(x)=\begin{cases}\Omega_N(K)^{-1},&K(x)=K,\\0,&K(x)\ne K.\end{cases}
$$

The second distribution excludes other shells by its preparation constraint. That is a genuine support zero. It differs from an allowed state's positive probability that becomes too small for a numerical display.

Under the first preparation, summing the probabilities inside a shell gives:

$$
P_{\mathrm{full}}(K=k)=\frac{\Omega_N(k)}{2^N}.
$$

Uniformly choosing one of the N+1 possible counts would instead give probability 1/(N+1) to every shell. If preparation were then uniform inside that shell, a vector in shell k would receive 1/[(N+1)ΩN(k)]. That is a different, nonuniform distribution over vectors. Neither experiment is selected by counting alone. Equal a priori weights are a modeling assumption, not a consequence of the word microstate.

## Entropy of a shell and entropy of a preparation

For the selected shell, Boltzmann entropy counts its configurations. For an arbitrary finite preparation p, Gibbs entropy uses its probabilities. Information entropy H uses natural logarithms and has units of nats:

$$
S_B(K)=k_B\ln\Omega_N(K),\qquad H[p]=-\sum_xp(x)\ln p(x),\qquad S_G[p]=k_BH[p].
$$

Use the continuous extension 0 ln 0=0. SB and SG have units J/K; H is dimensionless information. The SI Boltzmann constant is exactly 1.380649×10⁻²³ J/K, while the calculator uses its binary64 approximation. The value and unit are listed in the [NIST CODATA constants table](https://physics.nist.gov/cuu/pdf/all.pdf).

For the uniform selected shell, substituting p=1/Ω gives SG=SB. The same multiplicity can support a nonuniform preparation with less entropy. Writing uK for the uniform selected-shell distribution, with p supported on that shell:

$$
S_B(K)-S_G[p]=k_BD(p\Vert u_K)\ge0.
$$

Thus a state count does not imply that every preparation attains the corresponding maximum entropy. A preparation concentrated on one pattern has H=0 even when many other patterns are compatible with the meter reading. Its missing conditional uncertainty must not be silently restored by an observer's grouping operation.

## Split uncertainty into the meter reading and its hidden details

Under the uniform full-space preparation, K is a deterministic function of the random microstate X. First naming the shell, then naming the vector inside it, names the original vector. The entropy chain rule becomes:

$$
H(X)=N\ln2=H(K)+\sum_{k=0}^NP_{\mathrm{full}}(k)\ln\Omega_N(k).
$$

One direct derivation substitutes p(x)=P(k)/ΩN(k) into−Σp ln p and groups terms by k. Each group contributes−P(k)lnP(k)+P(k)lnΩN(k). The first term is uncertainty about the meter reading; the second is the mean uncertainty remaining after that reading.

For four sites, H(X)=2.772589 nats. The count entropy is 1.407532 nats, and the average within-shell entropy is 1.365057 nats. Their sum recovers the full information. This average is not the selected K=2 shell entropy ln 6=1.791759 nats. Averaging across a preparation and conditioning on one selected shell are different operations.

## Divide a fixed shell between two blocks

Now fix a total excitation count K and divide the labeled sites into blocks A and B, with sizes NA and NB=N−NA. If A contains a excitations, B must contain K−a. The compatible configurations and integer support are:

$$
W(a\mid K)=\binom{N_A}{a}\binom{N_B}{K-a},\qquad
\max(0,K-N_B)\le a\le\min(K,N_A).
$$

Choose the occupied labels in each block independently for counting, then combine the two choices. Every full vector in the selected shell appears in exactly one such product. This gives Vandermonde normalization and the hypergeometric probability under the uniform selected-shell preparation:

$$
\sum_aW(a\mid K)=\binom NK,\qquad
P(A=a\mid K)=\frac{W(a\mid K)}{\binom NK}.
$$

For N=4, K=2 and NA=2, the compatible counts for a=0, 1, 2 are 1, 4, 1. The balanced split is more common because more labeled configurations realize it, not because one individual balanced pattern has an extra weight.

## Conditional fluctuations without an interaction

Introduce the indicator that a particular site in A is excited. Given the uniform fixed-K shell, its mean is K/N. For two distinct sites, counting simultaneous excitation gives K(K−1)/[N(N−1)]. Subtracting the product of means reveals negative covariance:

$$
\operatorname{Cov}(X_i,X_j\mid K)=-\frac{(K/N)(1-K/N)}{N-1}\quad(i\ne j,\ N>1).
$$

The global constraint creates dependence even though the Hamiltonian has no interaction term. Summing indicators in A gives the finite-population moments:

$$
\mathbb E[A\mid K]=N_A\frac KN,\qquad
\operatorname{Var}(A\mid K)=N_A\frac KN\left(1-\frac KN\right)\frac{N-N_A}{N-1}\quad(N>1).
$$

For N=0 the block is empty. For N=1, or an empty/full block, or K=0/N, the corresponding conditional variable is deterministic; use its exact variance 0 rather than inserting an undefined denominator. In the calculator, moments come from exact integer sums over the conditional distribution, and the reference calculation independently checks these formulas.

For N=8, K=4, NA=4 the counts are 1, 16, 36, 16, 1 with denominator 70. Mean A is 2 and variance is 4/7≈0.571429. The fixed total reduces fluctuations relative to treating the four site indicators as independent fair coins, which would have variance 1.

## Typical windows and exact peak values are different

The ratio of neighboring multiplicities shows where a bar grows and where it falls. Even N has one central maximum; odd N has two equal central maxima. These are exact integer ties. A tolerance for nearly equal floating bars is unnecessary.

Under uniform full-space preparation, K has mean N/2 and variance N/4. Consequently the excitation fraction concentrates in a fixed-width window around one half:

$$
\operatorname{Var}(K/N)=\frac1{4N},\qquad
P\!\left(\left|K/N-\tfrac12\right|\ge\eta\right)\le\min\!\left(1,\frac1{4N\eta^2}\right)\quad(N>0,\eta>0).
$$

A narrow fraction window can contain many discrete counts. Concentration in that window does not imply that the probability of the single central count tends to one. The Stirling comparison for even N=2 m illustrates the distinction:

$$
\frac{\binom{2m}{m}}{2^{2m}}\sim\frac1{\sqrt{\pi m}}\quad(m\to\infty).
$$

The calculator still uses exact finite counts, not this asymptotic formula. Already the central-count probability decreases from 6/16 at N=4 to 70/256 at N=8. Its fraction coordinate becomes more typical within a window while its exact discrete value becomes less probable. Neither statement supplies a mixing time or a thermodynamic-limit result for interacting matter.

## Keep a small probability separate from a numerical zero

Multiplicities are exact integers, serialized as decimal strings when necessary. The model multiplies each count by logarithmic factors before projecting the normalized result. Logs and entropies remain numerical approximations even though counts, normalization and conditional moments are exact.

A useful extreme case is N=1080, K=0. The shell has one configuration, yet its probability under uniform full-space preparation lies below the smallest positive binary64 value. Its information-entropy contribution can still be displayed:

$$
P_{\mathrm{full}}(0)=2^{-1080}>0,\qquad
-P_{\mathrm{full}}(0)\ln P_{\mathrm{full}}(0)=1080\ln2\,2^{-1080}>0,\qquad S_B(0)=0.
$$

The second quantity rounds to approximately 6×10⁻³²³ nats. The last is exactly zero because ln 1=0. The selected-shell conditional probability is exactly 1. A dash in the probability column, a small positive entropy contribution and a zero Boltzmann entropy are all consistent because they refer to different quantities and preparations. For larger counts or energies, retain sign and logarithmic evidence when a linear value is unavailable.

## What a different observation would require

The general structure is a map from a microstate space to observation labels, together with a chosen probability or reference measure. K is only one map. Counting block excitations reveals more information; forgetting the block labels reveals less. Nonuniform weights require conditional distributions instead of assuming 1/Ω inside every group.

Continuous phase-space variables require a reference measure and resolution; a volume is not an integer multiplicity without further choices. Quantum indistinguishability changes which configurations count as distinct. An automatic division by N! would alter this labeled-site model and is not a universal correction.

Noisy observations require a conditional channel. Their output entropy can include measurement noise, so it must not be called an increase of the original system's physical entropy. Subsequent families will make these distinctions explicit. In AI terms, a count or histogram is a representation that can discard labels: recovering those labels needs additional assumptions, rather than merely renaming the coarse representation a full state.

## Counts do not provide dynamics

A multiplicity curve compares possible configurations. It does not say which transitions are permitted, whether a trajectory explores a shell, how rapidly a block exchanges excitations, or whether coarse entropy increases monotonically. Equal weights in a preparation do not prove ergodicity of an unspecified evolution.

Changing K compares different constrained shells. Changing N compares systems of different size. Neither control is automatically a physical heating protocol. A thermodynamic interpretation of a changing distribution needs a system boundary, environment, allowed transfers and dynamical law. Keep these ingredients separate from the finite combinatorial calculation.

## Research extension: give the shell an exchange rule

A hypergeometric histogram can be stationary without describing how a preparation gets there. Add a specific random exchange and ask whether the block count itself has closed dynamics. This also lets us separate a correct mean, a stationary law and actual convergence.

Keep fixed N labeled binary sites, fixed K excitations and two nonempty blocks. At each integer attempt, with probability ρ choose one site uniformly from each block and swap their bits; otherwise do nothing. There is no energy change of the full fixed-K shell. The attempt index is not seconds, and this rule does not specify a physical exchange apparatus or its cost. The existing reader still computes counts rather than this new dynamics.

If block A has a excitations, increasing a requires an empty A site and an occupied B site. Decreasing it requires the opposite pair. Counting those pairs gives the two transition probabilities.

$$
\lambda_a=\rho\frac{(N_A-a)(K-a)}{N_AN_B},\qquad\mu_a=\rho\frac{a(N_B-K+a)}{N_AN_B},\quad N_B=N-N_A.
$$

An equal-bit swap or a deliberately idle attempt leaves the count unchanged. Every microstate with the same a has these same probabilities, so this particular count projection is exactly Markov for every preparation. Empty blocks would make the denominators invalid; those static counting cases remain outside this exchange rule.

$$
T_{a,a+1}=\lambda_a,\quad T_{a,a-1}=\mu_a,\quad T_{a,a}=1-\lambda_a-\mu_a.
$$

The original hypergeometric law is the stationary candidate on the feasible count support. The probabilities are defined only where the binomial counts are positive.

$$
W(a)=\binom{N_A}{a}\binom{N_B}{K-a},\qquad\pi_a=\frac{W(a)}{\binom NK}.
$$

The neighboring binomial ratios show equal stationary flux in both directions. This verifies stationarity but does not, by itself, prove convergence of an arbitrary starting law.

$$
\pi_a\lambda_a=\pi_{a+1}\mu_{a+1},\qquad\pi T=\pi.
$$

Subtract the downward probability from the upward probability. The conditional mean increment is linear in the current count, even though the distribution need not yet be stationary.

$$
\mathbb E[A_{t+1}-A_t\mid A_t=a]=\lambda_a-\mu_a=\rho\frac{N_AK-Na}{N_AN_B}.
$$

Iterating that linear expectation gives an exact mean relaxation factor. A preparation whose mean already equals the stationary mean can still have the wrong histogram, as the table shows.

$$
\bar a=N_AK/N,\quad\eta=1-\frac{\rho N}{N_AN_B},\qquad\mathbb E A_t-\bar a=\eta^t(\mathbb E A_0-\bar a).
$$

A minimal counterexample has one site in each block and one excitation. With ρ=1 every attempt flips the count. Uniform probability is stationary, but a point preparation oscillates forever. Counting and detailed balance alone have not supplied mixing.

$$
N=2,\ N_A=N_B=K=1,\ \rho=1:\qquad T=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

Adding a half-probability idle step removes that example's period. More generally 0<ρ<1 gives a self-loop and connects the feasible count chain; a nontrivial finite chain then converges. At K=0 or K=N the count is already a singleton. A physical time scale still has not been specified.

$$
N=2,\ N_A=N_B=K=1,\ \rho=\tfrac12:\qquad T=\begin{pmatrix}1/2&1/2\\1/2&1/2\end{pmatrix}.
$$

For the next entropy comparison, prepare microstates uniformly conditional on their block count. Symmetry within each block preserves this property under the exchange rule. Only with that extra preparation assumption can the full-state entropy be recovered from the count law and multiplicities.

$$
p_t(x)=\frac{P_t(a)}{W(a)}\quad(A(x)=a),\qquad H(X_t)=H(A_t)+\sum_aP_t(a)\log W(a).
$$

Data processing with the stationary transition kernel makes count-to-stationary KL nonincreasing. The displayed identity then makes full-state entropy nondecreasing, but count entropy alone need not increase. This probability statement does not identify a bath entropy-production budget or an equilibration time for a real device.

$$
D(P_{t+1}\Vert\pi)\le D(P_t\Vert\pi),\qquad H(X_t)=\log\binom NK-D(P_t\Vert\pi).
$$

### Read opposite entropy trends

The upper panel compares initially uniform counts with the hypergeometric target and the later count law. The lower panel separates full-state information entropy from count entropy. The latter starts at its maximum over five labels and decreases, while full-state entropy increases. A mean of two is unchanged throughout this example.

![A lazy exchange preserves a fixed energy shell; full-state entropy rises while count entropy falls toward the hypergeometric law.](/assets/learning/micro-macro-shell-exchange.en.svg)

### Reproduce a nonstationary law with the correct mean

Use N=8, K=4, equal blocks and ρ=1/2. Initially each of the five feasible counts has probability 1/5, with uniform microstates inside each count. Probability updates use exact fractions; logarithms are ordinary numerical approximations. All entropies below are in nats, or Gibbs entropy divided by kB for the full microstate law.

| Attempt t | Mean count | D to stationary | Count H | Full-state H | Total variation |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.00000000 | 0.81331805 | 1.60943791 | 3.43517719 | 0.37142857 |
| 1 | 2.00000000 | 0.32251129 | 1.52970116 | 3.92598396 | 0.25178571 |
| 2 | 2.00000000 | 0.12230314 | 1.41428792 | 4.12619210 | 0.16975446 |
| 10 | 2.00000000 | 0.00001738 | 1.14198895 | 4.24847787 | 0.00224315 |

```python
from fractions import Fraction as F
from math import comb, log
N, NA, K, rho = 8, 4, 4, F(1,2)
NB = N-NA
W = [comb(NA,a)*comb(NB,K-a) for a in range(5)]
pi = [F(w,comb(N,K)) for w in W]
T = [[F(0) for _ in W] for _ in W]
for a in range(5):
    birth = rho*F((NA-a)*(K-a),NA*NB)
    death = rho*F(a*(NB-K+a),NA*NB)
    if a < 4: T[a][a+1] = birth
    if a > 0: T[a][a-1] = death
    T[a][a] = 1-birth-death
P = [F(1,5)]*5  # uniform counts, uniform conditional microstates
for t in range(11):
    if t in [0,1,2,10]:
        mean = float(sum(a*p for a,p in enumerate(P)))
        D = sum(float(p)*log(float(p/q)) for p,q in zip(P,pi))
        Hcount = -sum(float(p)*log(float(p)) for p in P)
        Hmicro = Hcount+sum(float(p)*log(w) for p,w in zip(P,W))
        TV = float(sum(abs(p-q) for p,q in zip(P,pi))/2)
        print(f"{t} {mean:.8f} {D:.8f} {Hcount:.8f} {Hmicro:.8f} {TV:.8f}")
    P = [sum(P[a]*T[a][b] for a in range(5)) for b in range(5)]
```

### Three research questions with explanations

#### 1. Does the correct block mean certify equilibrium?

**Explanation.** No. The initially flat count law already has mean two, but differs from the weights (1,16,36,16,1)/70. Its positive KL and total variation expose that mismatch. The exact mean recurrence tests one observable, not the entire distribution.

#### 2. Why can count entropy decrease without contradicting the full-state result?

**Explanation.** The labels have different multiplicities. Full-state entropy includes the expected log multiplicity hidden within each count. The Markov rule and conditional-uniform preparation make that full entropy increase, while a flat distribution over count labels can become less flat.

#### 3. Why add an idle probability?

**Explanation.** The two-site always-swap chain is periodic despite having the correct stationary law. The half-idle example reaches its stationary distribution in one step. In a larger finite connected chain an idle probability removes periodicity, but it does not supply a physical exchange rate.

### Source and research boundary

The stationary-chain and laziness framework is described in [Levin and Peres, Markov Chains and Mixing Times](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf); KL contraction follows the [MIT 6.441 information-measure framework](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/184197ca5d5418da2415d37e929860b9_MIT6_441S16_chapter_2.pdf). The exchange probabilities, preparation and examples are derived here for the declared shell. No universal dynamics follows from multiplicity alone.

## Worked exercises

### Exercise 1: Name the six compatible patterns

For four labeled sites and K=2, list every compatible pattern. Give the probability of 1100 under the full-space preparation, the probability of observing K=2, and the conditional probability of 1100 after K=2 is known.

**Answer:** The patterns are 1100, 1010, 1001, 0110, 0101 and 0011. Their multiplicity is 6. The three requested probabilities are 1/16, 6/16 and 1/6. The last uses a different denominator because conditioning removes the other ten full-space patterns.

### Exercise 2: Replace the preparation

Choose a count uniformly from 0 through 4, then choose a compatible pattern uniformly. Compare the probabilities of 0000 and 1100. Is this the uniform full-space preparation?

**Answer:** The first is 1/5 because the K=0 shell has one pattern. The second is 1/(5×6)=1/30. They differ, so the preparation is not uniform over the 16 labeled patterns. The macro histogram is flat precisely because microstate weights were changed.

### Exercise 3: Split four sites into two pairs

Given N=4, K=2, NA=2, find the count and probability for each possible A, then its mean and variance.

**Answer:** A=0, 1, 2 have compatible counts 1, 4, 1 and probabilities 1/6, 4/6, 1/6. Mean A is 1. The variance is[(0−1)²+4(1−1)²+(2−1)²]/6=1/3. The finite-population formula gives 2×(1/2)×(1/2)×(2/3)=1/3 too.

### Exercise 4: Recover the hidden information

For uniform full-space preparation with N=4, use H(K)=1.407531741 nats and the average within-shell entropy 1.365056982 nats. How do these relate to SB at K=2?

**Answer:** Using the unrounded quantities, their sum is 4 ln 2 ≈ 2.772588722 nats, the full-vector entropy. In contrast, SB(2)/kB=ln 6=1.791759469 nats is conditioned on one selected shell. It is neither H(K) nor the average over shells. Multiply a dimensionless entropy by kB to express it in J/K.

### Exercise 5: Identify two peak counts

For N=9 under uniform full-space preparation, identify the most probable counts and their multiplicities. Does this select a privileged individual vector?

**Answer:** K=4 and K=5 are tied with multiplicity 126, so each macro probability is 126/512. Every individual vector still has probability 1/512. The tie is exact by bit-complement symmetry and the binomial recurrence; it is not a numerical equality threshold.

### Exercise 6: Interpret the rare empty-excitation shell

For N=1080, K=0, a display shows no linear shell probability, SB=0 and within-shell probability 1. Can these observations be correct together?

**Answer:** Yes. The full-space shell probability 2⁻¹⁰⁸⁰ is positive but underflows. Its entropy contribution 1080 ln 2·2⁻¹⁰⁸⁰ rounds to a positive value near 6×10⁻³²³ nats. The shell contains one pattern, so lnΩ=0 and conditioning on it gives probability 1. The unavailable value must retain positive sign and logarithmic evidence.

## Explore the experiment

Choose “Four sites, six patterns” and read three separate rows: one full-space pattern, the selected shell, and one pattern within that shell. Their values are 1/16, 6/16 and 1/6. The three denominators refer to different preparations or conditioning; the large K=2 bar does not give pattern 1100 an extra weight.

| Example | Compare | What the comparison shows |
| --- | --- | --- |
| Two-block fluctuations | Block counts (1,16,36,16,1), total 70, variance 4/7 | Fixing the total excitation count creates conditional dependence without an interaction energy. |
| Odd-size tied modes | The K=4 and K=5 shell counts | The tie is exact, while every labeled pattern still has the same full-space probability. |
| Empty system | Multiplicity 1, entropy 0 | There is one empty configuration, rather than no configuration. |
| Rare single-pattern shell | Shell probability, entropy contribution and SB | A missing positive probability, a representable tiny contribution and an exact zero can coexist. |
| Large exact counts | Sampled bars and the paged count tables | The graph selects labels for readability; the tables retain all exact integer counts. |

Try changing N, K and the block size one at a time. Reducing N also clips K and the block size to the new valid range; check those settings before comparing results. The gap ε is an energy in joules, not a reservoir temperature. The bar ordinate shows probability with an automatically chosen scale: read its tick labels rather than assuming the highest bar has probability one. Horizontal movement reveals the whole diagram on narrow screens.

The experiment, guides and quizzes are available in this reader. Calculations run in a worker; a pending result is not the result of the newly selected preparation. The example scan waits for calculations and compares preparations, not physical time. Use stop/retry for a long calculation and reset to return to the first preparation. Continue with weighted grouping to distinguish a state's count from its conditional weight.
