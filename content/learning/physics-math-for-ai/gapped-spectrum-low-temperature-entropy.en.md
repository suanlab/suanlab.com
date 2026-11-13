## Does the same temperature change always require the same energy?

Earlier we used a constant molar heat capacity for an ideal gas. That approximation cannot simply be extended down to absolute zero. Connecting to a finite low-temperature entropy requires examining temperature-dependent heat capacity. We will count states in a small system with an energy gap.

The system is in canonical equilibrium with a reservoir. It has g₀ ground states at energy zero and g₁ excited states at energy Δ≥0, with no other states included. The spectrum is independent of temperature. Degeneracy counts physically distinct states with the same energy.

## Start from individual state probabilities

At positive physical temperature T, let x=Δ/(kBT):

$$
Z=g_0+g_1e^{-x},\qquad q=\frac{g_1e^{-x}}{Z}.
$$

Each ground state's probability is 1/Z; each excited state's is e^(−x)/Z. q is the probability of the **entire excited group**. Multiplying by g₁ again after using q would count those states twice.

kB has units J/K, Δ has units J and T has units K, so x is dimensionless. Distinguish the equivalent gap temperature Δ/kB from the energy gap Δ. Calling an arbitrary AI score a physical energy requires additional definitions.

## Mean, entropy and heat capacity

Directly summing U=ΣpᵢEᵢ and S=−kBΣpᵢ ln pᵢ gives

$$
U=\Delta q,\qquad S=k_B(\ln Z+xq),\qquad
C_V=k_Bx^2q(1-q).
$$

Cᵥ here is the entire finite system's heat capacity in J/K, unlike the earlier ideal gas's **molar** heat capacity in J/(mol K). For this fixed-spectrum canonical system,

$$
\operatorname{Var}(E)=\Delta^2q(1-q),\qquad
C_V=\frac{\operatorname{Var}(E)}{k_BT^2},\qquad
\frac{dS}{dT}=\frac{C_V}{T}.
$$

Sensitivity of mean energy to temperature is related to equilibrium energy fluctuations. Do not apply this relation without review when energy levels themselves depend on temperature.

## Count a numerical example

Set g₀=g₁=1, Δ/kB=300 K and T=300 K. Then x=1 and q=1/(1+e)≈0.268941. U/kB≈80.6824 K, S/kB≈0.582203 and Cᵥ/kB≈0.196612. The last two ratios are dimensionless; multiply by kB to restore physical entropy and heat-capacity units.

At much lower temperatures the excited-state probability becomes exponentially small. The mean energy available to change with a small temperature adjustment also shrinks, making heat capacity tend to zero. This differs from extrapolating a constant high-temperature heat capacity to zero temperature.

## One ground state versus several

At fixed Δ>0, taking T→0 excludes excited states. The canonical limit assigns equal probability to the g₀ ground states:

$$
S(0^+)=k_B\ln g_0.
$$

This vanishes for g₀=1 but has a finite residual value for g₀>1. It is a specified canonical equilibrium limit. It does not determine preparation of an isolated system in one pure ground state, or a symmetry-breaking process.

Also distinguish a single finite system with fixed total ground degeneracy from N independent copies. The copies have g₀^N total ground states and residual entropy NkB ln g₀. When g₀>1 a residual per copy remains; this idealization can lie outside the strict low-temperature assumption of vanishing entropy density. Omitted interactions and degeneracy lifting in real matter must not be replaced by a universal conclusion about all materials.

## Zero gap and low temperature have different orders of limits

Setting Δ=0 first makes all g₀+g₁ states equal in energy. At positive T, entropy is kB ln(g₀+g₁) and heat capacity is zero. Holding Δ positive and taking T→0 first instead leaves kB ln g₀.

The difference is not a numerical error. The gap and the number of available ground states differ between the limiting problems. A figure must distinguish an exactly zero gap from a very small positive one.

## The boundary between the third law and cooling dynamics

Finite S(0) requires convergence of S(T)−S(0)=∫₀ᵀ Cᵥ(t)/t dt. A positive constant heat capacity extended down to zero makes this integral diverge. Classical constant-capacity extrapolation therefore has a low-temperature limit. Exponential suppression in a gapped model illustrates a different behavior.

A T=0 display denotes the equilibrium limit at fixed spectrum. It must not be described as cooling to absolute zero in finite time. Without reservoir coupling, cooling rates and control resources, this model does not provide a dynamical proof about attainability.

## Six questions with worked answers

### Exercise 1: Is q the probability of each excited state?

**Answer:** Only when g₁=1. q sums all g₁ excited probabilities; each individual probability is e^(−x)/Z.

### Exercise 2: Why not simply write x=Δ/T?

**Answer:** In SI, Δ is measured in J and T in K. kB supplies the energy-temperature conversion. A dimensionless convention setting kB=1 must be declared explicitly.

### Exercise 3: What entropy remains with two ground states?

**Answer:** This finite canonical equilibrium gives kB ln2. Distinguish it from an actually prepared pure state or the total degeneracy of N independent copies.

### Exercise 4: If Δ=0 and heat capacity is zero, must entropy be zero?

**Answer:** No. Equal state energies make mean energy independent of temperature while state mixing remains. S=kB ln(g₀+g₁).

### Exercise 5: Can the positive-gap and zero-gap low-temperature ends be joined as the same value?

**Answer:** No. The order of limits and ground-state count differ. A plot must not automatically identify their endpoints.

### Exercise 6: Does a vanishing heat capacity determine the time needed to reach absolute zero?

**Answer:** No. Equilibrium heat capacity and cooling dynamics provide different information. A process model with reservoir coupling and finite resources is also required.

## Distinguish small values from display range

An unrepresentable excited probability is not physical zero. Sign and logarithmic magnitude can retain information when the value itself is outside the number range. If its logarithm also exceeds that range, the logarithm of the negative log magnitude can describe its scale when representable. An unrepresentable layer remains unavailable. This does not promise arbitrary precision.

A ground-group probability displayed as one can still have a nonzero excited tail at positive temperature. Distinguish this from the exact zero excited probability of the T=0 equilibrium limit. Also inspect residual entropy and the excitation correction separately when total entropy rounds to the same displayed value.

Scaling a tiny gap and temperature together can leave x unchanged. Losing the gap's displayed SI energy must not erase its finite probabilities or heat capacity. Combine ratios and unit factors before projecting observations. Such extreme examples compare the declared model with numerical representation, rather than asserting experimentally accessible conditions in real materials.

## Read copy counts and derivative units

The probabilities describe groups in one copy. They are not probabilities that at least one of N copies is excited. Mean energy, entropy, heat capacity and independent energy variance scale by N; per-copy probabilities and normalized curves remain unchanged. Count large state numbers exactly as integers or compare their logarithms.

dS/dT has units J/K², unlike a time-dependent entropy production rate in W/K. Comparing equilibrium temperatures differs from solving heat transfer over physical time. T=0 denotes an equilibrium limit; do not substitute it into a positive-temperature derivative formula as 0/0.

## Research extension: cooling a growing number of independent copies

A single copy can be almost certainly in its ground state while an entire collection still has a substantial chance of an excitation somewhere. There are increasingly many places for a rare event to occur. We compare equilibrium systems whose size grows as their temperature falls; this differs from cooling one fixed finite system.

Specialize to one ground and one excited state per copy, a fixed positive gap Δ and N independent copies, with no interactions. Set δ=Δ/kB in K. Choose a fixed real a and sufficiently large integer N that x_N>0. Each N labels a different equilibrium system; this sequence is not a cooling protocol or an interacting thermodynamic phase transition.

Let the temperature decrease only logarithmically with N. The excited probability per copy tends to zero, but its product with the number of copies remains finite.

$$
\delta=\Delta/k_B>0,\quad x_N=\log N+a,\quad T_N=\delta/x_N,\quad q_N=\frac1{1+N e^a},\qquad \lambda=e^{-a}.
$$

The number M_N of excited copies has a binomial law. Its mean and variance approach the same positive λ. M_N is a count, not the per-copy occupancy plotted in the original lab.

$$
M_N\sim\operatorname{Binomial}(N,q_N),\qquad \mathbb E M_N=Nq_N\to\lambda,\quad\operatorname{Var}(M_N)=Nq_N(1-q_N)\to\lambda.
$$

All copies are in their ground states only if every independent preparation is ground. The limit of this probability is strictly between zero and one for finite fixed a. At a=0 it approaches e⁻¹≈0.367879 even though each individual ground probability approaches one.

$$
\Pr(M_N=0)=(1+e^{-x_N})^{-N}\longrightarrow e^{-\lambda}.
$$

For a fixed count m, write the binomial coefficient as a finite product, use Nq_N→λ and the previous zero-count limit. This gives the Poisson probabilities below. Large counts do not hide extra probability: the bounded mean gives a Markov tail bound, while the limiting Poisson probabilities sum to one.

$$
\Pr(M_N=m)=\binom Nm q_N^m(1-q_N)^{N-m}\longrightarrow e^{-\lambda}\lambda^m/m!\qquad(m\text{ fixed}).
$$

The total mean energy remains finite in this scaling. It is not the same limit as letting T tend to zero for a fixed number of copies, where mean energy vanishes.

$$
U_N=\Delta Nq_N\longrightarrow\Delta\lambda.
$$

Entropy counts the many possible locations of the finite expected number of excitations. Using N log(1+λ/N)→λ and Nq_N=λ+O(1/N) gives the centered limit below. Thus total entropy grows logarithmically even though entropy per copy vanishes.

$$
\frac{S_N}{k_B}=N\log(1+e^{-x_N})+x_NNq_N,\qquad \frac{S_N}{k_B}-\lambda\log N\longrightarrow\lambda(1+a).
$$

Compute heat capacity by differentiating at fixed N and fixed spectrum, then evaluate it at T_N. Differentiating along a sequence in which N also changes would give a different quantity. The actual finite-system heat capacity grows as the square of log N on this sequence.

$$
\frac{C_N}{k_B}=N x_N^2q_N(1-q_N),\qquad \frac{C_N/k_B}{(\log N+a)^2}\longrightarrow\lambda.
$$

Per-copy entropy and heat capacity both tend to zero, whereas total entropy diverges. This does not contradict the unique-ground-state limit at fixed finite N. The system size, state space and order of limits have changed; no universal statement about interacting matter follows.

$$
\frac{S_N}{Nk_B}\to0,\qquad \frac{C_N}{Nk_B}\to0,\qquad S_N/k_B\to\infty\quad(a\text{ fixed}).
$$

For a finite collection at a general positive x, a union bound supplies a simple sufficient temperature requirement for all-ground probability at least 1−ε. For N=1000 and ε=0.05 it requires x≥log(20000), or T≤300/log(20000) K when δ=300 K.

$$
\Pr(M_N\ge1)\le Nq_N\le Ne^{-x},\qquad x\ge\log(N/\varepsilon)\ \Longrightarrow\ \Pr(M_N=0)\ge1-\varepsilon.
$$

The exact independent-copy threshold is obtained by solving (1+exp(−x))⁻ᴺ≥1−ε. The expression below uses expm1 to avoid subtracting nearly equal exponentials. It is an equilibrium probability requirement, not a time or resource cost of cooling. The stated ε range makes the threshold positive for every N≥1.

$$
x\ge-\log\!\left[\operatorname{expm1}\!\left(\frac{-\log(1-\varepsilon)}{N}\right)\right],\qquad 0<\varepsilon<\tfrac12.
$$

### Read a rare tail across many copies

The upper panel compares the per-copy excitation probability with the probability of at least one excitation in the whole collection. The lower panel shows S_N/(kB log N) and C_N/[kB(log N)²], both approaching one at a=0. Those normalized curves hide growing totals; inspect the table for actual entropy and heat capacity.

![Per-copy excitation vanishes while collection-wide excitation remains probable; normalized entropy and heat capacity show distinct logarithmic growth.](/assets/learning/laws-growing-copy-low-temperature.en.svg)

### Follow the joint size–temperature sequence

Use δ=300 K and a=0. N is a count, T is in K, and S_N/kB and C_N/kB are dimensionless total quantities. No random samples are drawn; the table evaluates the exact finite canonical law.

| N | T_N (K) | Mean excited count | All-ground probability | S_N/kB | C_N/kB |
| --- | --- | --- | --- | --- | --- |
| 10 | 130.28834457 | 0.90909091 | 0.38554329 | 3.04636097 | 4.38173398 |
| 100 | 65.14417229 | 0.99009901 | 0.36971121 | 5.55460753 | 20.78971909 |
| 1000 | 43.42944819 | 0.99900100 | 0.36806330 | 7.90035476 | 47.62179179 |
| 10000 | 32.57208614 | 0.99990001 | 0.36789783 | 10.20936943 | 84.81340624 |

```python
from math import exp, log, log1p
for N in [10, 100, 1000, 10000]:
    x = log(N)  # a = 0, gap temperature delta = 300 K
    z = exp(-x)
    q = z/(1+z)
    ground = exp(-N*log1p(z))
    S = N*log1p(z)+x*N*q
    C = N*x*x*q*(1-q)
    print(f"{N} {300/x:.8f} {N*q:.8f} {ground:.8f} {S:.8f} {C:.8f}")
```

### Three research questions with explanations

#### 1. Does q_N→0 mean the full collection is in its ground state with probability tending to one?

**Explanation.** No. Nq_N tends to λ>0, so the probability of no excitation tends to exp(−λ). There are more opportunities for an excitation as each becomes rarer. At a=0 the collection has a roughly 63.2% chance of at least one excitation in the limit.

#### 2. Does growing total entropy at T_N→0 violate the fixed-system third-law limit?

**Explanation.** No. N grows and the state space changes along this sequence. For each fixed N with unique ground state, entropy tends to zero as temperature goes to zero. Here entropy per copy still tends to zero. Exchanging these different limits would erase the location uncertainty of rare excitations.

#### 3. Can dU_N/dT along the chosen N,T_N sequence replace canonical heat capacity?

**Explanation.** No. Canonical heat capacity differentiates mean energy with the number of copies and energy levels fixed. Changing N changes the system as well as temperature. The sequence has finite limiting mean energy while the fixed-N heat capacity evaluated on it grows; those are compatible statements about different derivatives.

### Source and research boundary

[David Tong, Statistical Physics, sections 1.3.1–1.3.2](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf) and [Classical Thermodynamics, section 4.5](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) provide fixed-spectrum canonical and low-temperature conventions. The joint sequence, finite-count limit and probability thresholds are derived here for independent copies. Numerical smallness, physical zero, large-system limits and achievable cooling remain separate concepts.

## Connection to AI research

Test normalization, mean energy, entropy and fluctuation relations as separate observations. An unrepresentably small excited probability is not necessarily physical zero. Conversely, residual entropy from exact degeneracy is not a rounding error to erase. Tracking state counts and orders of limits separates numerical behavior from model assumptions in new research.

Sources: [David Tong, Classical Thermodynamics](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf), [NIST constants](https://physics.nist.gov/cuu/Constants/Table/allascii.txt).
