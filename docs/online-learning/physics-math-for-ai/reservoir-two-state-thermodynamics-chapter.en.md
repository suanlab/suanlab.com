# Two-reservoir heat flow: the same state path, different entropy

Authoring manuscript, 2026-11-08. The numerical model and chapter are being prepared; this unit is not yet registered in Interactive Book.

## 1. Can a population snapshot reveal heat flow?

Imagine two rooms,0 and1, connected by two doors. Reservoir A causes transitions through door A; reservoir B causes transitions through door B. A detector that records only room numbers loses the identity of the door used.

The average populations can remain constant, with zero total current between rooms, while heat passes between reservoirs. Read a stationary snapshot, an unlabelled film and a reservoir-labelled film as different observations.

## 2. Specify the energy and two temperatures

Set E₀=0,E₁=ε. A positive reference temperature T* and positive ratio r define ε=kBT* ln r. Reservoir temperatures are T_A=T*/b_A and T_B=T*/b_B. Here b is a dimensionless inverse-temperature ratio, with different units from the physical inverse temperature1/(kBT) in the preceding chapter.

A and B are names. If b_A<b_B, A is hotter; reversing the inequality makes B hotter. At r<1 the gap is negative and state1 lies lower. Increasing the state number does not necessarily increase energy.

Measure heat and energy first in kBT* and entropy in nats. Joules follow by multiplying by T* and the exact Boltzmann constant1.380649×10⁻²³J/K. Changing T* at fixed r and b rescales both the energy and the two bath temperatures.

## 3. Local detailed balance for each reservoir

Give each bath a positive clock ν_α and use these channel rates.

$$
\begin{aligned}
q_{01}^{\alpha}&=\frac{\nu_\alpha}{1+r^{b_\alpha}},
& q_{10}^{\alpha}&=\frac{\nu_\alpha r^{b_\alpha}}{1+r^{b_\alpha}},\\
\ln\frac{q_{01}^{\alpha}}{q_{10}^{\alpha}}
&=-b_\alpha\ln r=-\frac{\varepsilon}{k_BT_\alpha},
& \alpha&\in\{A,B\}.
\end{aligned}
$$

This relation is a physical modeling assumption. It does not automatically recover reservoir temperatures or heat from an aggregate rate fitted to state observations. We include no particle exchange, external work or internal state entropy in this family.

The clock sets the overall coupling scale. Temperature alone does not determine that coupling. Both channels remain positive; a weak coupling is distinct from an exactly disconnected channel.

## 4. The aggregate generator fixes the stationary law

Let a be the total0→1 rate and d the total1→0 rate. Add the channel rates before finding the stationary distribution.

$$
a=q_{01}^A+q_{01}^B,\qquad d=q_{10}^A+q_{10}^B,\qquad
\pi_0=\frac{d}{a+d},\quad \pi_1=\frac{a}{a+d}.
$$

Ensemble averages start from π. Occupancy and Shannon entropy H(π) are therefore constant. Choosing0 or1 as the initial state of an example path does not replace the ensemble initialization by a delta distribution.

A single bath alone would target occupancy1/(1+rᵇ). Two baths can compete with different target occupancies. The actual stationary law need not be canonical for either bath separately.

## 5. Channel currents cancel in the population ledger

Define J_A in the0→1 direction. The B current, defined in the same direction, is its negative.

$$
J_A=\frac{\nu_A\nu_B}{\nu_A+\nu_B}
\left(\frac1{1+r^{b_A}}-\frac1{1+r^{b_B}}\right),
\qquad J_B=-J_A.
$$

Going0→1 through A and1→0 through B returns to the same state but transports energy between reservoirs. A closed state path need not have zero heat exchange.

Two-way traffic also differs from net current. Compare π₀q₀₁ and π₁q₁₀ for each channel before taking their difference.

## 6. A reservoir's entropy change can be negative

Heat is positive when it enters the system. The gap is fixed and external work is zero, giving

$$
\frac{\dot Q_\alpha}{k_BT_*}=J_\alpha\ln r,\qquad
\dot s_\alpha^{\mathrm{bath}}=-b_\alpha J_\alpha\ln r,\qquad
\dot Q_A+\dot Q_B=0.
$$

A hotter bath loses heat and its entropy decreases. The colder bath receives the same energy at a lower temperature and gains more entropy. Nonnegativity applies to total production, rather than to every reservoir's exchange term separately.

Constant mean system energy does not imply that heat exchange has stopped. Energy can continuously pass through without accumulating.

## 7. Nonnegative total production

With F₀₁^α=π₀q₀₁^α and F₁₀^α=π₁q₁₀^α, total production has two equivalent expressions.

$$
\sigma=\sum_{\alpha}(F_{01}^\alpha-F_{10}^\alpha)
 \ln\frac{F_{01}^\alpha}{F_{10}^\alpha}
=(b_B-b_A)J_A\ln r\ge0.
$$

Each paired contribution follows (x−y)ln(x/y)≥0. Unlike the reservoir exchange terms above, these channel contributions also contain an allocation of system entropy. Their names and signs should be distinguished.

If b_A=b_B or r=1, each channel satisfies detailed balance and production is exactly zero. Otherwise two positive couplings leave positive production. A small positive value that looks rounded to zero is not the same equilibrium control.

## 8. The film without bath labels is reversible

After aggregation, stationary traffic in the two directions is equal. The production rate inferred from states alone is

$$
F_{01}=F_{10},\qquad
\sigma_{\mathrm{state}}=(F_{01}-F_{10})\ln\frac{F_{01}}{F_{10}}=0
$$

and the total state-only path log ratio is also zero. This does not establish zero physical production. Information was lost by erasing which reservoir caused each jump.

Two learned models with the same state-prediction accuracy need not have the same thermodynamic interpretation. Identifying heat needs additional observations or assumptions about reservoirs, transition labels and energies.

## 9. Read the missing information as conditional KL

Given an observed0→1 jump, let f_α=q₀₁^α/a be its bath distribution; for1→0 use g_α=q₁₀^α/d. Both are normalized distributions on two labels. With common aggregate traffic F=π₀a=π₁d,

$$
\sigma=F\left[D(f\Vert g)+D(g\Vert f)\right].
$$

This measures how the likely cause of a jump changes with its direction. The aggregate state film can be direction-symmetric while these conditional bath distributions differ.

It is a small example of the distinction between prediction and hidden-variable identification in AI. Fitting the state transition matrix alone does not determine the components of f and g. This information ledger requires observed or physically specified channels.

## 10. Heat along one marked path

Let Δx_j be the state change at jump j and α_j its reservoir. Pathwise heat obeys

$$
\frac{Q_\alpha}{k_BT_*}
=\ln r\sum_{j:\alpha_j=\alpha}\Delta x_j,\qquad
W=0,\qquad Q_A+Q_B=\varepsilon(x_T-x_0).
$$

Upward and downward jumps through the same channel cancel that channel's heat. A loop through different channels can transfer heat even though total system energy change is zero. State endpoints alone do not determine Q_A and Q_B separately.

The example path is constructed from a specified bath bit word and equal dwells, rather than sampled. Its survival term must use the sum of both channels' escape rates. Using only the bath responsible for the next jump would define a different path law.

## 11. Path reversal and an integral identity

Total path entropy contains both the stationary boundary term and the bath-resolved jump terms.

$$
S_{\mathrm{tot}}=\ln\frac{\pi_{x_0}}{\pi_{x_T}}
-\sum_j b_{\alpha_j}\Delta x_j\ln r
=\ln\frac{P(\text{bath word}\mid\text{state path})}
{P(\text{reversed bath word}\mid\text{reversed state path})}.
$$

The last equality uses zero state-only log ratio in this stationary binary process. Reverse the order of states and labels; do not exchange the temperatures assigned to A and B.

$$
\left\langle e^{-S_{\mathrm{tot}}}\right\rangle=1.
$$

This follows from normalization of the reversed stationary marked-path law. Averaging a few two-jump examples or cutting off a path sum at a maximum jump count is not an empirical test of it. We also make no claim to compute a general tilted-moment curve here.

The mathematical rate of the reversed film is π_jq_ji^α/π_i. Its aggregate generator is unchanged while channel allocations differ. This construction is distinct from physically swapping reservoir names or temperatures.

## 12. Four bath words for the same state path

Choose r=3,b_A=1,b_B=2,ν_A=ν_B=1/s and T*=300K. A is300K and B is150K. The stationary law is π=(33,7)/40.

| Channel | q₀₁ [s⁻¹] | q₁₀ [s⁻¹] | J [s⁻¹] |
| --- | ---: | ---: | ---: |
| A | 1/4 | 3/4 | 3/40 |
| B | 1/10 | 9/10 | −3/40 |

Total state current is zero; actual production is3ln3/40≈0.082396nats/s. Conditional on the state path0→1→0, the bath-word probabilities are as follows. These are not probabilities of an exact path with precisely specified jump times.

| Bath order | Conditional probability | S_tot [nats] |
| --- | ---: | ---: |
| AA | 25/77 | 0 |
| AB | 30/77 | ln3 |
| BA | 10/77 | −ln3 |
| BB | 12/77 | 0 |

AB absorbs ε from A and releases ε to B. BA carries the reverse heat transfer and negative production. Their state records are identical but their conditional probability ratio is3. A path such as AA or BB with cancelling heat is not a replacement for the ensemble production rate.

## 13. Exercises with explanations

**1. Does zero total state current imply zero heat current?** No. The example has J_A=3/40/s and J_B=−3/40/s. These cancel in the population ledger while transferring energy between different reservoirs.

**2. Is negative entropy change of the hotter A contradictory?** No. A loses heat. Entropy increases more in the colder B, leaving positive total production. Clipping each reservoir's exchange term to be nonnegative would alter the balance.

**3. Can observing0→1→0 distinguish AB from BA?** The state record alone cannot. Bath labels distinguish probabilities30/77 and10/77 and entropy values±ln3. Accurate state prediction does not guarantee physical identifiability.

**4. Does a BA path with negative entropy make the mean negative?** No. A path and an ensemble average are different observations. Changing only the example bath bits leaves mean curves and reservoir temperatures unchanged.

**5. Does r=1/3 reverse mean heat flow from cold to hot?** No. The signs of the energy gap and J_A change together. Their product εJ_A still denotes heat entering the system from hotter A. Separate state-number direction from energy direction.

**6. Is a bath word impossible when its numeric probability cannot be displayed?** No. A long rare word can have positive probability below binary64's display range. Inspect its exact rational bounds, sign and magnitude log. Exact zero and unavailable display are distinct.

## 14. Scope and further study

This family is a stationary weak-coupling two-state marked process with fixed energies and temperatures. Numerical inputs restrict b_A,b_B to integers1–8, example jumps to0–16 and time to0–4s. These are experiment bounds, not limits on curriculum volume or weeks.

Here we erased reservoir labels without merging the system states themselves. Memory under general state coarse-graining, unobserved internal energy, particle exchange, chemical potentials, strong coupling, feedback and finite-data estimation remain separate follow-up topics.

Continue with sectionII of [Esposito's original paper](https://arxiv.org/pdf/1112.5410) and [Seifert's review](https://arxiv.org/pdf/1205.4176) for reservoir-resolved rates and entropy balances. The binary model, conditional-label KL identity and worked examples here are derived directly under the stated assumptions.
