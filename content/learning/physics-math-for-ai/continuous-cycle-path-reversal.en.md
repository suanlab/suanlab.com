## A stationary room can still have one-way traffic

Imagine three rooms connected by doors. A photograph tells us the fraction of people in each room. A video also tells us which door they use, how often they leave and how long they wait. Equal room populations in successive photographs do not imply that the video looks the same backward.

The previous discrete cycle used a uniform stationary law. Here the rooms can have unequal populations and different waiting times. We will derive the stationary law, construct the actual reversed dynamics and compare complete paths. The calculation uses a specified path; it does not generate random samples or estimate a current from a handful of jumps.

## Fix rates, units and a row convention

Let the clockwise rates be a for0→1, b for1→2 and c for2→0. Let the opposite rates be d for1→0, e for2→1 and f for0→2. All six are strictly positive, constant and measured in s⁻¹. Our probability vector is a row:

$$
Q=\begin{pmatrix}-(a+f)&a&f\\d&-(d+b)&b\\c&e&-(c+e)\end{pmatrix},
\qquad \dot p(t)=p(t)Q,\qquad r_i=-Q_{ii}.
$$

A rate is a conditional hazard per unit time, rather than the probability of a jump in one second. Given state i, the probability of no departure for t seconds is exp(−r_i t). Conditional on departure, the next state is j with probability q_ij/r_i. There are no physical self-jumps associated with the diagonal of Q.

Positive rates make this finite chain irreducible and give a unique positive stationary law. We initialize it in that law. All states are even under time reversal: reversing a room label leaves the label unchanged. Momentum-like odd variables, hidden channels, zero rates and time-dependent protocols require further analysis.

## Solve stationarity without assuming uniform occupancy

At stationarity, total influx equals total outflux at every room. We solve πQ=0 together with normalization. One convenient expression is

$$
\tau_0=cd+de+bc,\quad \tau_1=ae+ac+ef,\quad
\tau_2=bf+ab+df,\qquad
Z=\tau_0+\tau_1+\tau_2,\quad\pi_i=\tau_i/Z.
$$

For example, the three terms in τ0 correspond to the directed trees1→0,2→0;1→0,2→1; and1→2,2→0. Each tree provides a route from both other states into0. You can also verify the result directly by multiplying π by Q. The tree weights and Z have units s⁻²; dividing them produces dimensionless probabilities.

Take(a,b,c,d,e,f)=(2,3,4,1,1,1)s⁻¹. The weights are17,11,10, so π=(17,11,10)/38. Uniform initialization would not be stationary for these rates. The model starts with the derived π rather than silently replacing it with(1/3,1/3,1/3).

## Distinguish gross flow, net current and detailed balance

The stationary one-way flow through i→j is F_ij=π_i q_ij. Its unit is probability per second. The net current compares both directions:

$$
J_{ij}=\pi_iq_{ij}-\pi_jq_{ji},\qquad
J_{01}=J_{12}=J_{20}=\frac{abc-def}{Z}.
$$

Why are the three oriented currents equal? Conservation at room1 gives J01−J12=0, and conservation at room2 gives J12−J20=0. A common nonzero circulation therefore changes no occupancy.

In the numerical example, the0→1 and1→0 flows are34/38 and11/38 per second, giving J=23/38 per second. The other two doors have different gross flows but exactly the same net current. Drawing equally thick arrows for all gross flows would misrepresent the model.

Detailed balance requires equality separately at every door. In this positive three-state family it is equivalent to abc=def. Equality of q_ij and q_ji is unnecessary: a more heavily occupied room can balance a smaller departure rate against the opposite flow.

## Use nonuniform reversibility as a worked counterexample

Set(a,b,c,d,e,f)=(2,3,1,1,2,3)s⁻¹. The stationary law is(1/6,1/3,1/2). At door0↔1, the two flows are(1/6)×2=(1/3)×1=1/3 per second. The other doors also balance. Here abc=def=6s⁻³, although the two rates at each door differ.

| Model | Stationary law | Circulation | Interpretation |
| --- | --- | --- | --- |
| All six rates1s⁻¹ | Uniform |0 | Reversible and symmetric |
| Rates(2,3,1,1,2,3)s⁻¹ |(1,2,3)/6 |0 | Reversible with unequal rates |
| Clockwise3, opposite1s⁻¹ | Uniform |2/3s⁻¹ | Same histogram, irreversible circulation |
| Rates(2,3,4,1,1,1)s⁻¹ |(17,11,10)/38 |23/38s⁻¹ | Nonuniform and irreversible |

The histogram alone cannot identify which generator produced it. Conversely, once all six rates are specified, the stationary law and currents are determined; there is no freedom to choose a different current for that same generator.

## Construct the reversed generator

For a stationary process with even state labels, the reversed rate is determined by stationary joint probabilities:

$$
q^*_{ij}=\frac{\pi_jq_{ji}}{\pi_i}\quad(i\ne j),\qquad
q^*_{ii}=-\sum_{j\ne i}q^*_{ij}=q_{ii},\qquad
\pi_i P_{ij}(t)=\pi_jP^*_{ji}(t).
$$

The equality of escape rates follows from πQ=0. Thus reversed dynamics have the same stationary occupancy and the same holding-time distribution at each state, while their destination choices may change. A transpose alone works only when the stationary reference is uniform.

In the irreversible example, q*01=(11/38)×1/(17/38)=11/17s⁻¹, not q10=1s⁻¹. Also q*02=40/17s⁻¹. Their sum is3s⁻¹, equal to the original escape rate a+f. If Q*=Q, all stationary door flows balance. Applying this stationary reversal twice returns Q.

## Write the density of a path including its waits

Specify states i0,…,in and their dwell times δ0,…,δn. Pre-jump waits are positive; the final wait may be zero when observation ends at a jump. The duration is T=Σδk. With stationary initialization, the ordered-jump-time path density is

$$
w_Q[\gamma]=\pi_{i_0}\left(\prod_{k=0}^{n-1}q_{i_ki_{k+1}}\right)
\exp\left[-\sum_{k=0}^{n}r_{i_k}\delta_k\right].
$$

For n jumps the density has units s⁻ⁿ. An exact continuous-time path has zero point probability; density describes small neighborhoods of jump times. The displayed log densities use1s as the time reference. Changing that reference changes both log densities by the same offset for fixed n, leaving their difference unchanged.

Reversal changes both the state sequence and the dwell sequence. For a time-independent Q, the sum of escape rate times dwell is identical after reversal. This cancellation fails if one reverses only the state list while leaving different dwell times attached to the wrong states.

## Separate endpoint evidence from jump evidence

Compare a path with its time-reversed path under the same Q and stationary initial law. Waiting factors cancel, giving

$$
R[\gamma]=\log\frac{w_Q[\gamma]}{w_Q[\gamma^{\leftarrow}]}
=\log\frac{\pi_{i_0}}{\pi_{i_n}}
+\sum_{k=0}^{n-1}\log\frac{q_{i_ki_{k+1}}}{q_{i_{k+1}i_k}}
=\sum_{k=0}^{n-1}\log\frac{\pi_{i_k}q_{i_ki_{k+1}}}{\pi_{i_{k+1}}q_{i_{k+1}i_k}}.
$$

The last form telescopes the intermediate stationary probabilities. It is also log(w_Q[γ]/w_Q*[γ]) for this stationary even-state setting. Distinguish the two comparisons: one reverses a path under Q; the other retains the path and changes its generator to Q*.

In the nonuniform reversible example, path0→1→2 has a jump-rate ratio(2×3)/(1×2)=3. Its endpoint ratio is(1/6)/(1/2)=1/3. Their product is1, so R=0. Dropping the endpoint term would falsely diagnose irreversibility in a reversible chain.

A path0→1→0 has ratio1 in any positive-rate model: both rates occur once in each direction and the endpoints agree. A zero value on that one path therefore does not prove detailed balance of the generator.

## Compare a cycle, its reverse and ensemble irreversibility

For a completed clockwise cycle, the endpoint factor is1 and the ratio is abc/def. Define its dimensionless cycle affinity A. At stationarity the mean log path ratio per unit time is

$$
A=\log\frac{abc}{def},\qquad
\sigma=\sum_{i<j}(F_{ij}-F_{ji})\log\frac{F_{ij}}{F_{ji}}
=J A\ge0,\qquad \mathbb E_{\pi}[R_T]=T\sigma.
$$

Each undirected-edge contribution is nonnegative because(x−y)log(x/y)≥0 for positive x,y. In the three-state ring the oriented net currents agree and the product of flow ratios telescopes, producing J A. There is no extra factor1/2 when each undirected edge is counted once.

For rates(2,3,4,1,1,1), A=log24≈3.17805 and σ=(23/38)log24≈1.92356s⁻¹. A chosen clockwise cycle has R=log24; its reversed cycle has R=−log24. This negative individual value is consistent with a nonnegative ensemble mean. A specified path is not an estimate of that mean.

Here σ is a stationary path relative-entropy rate. Connecting it to dissipated heat, temperature or chemical work needs an explicit physical model and local detailed balance. We have not supplied those inputs. The [Seifert review](https://arxiv.org/pdf/1205.4176), Sections6.1–6.2, provides the rate/path framework; the present three-state formulas and numerical examples are direct calculations.

## Read the controlled comparisons correctly

Multiplying every rate by κ>0 preserves π and every cycle rate ratio. It multiplies J and σ by κ and makes the physical clock κ times faster. The same state sequence has the same R. With fixed dwell times, however, its individual path density changes; matching the physical process also requires dividing dwell times by κ.

Changing dwell allocation for a fixed state sequence changes its forward density when escape rates differ. The forward/reversed ratio R still depends only on endpoints and directed jumps here, because the correctly reversed waits cancel. This is a statement about this time-independent model, not a rule that waiting times never contain useful dynamical information.

The22 preparations include unequal reversible rates, opposite cycle directions, a reversible/irreversible pair sharing a histogram, backtracking, zero duration, an endpoint jump, near balance, clock rescaling and64-jump extreme ratios. A positive ratio as small as10⁻³⁸⁴ must not become a structural zero; its exact fraction and log near−884.193 remain available. An enormous ratio similarly retains its logarithm when its Number projection is unavailable.

The selected-path time occupation is not generally π. At T=0 it is undefined. For a long stationary chain observed only at jump epochs, the reference occupancy is proportional to π_i r_i, so fast-leaving rooms are overrepresented. Regular time observations, jump-only records and a hand-picked demonstration answer different questions.

## Connect the lesson to AI and research

A model trained only to match stationary histograms can miss directional dynamics. Transition data, observation intervals, waiting times and state definitions matter when identifying a generator. Matching a target law also does not by itself establish sampling efficiency; compare observables, dependence and cost at fixed physical or computational budgets.

Use the room/door diagram to read occupancy, gross flows and net currents separately. Use the path strip to associate each wait with a room, then reverse both lists. Read endpoint and jump terms before their sum. Uniform arrow lengths should communicate direction only; numerical magnitudes belong in an explicit scale or table.

Research extensions include finite-interval transition observations, hidden channels, coarse observations and odd time-reversal variables. A coarse record can hide currents; recovering microscopic dissipation from it requires assumptions and an observation model. These questions, general networks and physical entropy production remain part of the curriculum beyond this first family.

## Exercises with explanations

### 1. Verify a nonuniform stationary law

For rates(2,3,4,1,1,1), compute π and check balance at state0.

**Explanation.** The tree weights are17,11,10 and Z=38. Outflux from0 is(17/38)×3=51/38s⁻¹. Influx is(11/38)×1+(10/38)×4=51/38s⁻¹. Occupancy is stationary even though the0↔1 flows differ.

### 2. Reject a transpose shortcut

Find q*01 for the same model. Why is Q transpose not the reversed generator?

**Explanation.** q*01=π1q10/π0=11/17s⁻¹. Q transpose would give1s⁻¹. Its row0 sum is−3+1+4=2s⁻¹, violating generator conservation. The π factors are essential.

### 3. Retain the endpoint term

For the reversible rates(2,3,1,1,2,3), compare0→1→2 with2→1→0.

**Explanation.** The rate-product ratio is3, while the stationary endpoint ratio is1/3. Reversing the dwell sequence preserves its escape integral. The full density ratio is1, giving R=0 despite unequal rates and endpoint probabilities.

### 4. Explain a negative path value

The irreversible example has J>0. Can the selected path0→2→1→0 have R<0?

**Explanation.** Yes. Its ratio is def/abc=1/24, so R=−log24. Irreversibility constrains the ensemble mean σ≥0; it does not require each sample path to point in the more likely direction.

### 5. Rescale the clock

Multiply all six rates by10. Which of π,J,A,σ and the ratio for a fixed state sequence change?

**Explanation.** Tree weights and Z multiply by100, preserving π. The current numerator multiplies by1000, so J multiplies by10. A and the path ratio remain unchanged; σ=J A multiplies by10. Fixed-time path densities differ. To describe the rescaled trajectory, divide waits by10 too.

### 6. Avoid two false completion tests

Does R=0 for0→1→0 prove detailed balance? Does a finite chosen-path occupation prove the stationary law?

**Explanation.** Neither. Backtracking cancels the two directional rate factors in any positive chain, even when abc≠def. A chosen path can dwell mostly in one room; its occupation is a demonstration, not an unbiased stationary estimate. Verify all door flows or the cycle criterion for this family, and state how any empirical path was sampled.
