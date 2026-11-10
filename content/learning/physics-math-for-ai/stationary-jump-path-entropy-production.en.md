## 1. A stationary film can still have a direction

Film people moving between three rooms. The average population of each room stays constant. Now reverse the film. If people mostly circulate clockwise, the reversed film may look unlikely even though its population histogram is unchanged.

Stationarity describes one-time probabilities. Irreversibility compares probabilities of complete paths. The Shannon entropy of a stationary distribution can stay constant while the mean path irreversibility grows with observation time. We will compare a single path, an average rate and an exponential average; they are different quantities.

## 2. State the experiment before naming an entropy

Use a time-independent continuous-time Markov chain with positive rates a,b,c on 0→1→2→0 and d,e,f on the reverse edges. The probability law is a row vector and ṗ=pQ. Initialize in the unique stationary distribution π. States are even under time reversal: reversing the film does not change a state's label.

Every reverse edge is available. This support condition makes the logarithmic path ratios finite for paths with finitely many jumps. Time-dependent controls, odd variables, hidden states and missing reverse transitions require additional treatment. We have specified neither a heat bath nor energy levels, so our logarithmic entropy is in nats rather than heat divided by temperature.

## 3. Traffic, net current and dissipation are distinct

The stationary directed traffic is Fᵢⱼ=πᵢqᵢⱼ, in s⁻¹. A room can receive and send many people even when every edge has zero net current. In the three-state cycle the clockwise net currents agree:

$$
J=F_{01}-F_{10}=F_{12}-F_{21}=F_{20}-F_{02}
=\frac{abc-def}{Z}.
$$

Here Z is the sum of the three stationary tree weights from the earlier Markov chapter. Current is signed; activity is not the same observable. Increasing both directions equally can increase traffic while preserving detailed balance.

## 4. Obtain a nonnegative mean rate

Pair the two directions of each edge. The stationary entropy production rate is

$$
\sigma=\sum_{\{i,j\}}(F_{ij}-F_{ji})\ln\frac{F_{ij}}{F_{ji}}
=\sum_{i\ne j}F_{ij}\ln\frac{F_{ij}}{F_{ji}}.
$$

The first sum counts each unordered edge once; the second counts all directed edges. Do not insert another factor of two. For positive x,y, both x−y and ln(x/y) have the same sign, so each paired contribution is nonnegative. A single directed term or a single jump increment may still be negative.

For this one cycle, the stationary probabilities telescope from the product of flux ratios:

$$
\mathcal A=\ln\frac{abc}{def},\qquad \sigma=J\mathcal A\ge0.
$$

Affinity is dimensionless and σ has units nats/s. Reversing the signs of both current and affinity does not make their product negative. With positive rates, σ=0 is equivalent to detailed balance on this cycle.

## 5. Check a concrete driven example

For (a,b,c,d,e,f)=(2,3,4,1,1,1)/s, π=(17,11,10)/38 and J=23/38/s. The clockwise edge flux ratios are 34/11,33/10,40/17. Their product is24, so σ=(23/38)ln24≈1.92356 nats/s.

| Background | Stationary law | Mean rate |
|---|---|---|
| All rates1/s | (1,1,1)/3 | 0 |
| Clockwise3/s, reverse1/s | (1,1,1)/3 | 2ln3 nats/s |
| (2,3,1,1,2,3)/s | (1,2,3)/6 | 0 |

The first two rows have the same stationary histogram and Shannon entropy ln3. Their rates of path irreversibility differ. The third row shows that detailed balance does not require a uniform distribution or equal forward and backward rates.

## 6. A path includes the waiting times

A continuous-time path specifies its initial state, ordered jumps, jump times and final observation time T. Its density contains the initial probability, jump-rate factors and exponential survival factors for every dwell interval.

Under reversal by the same autonomous generator, each state's total dwell time is unchanged. The survival factors therefore cancel. This cancellation is a fact about the specified conjugate experiment, not a general license to omit waiting times when comparing different generators.

Including the stationary initial weights gives

$$
S_T=\ln\frac{dP}{d(P\circ\Theta)}
=\ln\frac{\pi_{X_0}}{\pi_{X_T}}
+\sum_{\rm jumps\ i\to j}\ln\frac{q_{ij}}{q_{ji}}
=\sum_{\rm jumps\ i\to j}\ln\frac{F_{ij}}{F_{ji}}.
$$

Θ reverses the entire path on [0,T]. The last equality telescopes the stationary boundary terms. Individual path densities have time units depending on their number of jumps; the ratio compares matching reversed paths and is dimensionless.

## 7. An open path needs its boundary term

One complete clockwise circuit returns to its original state, so its boundary term vanishes and S=𝒜. The reverse circuit gives−𝒜. An open path is different: its endpoints need not have equal stationary probabilities.

In the nonuniform detailed-balance example (2,3,1,1,2,3)/s, the open jump0→1 has ln(q01/q10)=ln2. Its boundary term is ln(π₀/π₁)=−ln2. The total S is0, as required by detailed balance. Reading only the rate ratio would incorrectly report nonzero total irreversibility.

For the driven example in section5, the same open jump has S=ln(34/11), not ln2. Stationary initialization does not mean the boundary term of every individual open path is zero; only its ensemble mean is zero.

## 8. A negative realization does not violate a positive mean

For the specified stationary experiment,

$$
\mathbb E[S_T]=T\sigma\ge0,\qquad
\mathbb E[e^{-S_T}]=1.
$$

A counterclockwise circuit in a clockwise-driven system has negative S and can occur. Such paths matter strongly in the exponential average. A histogram with few observed negative paths can be a poor estimator of the second relation even when the model satisfies it exactly.

The second identity follows by integrating the reversed path probability: the factor exp(−S) changes P into its normalized reversed law. It is not proved by renormalizing a truncated list of paths. Jensen's inequality gives the nonnegative mean when the expectations exist, but the explicit flux formula also identifies its rate and units.

## 9. The stationary adjoint makes the comparison concrete

The adjoint rates are

$$
q^\dagger_{ij}=\frac{\pi_jq_{ji}}{\pi_i}\quad(i\ne j),\qquad
q^\dagger_{ii}=q_{ii}.
$$

Stationarity ensures that the adjoint conserves probability and has the same exit rates. Its stationary law is π and its edge currents are reversed. It is generally not obtained by simply swapping the six scalar rates while keeping every other ingredient unchanged.

The path relative entropy between the original and adjoint processes has rate σ. Equal exit rates remove the survival contribution here. This links irreversibility to statistical distinguishability of dynamics, without assigning a heat bath to a formal transition network.

## 10. A homogeneous ring has an independent counting solution

Let every clockwise rate be α and every reverse rate β. The total exit rate is α+β in every state. Jumps form a Poisson clock whose direction marks are independent; thinning gives independent counts N₊∼Poisson(αT), N₋∼Poisson(βT). These statements use homogeneity and do not hold automatically for general state-dependent rates.

Since π is uniform, each clockwise jump contributes ln(α/β) and each reverse jump its negative:

$$
S_T=(N_+-N_-)\ln(\alpha/\beta),\qquad
M_T(z)=\exp\!\left\{T\left[\alpha((\alpha/\beta)^{-z}-1)
+\beta((\alpha/\beta)^z-1)\right]\right\}.
$$

For α=3/s and β=1/s, the mean is2(T/s)ln3 and the variance is4(T/s)(ln3)². A complete circuit has three jump increments, but an observed path need not complete a circuit. Do not replace its entropy by completed-cycle count alone.

## 11. A tilted operator computes an exponential average

For general positive rates, write Aᵢⱼ=ln(Fᵢⱼ/Fⱼᵢ). A tilted operator and its moment generating function are

$$
(L_z)_{ij}=q_{ij}e^{-zA_{ij}}\ (i\ne j),\qquad
(L_z)_{ii}=Q_{ii},\qquad M_T(z)=\pi e^{TL_z}\mathbf1.
$$

Usually L_z is not a probability generator. Its diagonal stays equal to Q's diagonal; forcing the row sums to zero would calculate a different quantity. At z=0 it is Q, and at z=1 it is the stationary adjoint. Thus M_T(0)=M_T(1)=1.

For this autonomous stationary even-state experiment, M_T(z)=M_T(1−z). In the homogeneous example, the closed counting expression shows this symmetry directly. General driving, incomplete observations or missing reverse support require revisiting the experiment and its hypotheses.

## 12. Keep mathematical and physical interpretations separate

Path entropy in nats is a well-defined statistical quantity for the stated model. Interpreting a jump-rate logarithm as physical reservoir entropy requires additional microscopic information, such as a suitable local detailed-balance relation, state energies and reservoir couplings. Multiplying by a constant does not supply that missing physical model.

For AI systems, a reversed event log can reveal directional statistical structure. It does not by itself measure the hardware's dissipated heat. Hidden states, sampling intervals, incorrect Markov assumptions and imperfect rate inference can all change what is observed. These are research questions rather than reasons to discard the path formulation.

## 13. Exercises with explanations

**1. Can two equal histograms have different entropy production rates?** Yes. The first two rows of the table both have uniform π, while their rates are0 and2ln3 nats/s. The histogram does not specify transition traffic or path probabilities.

**2. A reverse circuit has S=−ln24. Is σ negative?** No. The path is one allowed realization. The mean in the driven example is σ=(23/38)ln24>0. A jump or circuit contribution is not an ensemble rate.

**3. Why does the open equilibrium jump0→1 have zero total S?** Its rate contribution is ln2 and its stationary boundary contribution is−ln2. Their sum is0. Keeping only one term changes the observable.

**4. Check the homogeneous integral fluctuation relation.** Put z=1 in the counting formula: α(β/α−1)+β(α/β−1)=β−α+α−β=0. Hence M_T(1)=exp0=1 for every finite T, with no path truncation.

**5. Is P(S=0) just the no-jump probability?** No. For α≠β, equal positive clockwise and reverse counts also give S=0. The no-jump probability exp[−(α+β)T] is only one contribution. At α=β, every path has S=0, including paths with many jumps.

**6. What happens when all rates are multiplied by c>0?** π and affinity stay the same, J and σ multiply by c. At the shortened observation time T/c, the full path-increment distribution matches the original observation at T after rescaling the jump times. The film's physical clock matters even when its stationary picture does not change.

## 14. Scope and further study

This unit assumes autonomous, time-independent rates, stationary initialization, states even under time reversal and positive reverse transitions on every edge. Continue with sections6.1–6.2 of [Seifert’s author manuscript](https://arxiv.org/pdf/1205.4176) for path laws and stationary entropy production. The three-state ring and homogeneous count reference make these assumptions concrete.

The plots calculate ensemble averages for the specified stationary process. Exact identities at zero time, tilt0/1 or detailed balance have different evidence from propagation at other tilts. A few example paths, or a truncated path sum renormalized to one, do not constitute an empirical fluctuation test.

Interpreting physical heat and work requires additional energies, reservoirs and local detailed balance. Nonstationary balances, multiple reservoirs, hidden production, coarse graining, finite-data estimation, uncertainty relations and research applications continue throughout the full curriculum, with no week or volume limit.
