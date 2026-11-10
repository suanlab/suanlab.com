# Vanishing heat capacity and residual low-temperature entropy

> Manuscript draft. The new product model, figures and registration do not exist yet. This is finite-system canonical equilibrium, not cooling dynamics.

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

## Connection to AI research

Test normalization, mean energy, entropy and fluctuation relations as separate observations. An unrepresentably small excited probability is not necessarily physical zero. Conversely, residual entropy from exact degeneracy is not a rounding error to erase. Tracking state counts and orders of limits separates numerical behavior from model assumptions in new research.
