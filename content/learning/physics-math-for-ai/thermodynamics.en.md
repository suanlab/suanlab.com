## From microscopic probabilities to physical quantities

Statistical mechanics connects probabilities of microscopic states with macroscopic energy and entropy. This connection needs a physical ensemble and units. A generic neural-network loss called “energy” does not automatically acquire temperature, heat or thermodynamic meaning.

Consider equilibrium with a thermal reservoir, one ground microstate of energy zero and $g$ excited microstates each of energy $\Delta>0$. Use units with Boltzmann’s constant $k_B=1$, so temperature has the corresponding energy units. The finite model is exact under these assumptions; it contains no phase transition or spatial interactions.

## Define the canonical weights

A microstate of energy $E_i$ receives weight $e^{-E_i/T}$. The partition function sums these weights:

$$
Z=1+g e^{-\Delta/T},\qquad
p_0=1/Z,\quad p_j=e^{-\Delta/T}/Z\ (j=1,\ldots,g).
$$

The excited energy-level occupancy is $P_1=g e^{-\Delta/T}/Z$. It sums $g$ microscopic probabilities. This distinction is crucial when counting entropy.

| Quantity | Definition |
| --- | --- |
| $Z$ | Dimensionless normalization |
| $U$ | Mean energy $\Delta P_1$ |
| $F$ | Helmholtz free energy $-T\log Z$ |
| $S$ | Entropy $-\sum_i p_i\log p_i$ in units $k_B=1$ |

## Work the two-state reference

With $\Delta=T=1,g=1$, $Z=1+e^{-1}\approx1.367879$. Excited occupancy is $1/(1+e)\approx0.268941$, so $U\approx0.268941$. Free energy is about $-0.313262$ and entropy about 0.582203.

Substitute $\log p_i=-E_i/T-\log Z$ into entropy to derive

$$
S=U/T+\log Z=(U-F)/T.
$$

This identity follows from the specified canonical probabilities. It is not a free identification of every information-theoretic entropy with heat divided by temperature.

## Degeneracy changes what must be counted

For $g>1$, let $H_{\rm binary}$ be the entropy of ground versus excited **energy occupancy**. All excited microstates have equal conditional probability $1/g$, giving

$$
S=H_{\rm binary}+P_1\log g.
$$

Thus the binary occupancy distribution loses information about which excited microstate is present. The lab displays both quantities, allowing an explicit check. Equal energies do not make distinct microstates disappear from the state count.

## Temperature limits

As $T\to0^+$, the unique ground state dominates and entropy tends to zero in this model. As $T\to\infty$, all $g+1$ microstates become equally likely, excited occupancy tends to $g/(g+1)$ and entropy to $\log(g+1)$.

For $g=1$, high-temperature excited occupancy tends to 1/2, not 1. Raising temperature does not mean every system is certainly excited. The reservoir exchanges energy; the system’s instantaneous energy fluctuates even though its equilibrium probabilities are stationary.

## Read the plotted curves

Blue is excited-level occupancy versus temperature; orange is dimensionless entropy versus temperature. They are different quantities with explicitly stated axes, not two estimates of the same probability. The table evaluates the selected temperature, whereas the curves show a temperature sweep for the current gap and degeneracy.

The curve grid covers positive temperatures only. Low-temperature floating-point saturation is a numerical representation issue, not a proof of a new physical limit. Moments and free energy are computed from the finite-state formula rather than estimated from drawn samples.

## AI connections and research limits

Energy-based models use normalized exponential scores and partition functions. The algebra resembles canonical probabilities, but physical thermodynamics requires actual state energies, reservoir conditions and units. Sampling a learned distribution is not automatically a physical thermal experiment.

A useful research extension adds interacting degrees of freedom, then compares exact enumeration on a small system with sampling. Check normalization, mean energy and autocorrelation. Do not claim a finite two-level example reproduces a thermodynamic phase transition or establishes universal laws of learning.

## Connect temperature and degeneracy to the experiment

Use the three registered preparations before changing individual controls. Treat an excited energy level as a shelf with g separate places: adding places changes its total occupancy even though each place has the same energy. The shelf probability is a sum; a microscopic probability belongs to one named place.

| Action | Observation to inspect | Question to explain |
| --- | --- | --- |
| Compare ‘Two states’ and ‘Higher temperature’ | Excited occupancy rises from about0.268941 to0.450166 at fixed gap and g=1 | Why does this approach1/2 rather than certainty of excitation? |
| Choose ‘Degenerate states’ | At g=3, total entropy is about1.268301 while binary occupancy entropy is0.691933 | Which uncertainty disappears when all excited microstates are merged into one label? |
| Change only selected temperature T | The numeric readout changes; the temperature-sweep curves retain the same gap and degeneracy | Why is selecting a point different from changing the family of curves? |

The blue curve shows an energy-level occupancy, while orange shows microscopic entropy in units of kB. The two vertical quantities are not probabilities on a common normalization. A curve is an exact finite-model temperature sweep evaluated numerically, not a sampled path through time. Manual controls and saved lab settings are independent of the fixed concept examples.

## Six exercises with explanations

### 1. Normalize named states before calculating a mean

Set gap and temperature to one with one excited microstate. Calculate the partition function, the excited occupancy and the mean energy. Why is the unnormalized excited weight not the occupancy?

**Explanation.** The weights are1 and exp(−1), so Z is about1.367879. Dividing the second weight by Z gives0.268941. The ground-state probability is the complement. Multiplying occupancy by the gap gives U=0.268941. A raw weight of about0.367879 does not sum with the ground probability to one and is not yet a probability. The same normalization is required when a learned exponential score defines a distribution, although physical temperature needs additional assumptions.

### 2. Count degeneracy without counting one state twice

Keep gap and temperature one and use g=3. Distinguish one excited microstate from the excited energy level. Compare the two entropy readouts.

**Explanation.** There are four named microstates, with weights1, exp(−1), exp(−1), exp(−1). Their sum is about2.103638. Total excited occupancy is0.524633; each of its three microscopic probabilities is one third of that value. Microscopic entropy is about1.268301. Merging all excited states leaves binary entropy0.691933, so the missing uncertainty is about0.576368, equal to excited occupancy times log3. A larger entropy here does not require a larger state energy.

### 3. Separate a warm preparation from an infinite-temperature limit

For g=1 and gap one, evaluate occupancy at T=5, then give the infinite-temperature result for g=3.

**Explanation.** At T=5, exp(−1/5) gives occupancy about0.450166. This is still a finite-temperature preparation. At infinite temperature every microscopic weight approaches one, so with g=3 there are four equally likely microstates. Each has probability1/4, total excited occupancy is3/4 and entropy is log4. Equally likely microstates do not imply equally likely energy levels when the levels have different degeneracies.

### 4. Change the energy reference consistently

Add the same temperature-independent energy C to every microscopic state. Determine how Z, probabilities, U, F and S change.

**Explanation.** Each Boltzmann weight gains the factor exp(−C/T), so Z gains the same factor and normalized probabilities stay fixed. The new mean energy is U+C and the free energy is F+C. Their difference and therefore entropy stay fixed. This is a specified energy-reference change; an arbitrary rescaling of weights must not silently acquire that physical interpretation. The gap between the two energy levels remains unchanged.

### 5. Distinguish a stationary law from a fixed instantaneous energy

For one ground and one excited state, write the energy variance. Can equilibrium probabilities be stationary while this variance is positive?

**Explanation.** Energy is either zero or the gap. Its first two moments are gap times excited occupancy and gap squared times excited occupancy, so variance is gap squared times occupancy times its complement. At gap=T=1 this is about0.196612. Stationarity means the law is unchanged, not that every realization has energy U. This lab computes the law directly; it does not simulate energy exchange with a reservoir or a heat/work protocol.

### 6. Read a temperature sweep without inventing a trajectory

Change only T from one to five while keeping the gap and g fixed, then change g. Which readouts and curves should change? Can entropy difference alone determine heat exchanged by an unspecified process?

**Explanation.** Selecting T changes the table's probabilities and thermodynamic state quantities. Both plotted curves still sweep the same positive temperature grid at fixed gap and g, so they retain their shape. Changing g changes that family and both entropy readouts. Neither operation supplies a physical path or a heat/work boundary condition. Relating an entropy difference to heat requires a defined process and its reversibility assumptions; the finite equilibrium table does not provide them.

## Source and next steps

For canonical probabilities and energy moments, see [MIT Statistical Physics: Micro-canonical and Canonical Ensembles](https://web.mit.edu/esfaki/www/8.08_files/canonical.pdf). The finite-level calculation above is this book's specified teaching model. Compare it with the finite-spectrum, low-temperature and connected-correlation readers before adding interactions. Keep exact finite statements, numerical display limits and research hypotheses separate.
