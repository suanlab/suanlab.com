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

## Research extension: can occupancy be a thermometer?

Imagine preparing the same tiny probe in equilibrium many times and asking only whether its energy is ground or excited. More excited observations suggest a warmer reservoir, but a nearly frozen probe seldom changes its answer; a very hot probe has nearly saturated occupancy. Neither extreme is automatically informative. We now derive a quantitative limit on this intuition.

Keep the original finite spectrum, known positive integer degeneracy g, a unique ground state and kB=1. Energies are independent of temperature. Each observation is an ideal, error-free energy-level label from a fresh independent equilibrium preparation at the same positive T. The observations below are a specified inference experiment, separate from the existing deterministic lab curves. A time series of correlated measurements does not satisfy these sampling assumptions. Restoring SI units replaces x by Δ/(kB T) and C here by physical heat capacity divided by kB.

Write X=1 for an excited energy-level observation. Its probability includes every excited microstate. The sufficient statistic M counts excited outcomes among N preparations; naming which of the g equal-energy excited microstates occurred adds no temperature information when g and the spectrum are known.

$$
x=\Delta/T,\qquad p=\frac{g e^{-x}}{1+g e^{-x}},\qquad X\in\{0,1\}.
$$

The count follows a binomial law. This is a probability model for repeated observations, whereas the original chart evaluates an equilibrium distribution without drawing samples.

$$
\Pr(M=m)=\binom Nm p^m(1-p)^{N-m},\qquad M=\sum_{n=1}^{N}X_n.
$$

Differentiate at fixed Δ and g. Since U=Δp and energy takes only the values zero and Δ, the response of mean energy to temperature is controlled by its variance. C is dimensionless under the chosen kB=1 convention.

$$
\frac{\partial p}{\partial T}=\frac{\Delta}{T^2}p(1-p),\qquad C=\frac{\partial U}{\partial T}=x^2p(1-p)=\frac{\operatorname{Var}(E)}{T^2}.
$$

For a dimensionless logarithmic temperature θ, choose any fixed reference T*. The score measures how much the log probability changes under a small fractional temperature change. Its mean is zero; squaring and averaging gives Fisher information per preparation. Thus the same C measures both thermal response and local statistical sensitivity.

$$
\theta=\log(T/T_*),\qquad \partial_\theta\log\Pr(X)=x(X-p),\qquad I_{\theta\theta}=C.
$$

Independence adds information: N preparations give N times the one-preparation value. The Cramér–Rao inequality below applies to estimators unbiased for T in a neighborhood of an interior positive parameter, under the usual differentiability assumptions. It is a lower bound, not a confidence interval, a guarantee of attainable accuracy, or a statement that an arbitrary fitted estimator is unbiased.

$$
\operatorname{Var}(\widehat T)\ge\frac{T^2}{NC},\qquad \frac{\sqrt{\operatorname{Var}(\widehat T)}}{T}\ge\frac{1}{\sqrt{NC}}.
$$

The interior maximum-likelihood estimate inverts the observed fraction. For example, g=1, Δ=1 and M=250 of N=1000 give T̂=1/log(3), about 0.910239 in the chosen energy units. Finite-sample noise matters: if M=0, the likelihood has its supremum as T approaches zero. If M/N is at least g/(1+g), it has its supremum as T tends to infinity. These cases have no finite positive maximizer. Do not clip them into a claimed finite estimate or insert the random MLE into an unbiased-estimator theorem.

$$
\widehat p=M/N,\qquad \widehat T=\frac{\Delta}{\log[g(1-\widehat p)/\widehat p]},\qquad 0<\widehat p<\frac{g}{1+g}.
$$

Now suppose Δ is also unknown and only the binary labels are recorded. With a fixed reference Δ*, the two log-parameter scores point in opposite directions. An increased gap and a decreased temperature both suppress excitation.

$$
\eta=\log(\Delta/\Delta_*),\qquad \nabla_{(\theta,\eta)}\log\Pr(X)=x(X-p)(1,-1).
$$

The information matrix has rank one for every finite positive x. Multiplying it by the common-scale direction gives zero. No number of independent repetitions of this same label experiment removes that null direction.

$$
I_{(\theta,\eta)}=C\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\qquad I(1,1)^{\mathsf T}=0.
$$

Indeed a simultaneous positive rescaling of T and Δ leaves the entire count distribution unchanged, even though the physical mean energy rescales. Calibrated gap spectroscopy, an absolute energy measurement or another independent physical reference can supply the missing scale. A tighter fit to labels alone cannot. This claim concerns the stated observation model; calibrated measurements of numerical energy values would be a different experiment.

$$
(T,\Delta)\mapsto(aT,a\Delta),\quad a>0:\qquad p\mapsto p,\quad U\mapsto aU.
$$

For known gap, the best operating ratio for fractional-temperature sensitivity maximizes C, rather than occupancy itself. The logarithmic derivative gives the stationary condition. For g=1 it has a unique positive solution x≈2.399357: the left side is strictly increasing from zero to infinity. C vanishes at both positive-temperature extremes, so this point is the global maximum, not a phase transition. The condition and numerical value are specific to this finite spectrum and the chosen relative-error objective; maximizing information about absolute T while varying T at fixed Δ is a different optimization.

$$
\frac{d\log C}{dx}=\frac2x-(1-2p),\qquad g=1:\quad x\tanh(x/2)=2.
$$

### Read the sensitivity and ambiguity diagram

The left panel plots information per independent preparation against x for g=1; its orange marker is the fractional-sensitivity maximum. The right panel marks three different physical pairs with the same ratio x=1 and therefore the same occupancy. The line depicts an exact observational ambiguity, not uncertainty caused by a small sample. All axes and numerical examples use the original kB=1 convention. The diagram and executable supplement do not add a new lab control.

![Information peaks at an intermediate gap-to-temperature ratio; common scaling of gap and temperature leaves occupancy unchanged.](/assets/learning/thermometry-information-scale.en.svg)

### Reproduce four operating points

For N=1000 and g=1, the last column is the relative standard-deviation lower bound for an unbiased estimator under the assumptions above. Smaller values indicate a less restrictive bound, not measured errors. The second operating point, x=1, gives approximately 7.13%; the optimum gives approximately 4.77%.

| x = Δ/T | Excited probability p | C = information per preparation | Relative SD lower bound |
| --- | --- | --- | --- |
| 0.20000000 | 0.45016600 | 0.00990066 | 0.31781022 |
| 1.00000000 | 0.26894142 | 0.19661193 | 0.07131733 |
| 2.39935728 | 0.08322172 | 0.43922884 | 0.04771496 |
| 8.00000000 | 0.00033535 | 0.02145521 | 0.21589054 |

```python
from math import exp, sqrt
for x in [0.2, 1.0, 2.3993572805154677, 8.0]:
    p = 1.0 / (1.0 + exp(x))  # known g = 1
    C = x * x * p * (1.0 - p)
    bound = 1.0 / sqrt(1000 * C)
    print(f"{x:.8f} {p:.8f} {C:.8f} {bound:.8f}")
```

### Three research questions with explanations

#### 1. Can more data distinguish (T, Δ)=(1,1) from (2,2)?

**Explanation.** No, if the recorded data are only the specified labels. Both give x=1 and the same probability for every finite label sequence. The information null vector (1,1) describes this common-scale change locally; the exact scaling identity establishes it globally. Calibrating one physical energy scale changes the inference problem.

#### 2. Why does p close to one half fail to give the best thermometer for g=1?

**Explanation.** A balanced Bernoulli observation has large variance, but its probability changes only weakly with fractional temperature at very high T. Information is x²p(1−p), including that sensitivity factor. It tends to zero when x tends to zero, and also when the probe freezes at large x. The finite intermediate maximum follows from the derivative above.

#### 3. Does N=1000 ensure a finite temperature estimate or 4.77% accuracy?

**Explanation.** No. A sample can lie at a likelihood boundary, and the numerical lower bound assumes an interior parameter and an unbiased estimator. It is neither an error bar for this particular sample nor a finite-sample guarantee for the nonlinear MLE. Correlated readings or detector errors would additionally change the likelihood and information. Specify those mechanisms before claiming the same bound.

### Sources and research boundary

The canonical variance–heat-capacity identity is supported by [MIT, Micro-canonical and Canonical Ensembles, problem 1](https://web.mit.edu/esfaki/www/8.08_files/canonical.pdf) and [David Tong, Statistical Physics, sections 1.3.1–1.3.2](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf). The binomial likelihood, score, matrix null direction and operating-point calculation are derived here for the explicitly declared observation model. They do not establish thermometer performance for an interacting probe, a noisy detector or a neural network. In an energy-based model, a score scale and a temperature-like scale can likewise be confounded algebraically, but physical temperature requires physical calibration.

## Source and next steps

For canonical probabilities and energy moments, see [MIT Statistical Physics: Micro-canonical and Canonical Ensembles](https://web.mit.edu/esfaki/www/8.08_files/canonical.pdf). The finite-level calculation above is this book's specified teaching model. Compare it with the finite-spectrum, low-temperature and connected-correlation readers before adding interactions. Keep exact finite statements, numerical display limits and research hypotheses separate.
