## A symmetric average can hide two ordered-looking populations

Imagine repeatedly preparing a small collection of arrows. Each arrow points up or down. Interactions favor alignment, so many preparations look mostly up and many look mostly down. Averaging the signed alignment over all preparations can nevertheless give exactly zero. Averaging its magnitude gives a positive answer. These statements describe different observables of the same probability distribution.

This distinction matters when studying phase transitions, energy-based models and sampling. Two peaks in a finite histogram do not establish a thermodynamic singularity. A sampler that remains near one peak need not be sampling the full equilibrium distribution. We first solve an explicitly finite model, then identify what a later thermodynamic-limit argument must add.

You should know finite Boltzmann weights, reference measures, conditional entropy and partition-function derivatives. [Tong's introduction to spins and fields](https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S1.html) provides background on magnetization and statistical ensembles. The completely connected finite preparation, counting formulas and worked examples below are specified separately here.

## Specify the system before naming its temperature

There are N named classical spins, each taking the value +1 or −1. These are discrete classical variables, without quantum superposition. Define the total spin M and normalized magnetization m by

$$
M=\sum_{i=1}^{N}\sigma_i,\qquad m=\frac{M}{N},\qquad \sigma_i\in\{-1,+1\}.
$$

Our completely connected Curie–Weiss Hamiltonian and its dimensionless parameters are

$$
E(\boldsymbol\sigma)=-\frac{J}{2N}M^2-hM,
\qquad K=\frac{J}{k_BT},\qquad b=\frac{h}{k_BT}.
$$

J and h are energies, and T is a positive physical temperature. The factor 1/N keeps the interaction energy extensive as N grows. The input K is a dimensionless coupling, rather than a temperature measured in kelvin. The input b is a dimensionless energy bias. Converting m into a magnetization in SI units requires a magnetic moment and a volume convention as well.

The finite calculator prepares N=1 through 128, K between 0 and 8, and b between −4 and 4. These are explicit computational bounds. The derivations below apply to larger finite N too. With physical J and h fixed, K and b scale with 1/T; a zero coefficient stays zero. At h=0, a temperature scan can therefore change K while b remains zero. For nonzero J and h, varying only K specifies a different preparation.

## Counting states is part of the model

If k spins point up, M=2k−N. The number of named configurations with that total is

$$
g_N(M)=\binom{N}{k},\qquad k=\frac{N+M}{2},
\qquad M\in\{-N,-N+2,\ldots,N\}.
$$

For N=2 the bookkeeping is small enough to inspect directly:

| Named configurations | M | m | Multiplicity of the M label |
| --- | --- | --- | --- |
| down, down | −2 | −1 | 1 |
| down, up; up, down | 0 | 0 | 2 |
| up, up | 2 | 1 | 1 |

At K=b=0 all four named configurations are equally likely. The three M labels have probabilities 1/4, 1/2 and 1/4. Assigning equal probability to those three labels would change the reference measure. It would no longer describe uniform named spins.

For any N, the multiplicities sum to 2 to the power N. Counting an M label once and counting every configuration once are different operations. The interaction favors large absolute M, while multiplicity can favor the center. Their competition determines the macrostate law.

## From microscopic weights to a finite magnetization law

Summing over configurations with the same M gives

$$
Z_N(K,b)=\sum_M g_N(M)\exp\!\left(\frac{KM^2}{2N}+bM\right),
\qquad
P_N(M)=\frac{g_N(M)\exp(KM^2/(2N)+bM)}{Z_N(K,b)}.
$$

A named configuration in the M class has probability P_N(M)/g_N(M). The macrostate probability can therefore be large even when each configuration in that class has a small probability.

Subtracting a common maximum exponent before exponentiating leaves P unchanged. It does not remove that exponent from log Z. Similarly, replacing the counting reference g by the normalized prior g divided by 2 to the power N leaves P unchanged but shifts log Z by minus N log 2. Free-energy and entropy statements must keep the reference convention.

All finitely many macrostate probabilities are strictly positive at finite K,b. A tiny plotted bar or a floating-point value outside the display range is not a forbidden state.

## The self-interaction convention shifts energy

Expanding M squared shows exactly what is included in our Hamiltonian:

$$
M^2=N+2\sum_{i<j}\sigma_i\sigma_j,
\qquad
E=-\frac J2-\frac JN\sum_{i<j}\sigma_i\sigma_j-hM.
$$

If the pair-only convention removes the first constant, its energy is E plus J/2. Probabilities and entropies stay the same, while

$$
\log Z_{\mathrm{pair}}=\log Z_N-\frac K2,
\qquad
\beta F_{\mathrm{pair}}=-\log Z_N+\frac K2,
\qquad \beta=\frac1{k_BT}.
$$

For one spin, the interaction is just a constant. At zero field, log Z is K/2+log 2, mean dimensionless energy is −K/2, and entropy is log 2. The pair-only system has log Z=log 2 and zero energy. Both give equal probabilities to up and down. A change in free-energy zero is not evidence that the probability law changed.

## Four different summaries answer four different questions

The signed mean asks which direction is preferred across preparations. The mean magnitude asks how aligned a typical preparation is, regardless of sign. The second moment measures squared alignment. A mode asks which discrete total has the largest probability.

$$
\langle m\rangle=\sum_M \frac MN P_N(M),\qquad
\langle |m|\rangle=\sum_M\frac{|M|}{N}P_N(M),\qquad
\langle m^2\rangle=\sum_M\frac{M^2}{N^2}P_N(M).
$$

The absolute value of the signed mean is usually different from the mean absolute value. At zero field the first can vanish while the second is positive. At a small nonzero field the mean can be positive even if the most likely total remains M=0.

A mode is a summary of this chosen macrostate labeling. The most likely named configuration need not belong to the most likely M class. Grouping changes which outcome the word “likely” refers to.

## Work a complete two-spin example

Set N=2, b=0 and stipulate the exact mathematical value K=log 2. The endpoint weights are exp(K)=2, and the central weight is its multiplicity 2. Thus Z=6 and all three M labels have probability 1/3.

$$
\langle m\rangle=0,\qquad
\langle |m|\rangle=\langle m^2\rangle=\frac23,
\qquad \operatorname{Var}(m)=\frac23.
$$

The four named configurations have probabilities 1/3, 1/6, 1/6 and 1/3. Hence the entropy of the three-label distribution and the entropy of named configurations differ:

$$
H(M)=\log3,\qquad
\frac{S_{\mathrm{micro}}}{k_B}=\log3+\frac13\log2.
$$

The same entropy follows from thermodynamics: mean dimensionless energy is −2K/3 and dimensionless free energy is −log 6. Their difference is log 6−2 log 2/3. This agreement would fail if the macrostate entropy were substituted for the microscopic entropy without its multiplicity correction.

## Exact finite symmetry is compatible with two peaks

At b=0 the transformation that flips every spin preserves energy and multiplicity. It pairs M with −M, so

$$
P_N(M)=P_N(-M),\qquad \langle m\rangle=0.
$$

This is an exact statement for every finite N. Odd N does not even allow M=0 as a configuration, yet the ensemble mean is zero. For N=32,K=2,b=0 the macrostate modes are M=−32 and M=32. The signed average still vanishes.

For fixed finite N and finite parameters, Z is a positive finite sum of analytic exponentials. Its real log and its finite derivatives are smooth. A changing mode, a two-peak shape and an average over one trapped sample trajectory do not overturn that fact.

The later variational chapter studies a limit in which N grows without bound and the order of a vanishing field matters. The completely connected model must also be distinguished from a nearest-neighbor lattice model, where a mean-field treatment is an approximation with its own validity conditions.

## Differentiate the preparation that was actually specified

Holding K fixed while varying b gives finite exact response identities:

$$
\frac{\partial\log Z_N}{\partial b}=\langle M\rangle,
\qquad
\chi_b:=\frac{\partial\langle m\rangle}{\partial b}
=\frac{\operatorname{Var}(M)}N
=N\operatorname{Var}(m)>0.
$$

The factor N matters: the variance of an average and its response to a common bias scale differently. For a physical energy field h at fixed J,T, the response is chi_b divided by k_B T. It has units of inverse energy and is not automatically a magnetic susceptibility in a chosen SI convention.

Because m lies between −1 and 1, its variance is at most one. Full finite support gives it strictly positive variance. Consequently

$$
0<\chi_b=N\operatorname{Var}(m)\leq N.
$$

Every finite preparation has a finite response, but the upper bound grows with N. It is therefore not a uniform bound for a thermodynamic limit. A later divergent limiting susceptibility would not contradict the finite-system calculation. Naming K=1 a critical parameter of the limiting model does not insert a singularity into the finite sum.

The coupling derivative follows from the self-included energy convention:

$$
\frac{\partial\log Z_N}{\partial K}=\frac{\langle M^2\rangle}{2N},
\qquad
\frac{\partial\log Z_{\mathrm{pair}}}{\partial K}
=\frac{\langle M^2\rangle-N}{2N}.
$$

Within a fixed M class, all g configurations are equiprobable. Conditional entropy therefore supplies the missing microscopic information:

$$
\frac{S_{\mathrm{micro}}}{k_B}
=H(M)+\sum_M P_N(M)\log g_N(M),
\qquad
\beta F=\beta\langle E\rangle-\frac{S_{\mathrm{micro}}}{k_B}=-\log Z_N.
$$

Temperature derivatives require an explicit statement of which dimensional coefficients are fixed. An independent K slider is not a general temperature derivative: the path of b must also be specified, including its constant zero value when h=0.

The two entropy maxima also refer to different labels:

$$
H(M)\leq\log(N+1),\qquad S_{\mathrm{micro}}/k_B\leq N\log2.
$$

For N=2 the stipulated K=log 2 preparation maximizes the three-label entropy. Uniform named configurations at K=b=0 instead maximize microscopic entropy. A model that optimizes entropy after grouping outcomes can therefore favor a different preparation. The conditional information explains the difference; it is not a normalization error.

## Check the independent-spin limit

At K=0 the named spins factorize. With the same bias on every spin,

$$
Z_N=(2\cosh b)^N,\qquad
\langle m\rangle=\tanh b,\qquad
\operatorname{Var}(m)=\frac{\operatorname{sech}^2 b}{N},
\qquad \chi_b=\operatorname{sech}^2 b.
$$

At zero bias, the variance is 1/N and the dimensionless response is one. For N=1, the mean absolute magnetization is always one, even though a single spin cannot establish the finite-temperature thermodynamic transition discussed for an infinite system.

Here cosh b is (exp(b)+exp(−b))/2, sech b is its reciprocal, and tanh b is (exp(b)−exp(−b))/(exp(b)+exp(−b)). The up probability of one spin is logistic(2b). Increasing N narrows the histogram of their average even without any interaction. A narrower histogram alone therefore does not prove cooperative ordering.

This limit is a useful research check. It tests normalization, the factor N in the response, spin counting and the energy convention without relying on an interacting numerical solution.

## A rounded tie is not an exact mode

Return to N=2,b=0. Endpoint and central log weights are K and log 2. Their comparison is exact at the stipulated mathematical K=log 2. A binary64 input produced by `Math.log(2)` is a nearby rational number, slightly below that real value.

| Preparation | Comparison of exact log weights | Certified macrostate modes |
| --- | --- | --- |
| Stipulated real K=log 2 | Equal | −2, 0, 2 |
| Actual binary64 `Math.log(2)` | Central is slightly larger | 0 |
| Next larger binary64 number | Endpoints are slightly larger | −2, 2 |

The displayed probabilities may all round to the same decimal. That display does not certify a tie. Equal multiplicities allow a common log factor to cancel exactly; other comparisons need directed bounds on log weights.

The selected argmax can jump as K crosses a threshold even in this three-label finite system. The probabilities and free energy remain smooth. A discontinuity introduced by an optimization summary is not a thermodynamic phase transition.

## Read numerical and graphical evidence with its scope

An informative diagram separates the binomial multiplicity, the energy bias and the normalized macrostate probabilities. A second display compares the signed mean, mean magnitude, second moment and certified modal totals. A sequence of different N,K,b preparations compares equilibrium laws; it does not advance a physical clock.

Large integer counts must remain exact. The central binomial multiplicity at N=128 cannot be stored exactly as a binary64 number, even though some much larger powers of two can. The sum of all multiplicities must still equal exactly 2 to the power 128.

For numerical variance, a positive pair identity avoids subtracting two nearly equal moments:

$$
\operatorname{Var}(m)=\sum_{i<j}P_iP_j(m_i-m_j)^2.
$$

The calculator uses directed rational enclosures and increases precision only within a stated resource limit. At that limit an unresolved value is distinct from zero. A positive probability outside the linear display range keeps its positive sign and its separately computed log probability. Magnitude-log displays for general observations are approximations; the stored directed bounds certify the enclosed quantity.

An extremely small nonzero field can select a unique mode while a capped calculation has not resolved the magnitude of the mean. Those are different mathematical questions. Comparing them is more informative than erasing the field or declaring an unresolved value to be zero.

## Connect the finite law to the experiment

Start with **Two spins, multiplicities 1:2:1**. Before reading the bars, count the four named configurations on paper. The middle magnetization label contains two configurations, so it receives twice the probability of either endpoint. The plot shows the full macrostate law; the paginated table retains multiplicities and log probabilities even when a bar is visually tiny.

| Preparation to compare | Predict before changing it | Read in the result |
| --- | --- | --- |
| Two independent spins → actual log(2) | Can equal-looking bars certify equal modes? | Compare rounded probabilities with certified modal M; the actual binary64 input selects M=0. |
| N=16 → N=32 → N=64 at K=2,b=0 | Does a sharper two-peak picture make the signed mean nonzero? | E[m] stays exactly zero; E[|m|] and N Var(m) answer different questions. |
| Positive → negative field at N=32 | Which measurements change sign under reflection? | The signed mean reflects; magnitude and entropy remain unchanged between reflected preparations. |

Enter N, K or b and press Enter or leave the field to commit. N must be an integer from 1 to 128; a fractional decimal must not become an integer merely through rounding. Use **Next state rows** to inspect distributions with more than sixteen rows. Changing the arithmetic precision ceiling changes the numerical resource limit, not the spin preparation. An unresolved observation must stay distinct from a forbidden state or exact zero. Automatic example scanning compares equilibria; it does not simulate a magnet's time evolution.

## Research extension: a finite symmetric system can have critical fluctuations

The signed mean alone hides an important experiment. Prepare the same finite system repeatedly at K=1,b=0 and record M. The histogram remains symmetric, but its width grows faster than it would for independent spins. A vanishing average therefore coexists with a growing response. We can predict the right rescaling and the entire limiting shape without introducing a finite-N singularity.

### Let the first surviving entropy term set the scale

Keep the original full-M² Hamiltonian and labeled-spin counting measure. Set X_N=M/N^(3/4). The exact macrostate weight is

$$
w_N(M)=\binom{N}{(N+M)/2}\exp\!\left(\frac{M^2}{2N}\right),\qquad P_N(M)=\frac{w_N(M)}{\sum_{M'}w_N(M')},\qquad X_N=N^{1/4}m.
$$

For independent spins, entropy penalizes a deviation with a quadratic exponent and M typically has scale sqrt(N). At critical coupling the interaction cancels precisely that quadratic penalty. The quartic entropy cost survives. Setting N m⁴ to order one gives m of order N^(−1/4), or M of order N^(3/4). This remains a shrinking magnetization per spin, even as the total magnetization fluctuates more strongly.

$$
s(m)=\ln2-\frac{m^2}{2}-\frac{m^4}{12}-\frac{m^6}{30}-\cdots,\qquad N\!\left[s(m)+\frac{m^2}{2}-\ln2\right]=-\frac{X_N^4}{12}+O(N^{-1/2})
$$

The remainder statement holds on bounded sets of the scaled coordinate, not uniformly up to m=±1. Stirling's formula also supplies a prefactor. On such bounded sets the lattice spacing in X is 2/N^(3/4), and the normalized mass divided by that spacing approaches a density. Plotting raw probability masses at different N would obscure this comparison.

$$
\binom{N}{N(1+m)/2}=\sqrt{\frac{2}{\pi N(1-m^2)}}e^{Ns(m)}[1+O(N^{-1})],\qquad \Delta X_N=\frac{2}{N^{3/4}}.
$$

A local Taylor expansion alone does not establish convergence of a whole distribution or its moments. Here all higher even entropy coefficients are negative, so s(m)+m²/2−ln2≤−m⁴/12. Inside any fixed |m|≤δ<1 the Stirling prefactor is bounded by a constant times N^(−1/2). Outside that interval the entropy bound and binomial type bound suppress the finite sum exponentially in N, including the endpoints. A central interval supplies a normalizer of order 2^N N^(1/4). These estimates control the tails after rescaling, also with fixed polynomial moment weights; Riemann sums then give the law and moments below. This argument uses the completely connected binary model and its specified preparation.

$$
X_N\Rightarrow X,\qquad f_X(x)=\frac{2}{12^{1/4}\Gamma(1/4)}e^{-x^4/12},\qquad x\in\mathbb R.
$$

### Read susceptibility and a shape diagnostic together

The Gamma integral follows by substituting t=x⁴/12 into the integral over x>0. Odd moments vanish by reflection. The second and fourth moments determine both the response scale and the Binder shape statistic U. The latter is dimensionless and unchanged by multiplying the random variable by a nonzero constant.

$$
\mathbb E[X^{2r}]=12^{r/2}\frac{\Gamma((2r+1)/4)}{\Gamma(1/4)},\qquad \mathbb E[X^2]=\sqrt{12}\frac{\Gamma(3/4)}{\Gamma(1/4)},\qquad \mathbb E[X^4]=3.
$$

$$
\frac{\chi_b(N)}{\sqrt N}=\mathbb E[X_N^2]\longrightarrow\mathbb E[X^2]\simeq1.170829,\qquad \chi_b(N)=\frac{\operatorname{Var}(M)}N.
$$

$$
U_N=1-\frac{\mathbb E[M^4]}{3\mathbb E[M^2]^2}\longrightarrow U_*=1-\frac{1}{\mathbb E[X^2]^2}\simeq0.270520.
$$

A centered Gaussian has U=0; two equally weighted, infinitely narrow opposite peaks have U=2/3. The quartic law has a different shape from either. U is a diagnostic under stated sampling and finite-size conditions, not a universal test that proves a phase transition from four small systems. In particular every finite probability here remains analytic in K,b, and every finite zero-field mean remains exactly zero.

![Scaled critical magnetization densities approaching a quartic law and finite-size Binder values approaching a non-Gaussian limit.](/assets/learning/phase-transition-research-critical-fluctuations.en.svg)

The upper panel divides each lattice mass by its scaled spacing. The lower panel compares exact finite sums with the limiting shape statistic. Both panels use equilibrium probabilities, not a time trajectory or a sampler convergence claim.

### Reproduce the finite sums before fitting an exponent

The table reports N, E[X_N²], U_N, and χ_b. The following independent Python example deliberately goes beyond the existing calculator's N≤128 bound; it does not add or imply a new slider. Only the standard library is needed. Log weights avoid overflow and `fsum` reduces summation error; these are floating-point illustrations rather than certified arbitrary-precision intervals.

| N | E[X_N²] | U_N | χ_b |
| --- | --- | --- | --- |
| 32 | 1.126622 | 0.333612 | 6.373139 |
| 128 | 1.147692 | 0.301425 | 12.984656 |
| 512 | 1.158990 | 0.285800 | 26.224955 |
| 2048 | 1.164840 | 0.278115 | 52.714659 |

```python
from math import exp, fsum, lgamma, gamma, sqrt

for n in (32, 128, 512, 2048):
    ms = list(range(-n, n + 1, 2))
    logs = [lgamma(n + 1) - lgamma((n + m)//2 + 1)
            - lgamma((n - m)//2 + 1) + m*m/(2*n) for m in ms]
    top = max(logs)
    weights = [exp(x - top) for x in logs]
    z = fsum(weights)
    x2 = fsum(w*(m/n**0.75)**2 for w, m in zip(weights, ms))/z
    x4 = fsum(w*(m/n**0.75)**4 for w, m in zip(weights, ms))/z
    binder = 1 - x4/(3*x2*x2)
    chi = sqrt(n)*x2
    print(n, f'{x2:.6f}', f'{binder:.6f}', f'{chi:.6f}')
```

Doubling N would asymptotically multiply χ_b by sqrt(2), while an ordered two-peak phase at fixed K>1 instead has χ_b proportional to N. Comparing these scalings helps distinguish mechanisms behind large response. A fit over a short range can still be distorted by finite-size corrections.

### Research checks and scope

#### Why does a growing total-spin width not imply nonzero signed order?

**Research explanation.** Reflection pairs every M with −M, leaving E[M]=0. Meanwhile E[M²] grows as N^(3/2). Dividing by N² shows E[m²] decreases as N^(−1/2). The fluctuation scale, signed expectation and selected thermodynamic branch answer different questions.

#### What factor is missing if a histogram plots P_N(M) directly against X_N?

**Research explanation.** The spacing changes with N. The density comparison requires P_N(M)/ΔX_N. Its Riemann-sum area is one, whereas an uncorrected mass curve becomes lower merely because there are more grid points per scaled interval.

#### What happens to the standard quadratic approximation at K=1?

**Research explanation.** Its curvature is zero, so a Gaussian formula would divide by zero and miss the N^(3/4) scale. Keeping the first positive quartic free-energy term gives a normalizable non-Gaussian law. This is a change of asymptotic approximation, not an infinite variance at finite N.

For background, [Cerf and Gorny's introduction, pp. 2–3](https://www.imo.universite-paris-saclay.fr/~raphael.cerf/papers/cwsoc.pdf) recalls critical generalized Curie–Weiss limits; its later self-organized model is a different preparation. The binary critical density, moment constants, tail argument and finite sums used here are derived explicitly above. No spatial critical exponent or physical relaxation time follows from them.

## Six exercises with explanations

### 1. Zero mean without a zero configuration

For N=3,K=b=0, find the macrostate law, signed mean, mean magnitude and response.

**Explanation.** The totals −3,−1,1,3 have counts 1,3,3,1 out of eight configurations. The signed mean is zero, mean magnitude is 1/2 and second moment is 1/3. Thus chi_b=3 times 1/3=1. No individual configuration has m=0. The ensemble average need not be a possible individual outcome.

### 2. Distinguish entropy and response in the two-spin preparation

At the stipulated N=2,K=log 2,b=0, compute the response and microscopic entropy. Is log 3 enough for the entropy?

**Explanation.** The signed mean is zero and the second moment is 2/3, so chi_b=4/3. The central class has probability 1/3 and contains two configurations. Its conditional entropy contributes one third of log 2, giving microscopic entropy divided by k_B equal to log 3+(log 2)/3. Log 3 alone is the entropy of the chosen three-label observation.

### 3. Remove a constant self energy

For one spin at K=6,b=0, compare the two energy conventions and explain whether their probabilities differ.

**Explanation.** The self-included law has log Z=3+log 2, mean dimensionless energy −3 and entropy log 2. Removing the self energy shifts every energy upward by 3 thermal units, leaving log Z_pair=log 2 and zero mean pair energy. Up and down remain equally likely. The shift cancels in entropy and every normalized probability.

### 4. Check the factor N in a field response

At K=0, show why the variance of m shrinks with N but the common-bias response does not.

**Explanation.** The N spins are independent with mean tanh b and variance sech squared b. Averaging divides the variance by N. Differentiating the mean with respect to their common b gives sech squared b. Therefore the response is N Var(m), rather than Var(m) or N squared Var(m). A physical h derivative introduces the additional factor beta.

### 5. Can a mode change prove a phase transition?

The certified modes change from M=0 to M=±2 across K=log 2 for N=2,b=0. Explain why this does not contradict finite analyticity.

**Explanation.** Comparing log 2 with K determines which smooth probability is largest. The argmax operation can switch its selected label when two probabilities cross. Z=2 exp(K)+2 remains positive and analytic, as do its log and finite derivatives. The discontinuity belongs to the selected summary, not to the finite system's thermodynamic free energy.

### 6. State a limit before interpreting broken symmetry

The next chapter establishes the Curie–Weiss variational limit. At K=log 3, its zero-field ordered solutions are m=±1/2. Contrast taking the finite-system zero-field average first with selecting a positive-field infinite-system branch first.

**Explanation.** At every finite N, setting b=0 gives exactly zero mean, so the subsequent limit of those means is zero. Taking the thermodynamic limit at positive b first can select the positive minimizer; its b approaching zero from above gives 1/2. Substitution checks tanh(log 3/2)=1/2. Selecting an infinite-system branch adds a limit prescription that a finite symmetric histogram does not contain.

## Research connections and the next step

For an energy-based model, distinguishing a distribution, a grouped label and a sample trajectory prevents three different errors. A grouped entropy omits conditional information, a trapped trajectory can look symmetry broken, and a rounded argmax can look like a sharp transition. Each requires a different diagnostic.

The next variational treatment derives the large-N constrained free energy, classifies stationary points and compares the order of limits. A later Landau treatment separates global coexistence, local stability and spinodals. [Tong's phase-transition discussion](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S5.html) supplies broader background and limitations of mean-field descriptions; the finite exact calculations here keep their stated fully connected scope.
