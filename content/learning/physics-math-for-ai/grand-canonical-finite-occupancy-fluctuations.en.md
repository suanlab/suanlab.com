## Change what the reservoir is allowed to exchange

Imagine two labeled parking spaces, each either empty or occupied by one particle. The number of spaces stays fixed, but particles may be exchanged with a reservoir. Repeating the same preparation can return zero, one or two particles. Specifying a chemical potential determines their statistical weights; it does not fix the count in every preparation.

The previous canonical binary chapter kept the set of physical sites and their two internal states. Its random count recorded excitations. Here an occupied site contributes one particle, and an empty site contributes none. The two problems can share a Bernoulli formula while describing different conserved quantities and different preparation constraints.

We will derive the full particle-count law, condition it on a selected count, and connect fluctuations to chemical response. A static ensemble does not specify when particles arrive, a flow rate or a relaxation time. A sequence of comparisons should therefore be labeled by preparation parameters rather than physical time.

## Define sites, particles and their energy

There are M labeled sites, with at most one particle per site. A microstate is its occupancy vector. The energy per occupied site is ε≥0 J:

$$
x\in\{0,1\}^{M},\qquad N(x)=\sum_{i=1}^{M}x_i,\qquad E(x)=\varepsilon N(x),\qquad 0\le N\le M.
$$

This is a classical finite exclusion lattice. The state count distinguishes which sites are occupied. We do not attach particle labels and then multiply by their permutations; doing so would change the specified sample space. The formula is not the continuum ideal-gas state count, and a similar-looking occupancy function does not by itself establish a quantum Fermi model.

For finite nonzero T, define the dimensionless chemical parameter α, scaled energy t and net occupancy bias y:

$$
\beta=\frac1{k_BT},\qquad \alpha=\beta\mu,\qquad t=\beta\varepsilon,\qquad y=\alpha-t=\frac{\mu-\varepsilon}{k_BT}.
$$

The chemical potential μ and ε use the same energy unit, joules per particle with particle count treated as dimensionless. The finite calculator supports M from 0 through 4096, finite ε≥0, finite μ and finite temperatures of either sign. Its zero-temperature convention will be defined separately. The site limit is a computational resource bound, not a thermodynamic limit.

## Sum over particle numbers to normalize the preparation

The grand-canonical weight of a configuration is exp[−β(E−μN)]. Each occupied site contributes exp(y), so summing over all binary configurations gives:

$$
\Xi=\sum_xe^{-\beta(E(x)-\mu N(x))}=\sum_{k=0}^{M}\binom Mk e^{yk}=(1+e^y)^M,\qquad \ln\Xi=M\ln(1+e^y).
$$

The sum includes all possible particle numbers. It does not choose each number uniformly. The factor C(M,k) counts the different site patterns with k particles. The occupancy states already define the counting convention, so adding an extra factorial correction without changing the sample space would count a different model.

The general preparation and potential are introduced in [Tong, sections 1.4.1–1.4.2](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf). Here the concrete state space is the finite lattice specified above. No volume derivative or pressure has yet been supplied, so we do not infer a pressure law merely by naming a grand potential.

## Resolve a count into conditional patterns

The site occupancy probability, full count law and within-count conditional law are:

$$
r=\frac{e^y}{1+e^y},\qquad P(N=k)=\binom Mk r^k(1-r)^{M-k},\qquad p(x\mid N=k)=\binom Mk^{-1}.
$$

The conditional law applies when the event N=k has positive probability. At a genuine support zero it is undefined. At finite nonzero T all allowed count values have positive probability, even if a tail cannot be shown as an ordinary floating-point number.

For M=2 and μ=ε, y=0 and r=1/2:

| Occupancy pattern | Particle count | Pattern probability | Probability of its count |
| --- | ---: | ---: | ---: |
| 00 | 0 | 1/4 | 1/4 |
| 01 | 1 | 1/4 | 1/2 |
| 10 | 1 | 1/4 | 1/2 |
| 11 | 2 | 1/4 | 1/4 |

The two one-particle rows share a count bar. Given one particle, each of the two patterns has probability 1/2. That differs from its unconditional probability 1/4. Also, μ=0 does not mean an empty system: when ε=0 it produces this balanced preparation at any finite nonzero T.

## Derive particle, hole and energy fluctuations

The sites are independent under this grand-canonical preparation. Thus the mean particle count, its variance and the mean number of empty sites are:

$$
\langle N\rangle=Mr,\qquad \operatorname{Var}(N)=Mr(1-r),\qquad \langle M-N\rangle=M(1-r).
$$

Compute particles and holes separately when one is very rare. Subtracting a displayed mean of M from M can erase a small positive hole population. The presence of a hard occupancy limit does not mean the count is fixed below that limit.

Because physical energy is ε times the particle count, the other moments and entropy follow:

$$
U=\varepsilon\langle N\rangle,\qquad \operatorname{Var}(E)=\varepsilon^2\operatorname{Var}(N),\qquad \operatorname{Cov}(E,N)=\varepsilon\operatorname{Var}(N),\qquad S=Mk_Bh(r).
$$

Here h(r)=−r ln r−(1−r)ln(1−r). Energy variance has units J²; energy/number covariance has units J. Entropy is on occupancy patterns, not only count readings. At ε=0, energy can be identically zero while particle number still fluctuates.

## Specify coordinates before differentiating

Treat β and α as independent coordinates in the partition sum. Its derivatives give:

$$
\left(\frac{\partial\ln\Xi}{\partial\alpha}\right)_{\beta}=\langle N\rangle,\qquad \left(\frac{\partial^2\ln\Xi}{\partial\alpha^2}\right)_{\beta}=\operatorname{Var}(N),\qquad -\left(\frac{\partial\ln\Xi}{\partial\beta}\right)_{\alpha}=U.
$$

Holding α fixed is different from holding μ fixed, because α=βμ. At fixed T, ε and M the chemical response is:

$$
\chi_N=\left(\frac{\partial\langle N\rangle}{\partial\mu}\right)_{T,\varepsilon,M}=\beta\operatorname{Var}(N),\qquad \left(\frac{\partial U}{\partial T}\right)_{\mu,\varepsilon,M}=\frac{\varepsilon(\varepsilon-\mu)}{k_BT^2}\operatorname{Var}(N).
$$

The α derivative is dimensionless, while χ_N has units 1/J. For positive T it is nonnegative. It is a particle-number response, not automatically an isothermal compressibility: that interpretation would require volume and density definitions and the appropriate constraints.

The slope describes a family of equilibrium preparations. It does not give the speed at which particles enter after a change in μ. Time-dependent response needs a transition or transport model.

## Use the grand potential with the correct constraints

The grand potential and its first derivatives satisfy:

$$
\Omega=-k_BT\ln\Xi=U-TS-\mu\langle N\rangle,\qquad \left(\frac{\partial\Omega}{\partial\mu}\right)_{T,\varepsilon,M}=-\langle N\rangle,\qquad -\left(\frac{\partial\ln\Xi}{\partial\beta}\right)_{\mu}=U-\mu\langle N\rangle.
$$

The last derivative differs from the fixed-α derivative in the previous section. A derivative label is part of the statement, not decoration. Substituting one coordinate constraint for the other would lose the chemical contribution.

At positive T, Ω is concave as a function of μ because its second derivative is −χ_N. In this finite model the curve is smooth for every finite nonzero T. A sharp-looking numerical sweep is not evidence of a singular phase transition at a finite temperature.

The open-system combination U−TS−μ〈N〉 also explains why the previous closed-system heat-capacity formula cannot be applied to an arbitrary temperature sweep here without checking what remains fixed and whether particle number changes. The fixed-μ energy slope above can be negative when μ>ε: increasing temperature can reduce the mean particle population. Label this as an open-system energy slope, not the earlier fixed-particle heat capacity.

## Check a biased two-site preparation by hand

Take M=2 and choose μ so that y=−ln 2 at a fixed positive T. Then r=1/3. The count probabilities are (4,4,1)/9, while the conditional one-particle patterns remain equally likely:

$$
\langle N\rangle=\frac23,\qquad \operatorname{Var}(N)=\frac49,\qquad \chi_N=\frac4{9k_BT},\qquad \ln\Xi=2\ln(3/2),\qquad \Omega=-2k_BT\ln(3/2).
$$

For the same state, U=2ε/3, Var(E)=4ε²/9 and Cov(E,N)=4ε/9. These are three different physical quantities, even though their dimensionless factors are related. Reversing the bias to y=ln 2 changes the count law to (1,4,4)/9, its mean to 4/3 and its variance remains 4/9.

A useful visualization places the count distribution beside its selected conditional patterns, then draws mean occupancy and chemical response against μ. A single occupancy bar hides the distinction between two preparations with the same mean and different number constraints.

## Shift energy and chemical reference together

Adding the same per-particle constant c to ε and μ preserves every grand-canonical weight:

$$
\varepsilon'=\varepsilon+c,\qquad \mu'=\mu+c,\qquad E'(x)-\mu'N(x)=E(x)-\mu N(x),\qquad U'=U+c\langle N\rangle.
$$

Occupancy, entropy, ln Ξ and Ω are unchanged. Physical mean energy changes as shown, and energy variance changes because the random particle count multiplies the added per-particle constant. This is not a constant shift of every complete configuration when N varies.

Keep the distinction between this common reference change and physically changing ε while the reservoir μ remains fixed. The latter changes the net bias and therefore the preparation. For numerical work, equal very large ε and μ should still produce exactly zero difference before the division by thermal energy.

## Define the positive-side zero-temperature limits

The calculator's T=0 input means approach from positive temperature at fixed ε and μ. Write Δ=ε−μ. For M>0 there are three preparations:

| Fixed parameters | Limiting preparation | Number variance | S/kB | Limit of ln Ξ | Limit of Ω |
| --- | --- | ---: | ---: | --- | --- |
| Δ>0 | Empty pattern | 0 | 0 | 0 | 0 |
| Δ<0 | Fully occupied pattern | 0 | 0 | Positive infinity | MΔ |
| Δ=0 | Uniform over all patterns | M/4 | M ln 2 | M ln 2 | 0 |

At the threshold Δ=0, every pattern has the same E−μN. A positive physical ε does not remove this grand-energy degeneracy. At fixed μ=ε, the finite-temperature number susceptibility is M/(4kBT), which diverges as T approaches zero. The discontinuous exact-zero-temperature count law has no finite μ derivative at the threshold; it must not be reported as zero response.

For Δ<0, ln Ξ diverges but Ω has a finite limit. Do not evaluate that case as a floating-point product of zero and infinity. The limiting expression MΔ is available directly. At M=0 there is one empty configuration and all extensive quantities and the chemical response are zero.

## Check the path when two controls approach a threshold

The threshold value above uses fixed μ=ε. A joint approach can give another limit. For any finite real a, choose a temperature-dependent chemical potential:

$$
\mu(T)=\varepsilon+a k_BT,\qquad T\to0^+,\qquad y=a,\qquad r=\frac1{1+e^{-a}}.
$$

All these paths approach the same parameter point (T,μ)=(0,ε), but their occupancy probabilities can differ. For a=ln 3, the limiting occupancy is 3/4 rather than 1/2. There is no universal joint limit that ignores how the controls approach the point.

The model's literal T=0 threshold convention is therefore a stated preparation prescription. It does not classify every possible physical approach to that point. This distinction matters when analyzing boundary layers, parameter schedules or low-temperature numerical sweeps: the ratio of two vanishing quantities can retain finite information.

## Separate fixed-number preparation from independent occupancy

Conditioning the grand-canonical law on a count N=k makes that count deterministic. The remaining uncertainty concerns the occupied site labels. Without conditioning, the number itself fluctuates. For M>0 and finite nonzero T, useful measures are:

$$
\frac{\sqrt{\operatorname{Var}(N)}}{\langle N\rangle}=\sqrt{\frac{1-r}{Mr}}=\frac{e^{-y/2}}{\sqrt M},\qquad P(N=0)=(1-r)^M,\qquad P(N=M)=r^M.
$$

At M=2 and balanced occupancy, the unconditioned number variance is 1/2. Given N=1, it is zero even though the two conditional patterns remain uncertain. Replacing a fixed-number dataset by independent Bernoulli occupancy with the same expected count changes its correlations and count variance.

More independent repetitions improve an estimate of 〈N〉; they do not change the ensemble variance. A sample with no particles need not imply r=0. Likewise, a sample with every site occupied does not prove the hole probability is zero. Display sample count separately from site capacity and identify any sampling seed.

## Interpret inverted bounded preparations carefully

Because the finite exclusion spectrum is bounded, the model also normalizes finite negative temperatures. Reversing T at fixed ε and μ reverses y:

$$
r(-y)=1-r(y),\qquad \langle N\rangle_T+\langle N\rangle_{-T}=M,\qquad \operatorname{Var}_T(N)=\operatorname{Var}_{-T}(N),\qquad \chi_N(-T)=-\chi_N(T).
$$

Entropy is unchanged under this reflection, while the chemical response changes sign with β. A negative susceptibility here is consistent with the signed fluctuation relation; it is not a negative variance. The positive-side T=0 table does not describe the negative-side approach, which selects the opposite extremum of E−μN.

This bounded construction should not be transferred to an unrestricted kinetic-energy gas. It also supplies no preparation lifetime or particle-exchange kinetics. The temperature label describes the statistical weighting being compared.

## Preserve cancellations and positive tails numerically

The net bias should be evaluated as a difference before scaling:

$$
y=\frac{\mu-\varepsilon}{k_BT},\qquad \mu=\varepsilon\ \Longrightarrow\ y=0.
$$

Computing βμ and βε separately can overflow before their difference is taken. That would destroy the exact balance of equal energies. The numerical model retains binary input differences and products before projection, together with exact integer multiplicities. Its exponentials and logarithms remain approximations. Exact arithmetic on the supplied inputs cannot recover a chemical difference already rounded away when those inputs were stored; a very small joint-limit schedule requires an appropriate bias coordinate or sufficient input precision.

For positive T with a large positive u=(ε−μ)/(kBT), the dilute leading scales are:

$$
r\sim e^{-u},\qquad \langle N\rangle\sim M e^{-u},\qquad \chi_N\sim\frac{M e^{-u}}{k_BT}.
$$

A probability can fall below the linear display range while a large reciprocal thermal-energy factor makes susceptibility representable. For example, take ε=10^−297 J, μ=0 and T=ε/(800kB). The occupancy probability is near exp(−800), while the susceptibility for M=2 is near 1600 exp(−800)/10^−297 per joule. Rounding the probability to zero first would lose a valid response.

The implementation distinguishes mathematical support zeros, positive tails, an infinite zero-temperature partition limit, and a finite value outside the display range. Nested logs diagnose tails, not certified tiny departures from unit-valued moments. For AI, compare fixed-count selection with independent stochastic gates, but do not interpret an arbitrary model's chemical-style bias as a calibrated energy or a particle flow without a physical mapping.

## Research extension: fluctuations can leave a parameter direction invisible

Two noisy measurements need not give two independent constraints. In the equal-energy occupancy model, measuring energy and particle number is like using two rulers whose marks are exact rescalings of each other. Both fluctuate, but the second cannot add a new direction of information.

Start with the original identical-site model, then explicitly introduce a comparison system of two independent labeled sites with known energies ε₁ and ε₂. Site occupancy still excludes more than one particle. This heterogeneous system is a separate research example; the existing reader has one common energy parameter and does not expose two new energy controls.

The determinant vanishes even when Var(N)>0. Energy in joules and number without units lie on the line E=εN. Zero determinant means a redundant observable direction, not zero fluctuations or failed thermodynamic normalization.

$$
E=\varepsilon N:\quad\Sigma=\operatorname{Cov}(E,N)=\operatorname{Var}(N)\begin{pmatrix}\varepsilon^2&\varepsilon\\\varepsilon&1\end{pmatrix},\qquad\det\Sigma=0.
$$

For the comparison use finite positive temperature, known energy levels and a shared chemical potential μ. Summing all four labeled occupancy states factorizes because their energies add and their preparation is independent. In these derivative coordinates β and α are independent parameters.

$$
\beta=(k_BT)^{-1}>0,\quad\alpha=\beta\mu,\qquad\log\Xi=\sum_{i=1}^2\log(1+e^{\alpha-\beta\varepsilon_i}).
$$

Each finite-temperature occupancy is interior, so its variance vᵢ is positive. Different known energies supply two distinct energy-to-number ratios when the sites can fluctuate independently.

$$
r_i=\frac1{1+e^{\beta\varepsilon_i-\alpha}},\quad v_i=r_i(1-r_i),\qquad\langle E\rangle=\sum_i\varepsilon_i r_i,\quad\langle N\rangle=\sum_i r_i.
$$

Differentiate the finite exponential sum. One derivative gives a mean and the next subtracts the product of means, giving covariance. The minus sign in the first coordinate matters; directly using (β,α) would change the off-diagonal signs.

$$
\theta=(-\beta,\alpha):\qquad\nabla_\theta\log\Xi=(\langle E\rangle,\langle N\rangle),\qquad\nabla_\theta^2\log\Xi=\Sigma.
$$

Add the independent contributions from each site. The energy variance has units J², the mixed covariance J, and the number variance no units. Do not compare their raw numerical sizes without a specified energy scale.

$$
\Sigma=\begin{pmatrix}\varepsilon_1^2v_1+\varepsilon_2^2v_2&\varepsilon_1v_1+\varepsilon_2v_2\\\varepsilon_1v_1+\varepsilon_2v_2&v_1+v_2\end{pmatrix}.
$$

Expanding the two-by-two determinant cancels both same-site squared terms. With different energies and positive v₁,v₂ it is positive, so the covariance matrix is positive definite. Equal energies restore the rank-one case; nearly equal energies make the inverse poorly conditioned.

$$
\det\Sigma=v_1v_2(\varepsilon_1-\varepsilon_2)^2.
$$

Known unequal energies and exact interior occupancies identify β and μ in this model. These expressions do not guarantee accurate finite-sample estimates. Near saturation the logit is sensitive to errors; empirical zero or one needs statistical treatment rather than substituting an infinite logit as a measured truth.

$$
\ell_i=\log\frac{r_i}{1-r_i}=\beta(\mu-\varepsilon_i),\qquad\beta=\frac{\ell_1-\ell_2}{\varepsilon_2-\varepsilon_1},\quad\mu=\varepsilon_1+\ell_1/\beta.
$$

Now change to the physical parameter coordinates (β,μ) with energies fixed. Centered scores have mean zero. Their covariance gives the Fisher information for one independently prepared observation of the full labeled state, equivalently the sufficient pair (E,N).

$$
\partial_\beta\log p=-(E-\mu N)+\langle E-\mu N\rangle,\qquad\partial_\mu\log p=\beta(N-\langle N\rangle).
$$

The Jacobian explains both the covariance transformation and its determinant. Positive determinant is local identifiability in the stated regular finite model. It is not a promise that a chosen estimator attains the inverse-Fisher bound or that information in the count alone is identical to full-state information.

$$
J=\frac{\partial(-\beta,\beta\mu)}{\partial(\beta,\mu)}=\begin{pmatrix}-1&0\\\mu&\beta\end{pmatrix},\qquad\mathcal I_{\beta,\mu}=J^{\mathsf T}\Sigma J,\quad\det\mathcal I=\beta^2v_1v_2(\varepsilon_1-\varepsilon_2)^2.
$$

A common per-particle energy-reference shift leaves the occupancy law and determinant unchanged. If that reference is unknown, absolute μ cannot be identified from occupancy alone. The calibrated-energy assumption is essential; varying two unknown energies does not remove their common-shift ambiguity.

$$
\varepsilon_i\mapsto\varepsilon_i+c,\quad\mu\mapsto\mu+c:\qquad r_i\mapsto r_i,\quad E\mapsto E+cN,\quad\det\Sigma\mapsto\det\Sigma.
$$

### Read the comparison

The upper panel plots the determinant as the second calibrated energy separates from the first, at fixed β and μ. The lower panel shows the energy/number support: equal energies put every point on one line; unequal energies open a second direction. Point coordinates indicate possible states, not a trajectory.

![An unequal calibrated energy gap opens the energy-number covariance determinant and separates states that were collinear in the identical-energy model.](/assets/learning/ensembles-grand-response-rank.en.svg)

### Reproduce the example

Use a known reference energy E₀, ε₁/E₀=1 and μ/E₀=0. Energies, β and μ in the code are scaled by E₀. The covariance determinant is divided by E₀²; the Fisher determinant for the coordinates (βE₀,μ/E₀) is dimensionless. A dash is an undefined inverse for equal energies, not zero information in every direction.

| Second energy | Inverse temperature | Covariance determinant | Fisher determinant | Recovered beta | Recovered mu |
| --- | --- | --- | --- | --- | --- |
| 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 | -- | -- |
| 2.00000000 | 0.50000000 | 0.04620453 | 0.01155113 | 0.50000000 | -0.00000000 |
| 2.00000000 | 1.00000000 | 0.02064299 | 0.02064299 | 1.00000000 | -0.00000000 |
| 2.00000000 | 2.00000000 | 0.00185447 | 0.00741788 | 2.00000000 | 0.00000000 |

```python
from math import exp, log
for e2,beta in [(1,1),(2,0.5),(2,1),(2,2)]:
    e1,mu = 1,0
    r1 = 1/(1+exp(beta*(e1-mu)))
    r2 = 1/(1+exp(beta*(e2-mu)))
    determinant = r1*(1-r1)*r2*(1-r2)*(e2-e1)**2
    fisher = beta**2*determinant
    recovered_beta = recovered_mu = "--"
    if e2 != e1:
        l1,l2 = log(r1/(1-r1)),log(r2/(1-r2))
        b = (l1-l2)/(e2-e1)
        recovered_beta = f"{b:.8f}"
        recovered_mu = f"{e1+l1/b:.8f}"
    print(f"{e2:.8f} {beta:.8f} {determinant:.8f} {fisher:.8f} "
          f"{recovered_beta} {recovered_mu}")
```

### Three research questions with explanations

#### 1. Does a singular covariance mean no thermal fluctuations?

**Explanation.** No. Identical-energy occupancy can have positive number variance, but E=εN makes the two observables redundant. One observable direction fluctuates; its linearly independent complement has zero variance.

#### 2. Why require known different energies?

**Explanation.** Different energies make the logit difference sensitive to β; the calibrated energy reference then fixes μ. If a common energy offset is unknown, shifting it together with μ leaves all occupancies unchanged. Full rank for the calibrated two-parameter problem does not solve that larger unknown-reference problem.

#### 3. Can the determinant alone certify useful estimation accuracy?

**Explanation.** No. Units, eigenvalue scales, sample size, saturation and estimator bias still matter. Nearly equal levels or nearly deterministic occupancies can give poor conditioning even when the determinant is positive. A Fisher statement needs its observation model and regularity assumptions.

### Source and research boundary

The grand-canonical normalization and response background is [Tong, section 1.4.1](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf). The heterogeneous two-site model, rank formula and score-coordinate transformation are direct finite calculations here. A covariance determinant is not an equation of state for an unspecified gas or a kinetic particle-exchange rate.

## Practice with answers and a verification checklist

Specify which variables are fixed, which quantities fluctuate and which sample space each probability uses. Keep physical units attached to response and covariance.

### Exercise 1: Condition a balanced open preparation

For M=2 and μ=ε at finite positive T, find the count law, the probability of pattern 01, its probability given N=1, the mean and the number variance.

**Answer:** The count law is (1,2,1)/4. Pattern 01 has probability 1/4, while N=1 has probability 1/2; its conditional probability is therefore 1/2. The mean is 1 and variance is 1/2. Fixing the observed count to one would remove number variance but retain the two possible occupied-site labels.

### Exercise 2: Restore chemical-response units

For M=2 and y=−ln 2, calculate the mean, number variance, susceptibility, energy/number covariance and grand potential.

**Answer:** The mean is 2/3 and number variance 4/9. The susceptibility is 4/(9kBT), with units 1/J. The covariance is 4ε/9, with units J. Since Ξ=(3/2)^2, Ω=−2kBT ln(3/2), in joules. Substituting variance alone for susceptibility would omit the inverse thermal-energy factor.

### Exercise 3: Compare a local slope with a finite change

For M=4 at μ=ε and fixed positive T, increase μ by 0.1kBT. Predict the mean-count change to first order, then calculate it from the exact occupancy law.

**Answer:** Initially Var(N)=1, so χ_N=1/(kBT) and the first-order change is 0.1. The exact new bias is y=0.1, giving change 4/[1+exp(−0.1)]−2≈0.09991675. A derivative is a local response, and a finite change need not equal its linear prediction exactly. No particle-arrival time follows from either calculation.

### Exercise 4: Apply a common reference change

Keep M and T fixed. Change ε=2 J, μ=1 J to ε'=10 J, μ'=9 J. Which preparation quantities stay the same, and does physical mean energy stay fixed?

**Answer:** The difference μ−ε remains −1 J, so all occupancy probabilities, number moments, entropy, ln Ξ and Ω stay the same. Mean energy changes by 8 J times 〈N〉. Energy variance changes with the squared per-particle energy scale; it is not invariant under this per-particle reference change when particle number fluctuates.

### Exercise 5: Approach the same threshold along two paths

For M=2, compare fixed μ=ε as T→0+ with μ(T)=ε+(ln 3)kBT. Find the limiting count laws, means and variances.

**Answer:** The fixed-threshold path has r=1/2, count law (1,2,1)/4, mean 1 and variance 1/2. The second path has r=3/4, count law (1,6,9)/16, mean 3/2 and variance 3/8. Both reach (0,ε) in parameter space but retain different limiting distributions. The literal zero-temperature threshold prescription must therefore state its path.

### Exercise 6: Keep a finite potential when the partition diverges

For M=2, ε=1 J, μ=2 J and T→0+, determine the limiting particle law, energy, entropy, ln Ξ and Ω. Explain the status of the chemical response away from the threshold.

**Answer:** The fully occupied pattern is selected, so N=2, U=2 J and S=0. The logarithm of Ξ diverges to positive infinity, but Ω tends to M(ε−μ)=−2 J. The same value follows from U−μN. Locally away from μ=ε the exact-zero-temperature count is constant, giving zero μ response. This case is different from the threshold, where no finite derivative exists; do not calculate the potential by multiplying 0 and infinity.

Before trusting a diagram, verify normalization, restore within-count labels, check chemical response units and distinguish finite changes from derivatives. Mark zero-temperature paths explicitly and retain positive particle and hole tails. The subsequent partition-function and distribution chapters will generalize these tools; continuous gases, interactions and dynamical exchange remain further work.

## Connect the calculation to the experiment

Start with **Balanced occupancy, mu=epsilon**, then set the inspected particle count N to 1. For M=2 the count probability is 1/2, whereas either labeled pattern 01 or 10 has unconditional probability 1/4 and conditional probability 1/2. The pattern inspector conditions on your selected count; it does not clamp the grand-canonical preparation to that count.

| Experiment | Comparison | What to explain |
| --- | --- | --- |
| Zero temperature, empty pattern → threshold → fully occupied | Hold ε=1 J and vary μ through 0,1,2 J at T=0 | The threshold retains count uncertainty and an undefined finite chemical derivative |
| Chemical response beyond probability range | Compare occupancy with ∂E[N]/∂μ and its 1/J unit | Multiplying a tiny positive variance by 1/(kBT) can restore a representable response |
| Bounded negative-temperature response | Keep the finite exclusion spectrum and inspect response sign | Nonnegative Var(N) accompanies a negative chemical response when T<0 |

For a finite nonzero T, hold M,ε,T fixed and perturb μ on both sides of a reference value. The change in mean occupancy divided by the chemical-potential change approximates βVar(N). Do not change T simultaneously: that follows a different path. The threshold at T=0 has no finite derivative of this kind.

As a second check, add the same finite offset to ε and μ, keeping ε nonnegative. The occupancy law depends on their difference and therefore stays the same; physical energy U and its covariance with particle count change. Ω refers to E−μN and stays the same for this common per-particle reference shift. Keep the units and reference convention visible when comparing two preparations. Comparison playback specifies independent distributions, not particle transport in time.
