## A specified temperature does not fix every measured energy

Prepare two labeled two-level sites in contact with a reservoir, then measure their total excitation energy. Repeating the preparation can give zero, one or two excitations. A specified reservoir temperature determines the probabilities of those outcomes; it does not require every repetition to return their mean.

The previous finite-reservoir chapter derived an exact subsystem marginal under a fixed total excitation budget. Here we specify a canonical preparation directly. This is a different probability law, even when its mean agrees with a fixed-budget comparison. The new questions concern the full energy distribution, its fluctuations, and the response of the mean to temperature.

Keep three kinds of spread separate: fluctuations between individual preparations, the relative size of those fluctuations compared with the mean, and the uncertainty of an average computed from many independent preparations. Increasing the number of independent repetitions changes the third question, not the physical temperature or ensemble variance. The reader plots the theoretical law; the repetition count M is a separate calculation, not a sampling control in this experiment.

## Define the Hamiltonian and the control variables

There are n distinguishable, noninteracting binary sites. A site's energies are 0 and a fixed gap ε≥0 J. For a finite nonzero temperature T, define inverse temperature β and the dimensionless scaled gap t:

$$
x\in\{0,1\}^{n},\qquad K(x)=\sum_{i=1}^{n}x_i,\qquad E(x)=\varepsilon K(x),\qquad \beta=\frac1{k_BT},\quad t=\beta\varepsilon.
$$

The Hamiltonian specifies which energy belongs to each labeled pattern. A probability preparation is an additional choice. In particular, noninteracting energies alone do not establish independence in an arbitrary fixed-budget preparation. Independence here follows from the product form of the canonical law for this Hamiltonian.

During a temperature response, keep n and ε fixed. Changing an external field that changes ε is a different protocol. The calculator supports integer n from 0 through 4096, finite ε≥0 and finite temperatures of either sign. Its T=0 convention will be stated explicitly below. The finite-state resource bound is not a thermodynamic-limit theorem. Canonical normalization and fluctuation relations are introduced in [Tong, sections 1.3.1–1.3.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf); the worked examples here use the specified binary Hamiltonian.

## Normalize one site before multiplying copies

One site's Boltzmann weights are 1 and exp(−t). For the full labeled system the partition function factorizes because both the configuration space and the energy separate:

$$
z_1=1+e^{-t},\qquad Z=z_1^n,\qquad \ln Z=n\ln(1+e^{-t}).
$$

The excitation probability r and its complement are:

$$
r=\frac{e^{-t}}{1+e^{-t}}=\frac1{1+e^t},\qquad 1-r=\frac1{1+e^{-t}}.
$$

For positive T and positive ε, r is below one half. For any finite nonzero T every labeled pattern has positive probability. A large positive t makes excited patterns rare, but it does not impose a strict support constraint. Directly computing exp(t) can overflow even while r has a meaningful logarithm, so equivalent formulas are not equally suitable numerically.

## Distinguish a pattern from an excitation count

A pattern containing k excitations has a product probability. The count reading k aggregates C(n,k) patterns with the same energy:

$$
p(x)=r^{K(x)}(1-r)^{n-K(x)},\qquad P(K=k)=\binom nk r^k(1-r)^{n-k},\qquad p(x\mid K=k)=\binom nk^{-1}.
$$

The conditional expression applies to patterns with the stated count when that event has positive probability. For n=2 and t=ln 2, r=1/3. Enumerating the four patterns gives a complete hand check:

| Pattern | Excitations | Pattern probability | Probability of its count |
| --- | ---: | ---: | ---: |
| 00 | 0 | 4/9 | 4/9 |
| 01 | 1 | 2/9 | 4/9 |
| 10 | 1 | 2/9 | 4/9 |
| 11 | 2 | 1/9 | 1/9 |

The two rows with count 1 share one count bar; do not add that count probability twice. Given K=1, the two patterns are equally likely with conditional probability 1/2 each. That conditional preparation differs from their unconditional probability 2/9.

## Add independent occupancies to derive the moments

Each site contributes a Bernoulli variable with mean r and variance r(1−r). Independence makes the cross covariances zero in this preparation. Summing the sites gives:

$$
\mathbb E[K]=nr,\qquad \operatorname{Var}(K)=nr(1-r),\qquad U=\mathbb E[E]=n\varepsilon r,\qquad \operatorname{Var}(E)=n\varepsilon^2r(1-r).
$$

U is in joules, while energy variance is in joules squared. A standard deviation takes the square root and is again in joules. It is not interchangeable with variance. At fixed r, doubling n doubles the mean and variance, but multiplies the standard deviation by the square root of two.

The fixed shell K=k has energy variance zero because all its patterns have the same energy. Its labels can still be uncertain. A canonical preparation with the same mean count can have nonzero energy variance. Uncertainty about the pattern and fluctuations of the energy are different observables.

## Relate fluctuations to a temperature response

At fixed Hamiltonian, differentiation of the finite partition sum gives:

$$
U=-\frac{\partial\ln Z}{\partial\beta},\qquad \frac{\partial^2\ln Z}{\partial\beta^2}=\operatorname{Var}(E),\qquad \frac{\partial U}{\partial\beta}=-\operatorname{Var}(E).
$$

Use dβ/dT=−1/(kBT²). The heat capacity and the fluctuation relation become:

$$
C=\left(\frac{\partial U}{\partial T}\right)_{n,\varepsilon}=nk_Bt^2r(1-r),\qquad \operatorname{Var}(E)=k_BT^2C.
$$

Heat capacity has units J/K. The relation connects a spread across equilibrium preparations to the slope of the equilibrium mean. It does not specify a relaxation rate, a heat current or a dissipation time. Those require a dynamics and a coupling model.

If ε itself changes with temperature along a chosen path, differentiating U along that path introduces extra terms. The result is not automatically the fixed-Hamiltonian heat capacity above. A visualization should therefore label which parameter remains fixed while it draws a temperature curve.

## Compute entropy on the correct state space

The entropy of the full pattern preparation is:

$$
S=-k_B\sum_xp(x)\ln p(x)=k_B(\ln Z+\beta U)=nk_Bh(r),\qquad h(r)=-r\ln r-(1-r)\ln(1-r).
$$

The product law is uniform inside each excitation shell. Consequently, the dimensionless pattern entropy splits into the count entropy and the average hidden label multiplicity:

$$
\frac{S}{k_B}=H(K)+\sum_{k=0}^{n}P_k\ln\binom nk,\qquad H(K)=-\sum_{k=0}^{n}P_k\ln P_k.
$$

A meter that reports only K hides which sites are excited. Its entropy H(K) cannot be substituted for S/kB without the multiplicity contribution. Use natural logarithms and nats for the information quantities; multiplication by kB gives physical entropy in J/K under this preparation.

The ε=0 case illustrates the distinction especially clearly: every pattern has energy zero, yet a uniform canonical preparation over the labeled patterns has entropy nkB ln 2. Zero energy variance need not mean zero pattern entropy.

## Work through two scales before plotting a sweep

At n=2 and t=ln 2, the count law is (4,4,1)/9. Its mean count is 2/3, variance is 4/9, and C/kB=4(ln 2)²/9. The earlier finite-reservoir example n=2,b=4,K_total=2 has the same mean but variance 16/45. A successful mean-energy comparison has therefore not verified the fluctuation law.

For t=1, the single-site excitation probability is approximately 0.26894142137. For four independent sites:

$$
\frac{U}{\varepsilon}\approx1.07576568548,\qquad \frac{\operatorname{Var}(E)}{\varepsilon^2}\approx0.78644773297,\qquad \frac{S}{k_B}\approx2.32881243555,\qquad \frac{C}{k_B}\approx0.78644773297.
$$

The last two fluctuation and response numbers coincide only in these normalized units at t=1. They have different physical dimensions before normalization. To obtain SI energy, multiply U/ε by the specified ε; to obtain SI variance, multiply by ε².

Plotting the count distribution, marking its mean and showing a separate variance readout gives more information than a single moving average. Bars are discrete count events. Cycling among temperatures compares different preparations and should not be labeled a particle trajectory or physical cooling time.

## Compare absolute and relative fluctuations

For n>0 and ε>0, the relative standard deviation of excitation energy and the variance per site are:

$$
\frac{\sqrt{\operatorname{Var}(E)}}{U}=\sqrt{\frac{1-r}{nr}}=\frac{e^{t/2}}{\sqrt n},\qquad \operatorname{Var}(E/n)=\frac{\varepsilon^2r(1-r)}{n}.
$$

At a fixed preparation parameter t, the absolute energy variance grows with n, while the variance of the energy per site shrinks. The mean grows faster than the standard deviation, so the relative fluctuation decreases. These statements describe different quantities and are mutually consistent.

A large n alone is insufficient to make relative fluctuations small when the excitation probability is extremely low. The relevant mean count nr can remain small. For example, one rare excitation carries a large change relative to a tiny mean even if most preparations return zero excitations.

The relative energy fluctuation also depends on the chosen energy reference. Adding a constant to every configuration changes the mean while leaving variance unchanged. Here it compares excitation energy above the stated zero-energy configuration. It is not a reference-independent measure of all thermodynamic fluctuations.

## Separate ensemble spread from uncertainty of an average

Prepare M independent systems from the same law and average their measured energies. With a known population variance, the standard error and a distribution-free bound are:

$$
\operatorname{SE}(\overline E)=\sqrt{\frac{\operatorname{Var}(E)}{M}},\qquad \Pr(|\overline E-U|\ge\delta)\le\frac{\operatorname{Var}(E)}{M\delta^2},\quad\delta>0.
$$

The variance of a single preparation does not shrink when more repetitions are recorded. The average becomes more precise. The bound follows from independence and Chebyshev's inequality; it is not the same as asserting an exact normal confidence interval.

If excitations are very rare, a small sample can return none even though r is positive. The probability that all M systems have no excitations is exactly:

$$
\Pr(K_1=\cdots=K_M=0)=(1-r)^{nM}.
$$

A histogram with no observed tail events is not evidence that the theoretical tail probability is zero. State the sample count and seed when displaying an empirical histogram. Correlated measurements from a time series need not supply M independent samples, and this static model does not define their correlation time.

## Explain why heat capacity peaks at an intermediate temperature

For n>0, ε>0 and positive T, write the heat capacity as a function of t:

$$
\frac{C}{nk_B}=\frac{t^2}{4\cosh^2(t/2)},\qquad t_{\rm peak}\tanh(t_{\rm peak}/2)=2,\qquad t_{\rm peak}\approx2.39935728052.
$$

The peak condition follows by setting the derivative of ln C with respect to t to zero. Its positive solution gives T_peak≈0.41677828 ε/kB and C_peak≈0.43922884 nkB. This is a smooth two-level heat-capacity maximum, not a singular phase transition.

At low positive temperature, excitations are exponentially suppressed. At very high temperature, the energy distribution has a substantial variance but changing T alters the nearly balanced occupancies only weakly. The factor 1/T² in the fluctuation relation explains why high variance need not imply high heat capacity at arbitrarily high T.

A curve should identify its horizontal variable. Increasing t means decreasing positive T at fixed ε; treating t and T as the same axis reverses the interpretation of a sweep.

## Use negative temperature only with a bounded spectrum

The finite two-level system has an upper energy bound. A finite negative T is a normalizable population-inverted canonical preparation. Reversing the sign of temperature at a fixed gap gives:

$$
r(-t)=1-r(t),\qquad U(T)+U(-T)=n\varepsilon,\qquad S(T)=S(-T),\qquad C(T)=C(-T).
$$

The count distribution reflects about n/2. Energy variance is also unchanged, while the mean and relative energy fluctuation change. A negative T is not a state colder than the positive-temperature ground limit. The inverse-temperature parameter passes through zero at the balanced infinite-temperature limit.

The bounded spectrum is essential to this construction. An unrestricted kinetic-energy gas does not acquire the same normalizable negative-temperature law. [Tong's two-state discussion, section 1.2.3](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf), supplies the physical context for population inversion. Here we compare probability preparations, not a laboratory inversion protocol or its lifetime.

## State the limits before interpreting a zero

The calculator's input T=0 denotes the positive-side equilibrium limit for the specified fixed gap. It is not a finite-temperature value of β and does not describe a cooling process. For ε>0 the negative-side limit would instead select the highest-energy pattern.

| Condition | Mean energy | Energy variance | Pattern entropy | Heat capacity |
| --- | --- | --- | --- | --- |
| n=0 | 0 | 0 | 0 | 0 |
| ε=0, uniform canonical preparation | 0 | 0 | nkB ln 2 | 0 |
| ε>0, T→0+ | 0 | 0 | 0 | 0 |
| ε>0, T→0− | nε | 0 | 0 | 0 |
| ε>0, T→±infinity | nε/2 | nε²/4 | nkB ln 2 | 0 |

At ε=0 the chosen canonical limit remains uniform over the degenerate states. It is different from taking a positive gap to zero only after selecting its unique ground state at T→0+. Orders of limits and the selected preparation matter.

Relative energy fluctuation is undefined when the mean energy is zero, including the empty system and the zero-gap system. Do not turn 0/0 into zero fluctuation. Count fluctuations can still be defined for a zero-gap system because the count observation remains distinct even when its energies coincide.

## Read tail diagnostics and connect the model to AI

For large positive t, useful leading scales are:

$$
r\sim e^{-t},\qquad \ln P(K=0)=n\ln(1-r)\sim-nr,\qquad U\sim n\varepsilon e^{-t},\quad \operatorname{Var}(E)\sim n\varepsilon^2e^{-t}.
$$

The displayed probability may fall below the numeric range while multiplying it by a large physical energy scale gives a representable mean or variance. Normalizing after prematurely rounding the probability to zero would erase that information. Conversely, a probability displayed as 1 can still have a small positive complementary probability.

The numerical model preserves exact binary input products and integer multiplicities, while exponentials and logarithms are approximate. It retains logarithmic tail scales and distinguishes their availability from actual support zeros. Nested logarithms diagnose extreme tails; they do not certify tiny departures from one in moments whose unit-scale values are rounded. Independent high-precision references check the exposed probability and response scales.

For AI, temperature-controlled Bernoulli gates or independent masks offer a mathematical comparison: changing a logit changes both occupancy and variability, and averaging more independent samples changes estimator error rather than the underlying gate distribution. A softmax temperature in an arbitrary model is a dimensionless design parameter unless a physical calibration has been supplied. This analogy does not assign heat capacity or joules of dissipation to a neural-network training step.

## Research extension: equilibrium spread does not set a response speed

Two systems can have the same equilibrium energy histogram and still follow a changing bath at different speeds. Think of thermometers with identical final readings but different response times. To predict a lag we must supply a dynamics, rather than read a clock from the partition function.

Keep n independent two-level sites with fixed positive gap ε. Add positive transition-rate scale γ, in inverse seconds, and a prescribed positive bath temperature T(t). This new illustrative kinetic model is separate from the reader’s static preparation controls and comparison scan. No energy levels are driven, so there is no parameter work on the sites.

Specify the upward and downward rates, not merely their ratio. Their sum γ sets the relaxation speed; many positive γ values share exactly the same equilibrium preparation.

$$
r_{\rm eq}(T)=\frac1{1+\exp[\varepsilon/(k_BT)]},\qquad u(T)=\gamma r_{\rm eq}(T),\quad d(T)=\gamma[1-r_{\rm eq}(T)].
$$

The rate ratio satisfies instantaneous detailed balance with the specified bath, but the actual occupancy r need not equal its moving equilibrium target. For a constant bath, a deviation decays exponentially at rate γ.

$$
\frac{u(T)}{d(T)}=e^{-\varepsilon/(k_BT)},\qquad \dot r=u(1-r)-dr=\gamma(r_{\rm eq}-r).
$$

Use a small amplitude |a|/T₀ and fixed n,ε. The heat capacity C₀ is the equilibrium derivative at T₀, in J/K. Temperature positivity is required throughout the cycle. We seek the periodic response after the transient has decayed.

$$
T(t)=T_0+a\cos\omega t,\quad T_0>|a|,\quad T_0>0,\qquad C_0=n\varepsilon r_{\rm eq}^{\prime}(T_0)=\frac{n\varepsilon^2}{k_BT_0^2}r_0(1-r_0).
$$

Expand the moving target to first order. The linear equation is a driven first-order filter. The O(a²) term can contain a mean shift and a second harmonic; it must not be silently discarded in a finite-amplitude claim.

$$
\delta\dot r+\gamma\delta r=\gamma r_{\rm eq}^{\prime}(T_0)a\cos\omega t+O(a^2).
$$

Insert a complex exponential into the linear equation and use E=nεr. The real part is the physical energy change. This susceptibility has units J/K; its imaginary part is negative in this chosen exp(+iωt) convention.

$$
\delta E(t)=\operatorname{Re}\!\left[\chi_E(\omega)a e^{i\omega t}\right]+O(a^2),\qquad\chi_E(\omega)=\frac{C_0}{1+i\omega/\gamma}.
$$

Finite speed both suppresses amplitude and delays the response. At ω=γ, amplitude is C₀/√2 and the lag is 45 degrees. The static limit is C₀; high-frequency response tends to zero. An identical static C₀ says nothing about which γ the apparatus has.

$$
|\chi_E|=\frac{C_0}{\sqrt{1+(\omega/\gamma)^2}},\qquad\phi=\arctan(\omega/\gamma),\qquad\delta E=|\chi_E|a\cos(\omega t-\phi)+O(a^2).
$$

Because the gap is fixed, energy changes by heat. Independent sites retain a product preparation under this independent transition model. Their Gibbs entropy is therefore n times the one-site entropy. This is a model for heat ports, not for the external cost of controlling the bath.

$$
\dot Q_{\rm sites}=\dot E=n\varepsilon\dot r,\qquad S_{\rm sites}=nk_B h(r),\qquad h(r)=-r\log r-(1-r)\log(1-r).
$$

Use the equilibrium log-odds to rewrite the entropy budget. Both factors have the same sign because log-odds is increasing. The expression is valid for interior occupancies; support endpoints require limits. Instantaneous entropy of the sites alone need not be monotone under a changing bath.

$$
\sigma(t)=\dot S_{\rm sites}-\frac{\dot E}{T(t)}=nk_B\gamma(r_{\rm eq}-r)\log\frac{r_{\rm eq}(1-r)}{r(1-r_{\rm eq})}\ge0.
$$

In the periodic steady state the entropy returns after each cycle. Expand 1/T(t) and average the linear energy response against the temperature drive. This gives the displayed positive leading term, in J/(K s). It is not an exact finite-amplitude formula or a statement about a complete bath-controller cycle.

$$
\langle\sigma\rangle_{\rm cycle}=-\left\langle\frac{\dot E}{T(t)}\right\rangle_{\rm cycle}=\frac{C_0a^2\omega^2}{2\gamma T_0^2[1+(\omega/\gamma)^2]}+O(a^3).
$$

Normalize the curves to separate static scale from kinetic scale. The lag component peaks at x=1, while amplitude continues to fall. The last ratio uses a²; at a=0 it is interpreted as its small-amplitude limit, not division by zero.

$$
x=\omega/\gamma:\quad\frac{\operatorname{Re}\chi_E}{C_0}=\frac1{1+x^2},\quad\frac{-\operatorname{Im}\chi_E}{C_0}=\frac{x}{1+x^2},\quad\frac{T_0^2\langle\sigma\rangle}{C_0a^2\gamma}=\frac{x^2}{2(1+x^2)}+O(a/T_0).
$$

### Read the comparison

The upper panel plots the normalized amplitude and in-phase/quadrature components. The lower panel compares the temperature signal with the energy response at ω=γ. Curves describe the linear periodic model; the existing comparison playback still switches static preparations.

![A declared relaxation rate changes amplitude and phase without changing equilibrium heat capacity; the energy response lags a sinusoidal bath.](/assets/learning/ensembles-driven-canonical-response.en.svg)

### Reproduce the example

Use n=3 and ε/(kBT₀)=1. Energy is in kBT₀, temperature in T₀, and entropy in kB. With γ=1/s the final column is the leading cycle-mean entropy production divided by the squared reduced amplitude, in kB/s. The x=0 row is the zero-frequency limit; it does not require a finite zero-frequency cycle. This code evaluates the linear expressions. The independent verifier also integrates the nonlinear master equation for two small nonzero amplitudes.

| x | Re response | Minus Im response | Amplitude | Lag degrees | Production / amplitude squared |
| --- | --- | --- | --- | --- | --- |
| 0.00000000 | 0.58983580 | 0.00000000 | 0.58983580 | 0.00000000 | 0.00000000 |
| 0.50000000 | 0.47186864 | 0.23593432 | 0.52756518 | 26.56505118 | 0.05898358 |
| 1.00000000 | 0.29491790 | 0.29491790 | 0.41707689 | 45.00000000 | 0.14745895 |
| 2.00000000 | 0.11796716 | 0.23593432 | 0.26378259 | 63.43494882 | 0.23593432 |

```python
from math import exp, sqrt, atan, degrees
n = 3
r0 = 1/(1+exp(1))
C = n*r0*(1-r0)
for x in [0,0.5,1,2]:
    real = C/(1+x*x)
    lag_component = C*x/(1+x*x)
    amplitude = C/sqrt(1+x*x)
    phase = degrees(atan(x))
    production = C*x*x/(2*(1+x*x))
    print(f"{x:.8f} {real:.8f} {lag_component:.8f} "
          f"{amplitude:.8f} {phase:.8f} {production:.8f}")
```

### Three research questions with explanations

#### 1. Can energy variance determine the delay?

**Explanation.** No. Equilibrium variance determines C₀ under the stated canonical conditions, but changing γ leaves that equilibrium law unchanged. The new transition rates supply the missing kinetic information.

#### 2. Why is there heat but no level-driving work?

**Explanation.** The Hamiltonian levels stay fixed. Changing their occupation changes energy through the declared bath transitions. Changing ε instead would add a parameter-work term and describe a different experiment. The work needed to operate a bath controller is outside these site equations.

#### 3. Is the leading production expression exact at any amplitude?

**Explanation.** No. The response was linearized and the entropy budget expanded. Nonlinear harmonics and higher orders matter at larger amplitudes. The small-amplitude numerical comparisons test particular finite cases rather than certify every frequency, amplitude or integration step.

### Source and research boundary

The equilibrium fluctuation background is [Tong, section 1.3.2](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf). The positive rates, sinusoidal protocol, filter response and cycle budget are derived here for the declared model. Equilibrium normalization alone does not supply those rates or prove a physical apparatus realizes them.

## Practice with answers and a verification checklist

Keep population quantities separate from sample estimates, and keep the Hamiltonian fixed during response calculations. Use exact fractions whenever the chosen odds allow them.

### Exercise 1: Resolve a count bar into labeled patterns

For n=2 and t=ln 2, calculate the probabilities of pattern 01, count K=1, and pattern 01 conditional on K=1.

**Answer:** The excitation probability is 1/3. Pattern 01 has probability (2/3)(1/3)=2/9. The count also includes 10, giving 4/9. Their ratio gives conditional probability 1/2. Equal probabilities within a selected count do not make all four unconditional patterns equiprobable.

### Exercise 2: Restore units in a fluctuation calculation

For n=4 and t=1, calculate U/ε, Var(E)/ε², S/kB and C/kB. Which two numbers agree here, and why are they not the same physical quantity?

**Answer:** With r≈0.26894142137, the results are approximately 1.07576568548, 0.78644773297, 2.32881243555 and 0.78644773297. The variance and heat-capacity numbers agree after the stated normalization because t=1. Variance in SI has units J², while heat capacity has units J/K; Var(E)=kBT²C supplies the conversion.

### Exercise 3: Change the number of independent sites

At the same ε and T, increase n from 4 to 16. Give the change factors for mean energy, energy variance, standard deviation and relative fluctuation.

**Answer:** Mean energy and variance each increase by four. Standard deviation increases by two. Relative fluctuation decreases by two because the mean increased faster. Entropy and heat capacity also increase by four. A smaller relative fluctuation does not require the absolute spread to decrease.

### Exercise 4: Plan independent repetitions

For n=2 and t=ln 2, how many independent preparations make the standard error of the mean count at most 1/30? For M=3, what is the exact probability of observing no excitation in any system?

**Answer:** Var(K)=4/9, so sqrt[4/(9M)]≤1/30 requires M≥400. This is a standard-error requirement, not an exact 95% confidence statement. For three systems the no-excitation probability is (2/3)^6=64/729. A zero observed tail count therefore does not establish a zero theoretical tail probability.

### Exercise 5: Invert the bounded preparation

For n=2, compare t=ln 2 and t=−ln 2. Give both count distributions, means and variances, and compare entropy and heat capacity.

**Answer:** The count laws are (4,4,1)/9 and (1,4,4)/9. The means are 2/3 and 4/3, while both variances are 4/9. Pattern entropy and heat capacity are unchanged. The relative count standard deviations are 1 and 1/2, respectively. Their product is 1/n, consistent with reversing the odds. The inverted preparation uses the bounded spectrum; it is not an ordinary gas at a colder temperature.

### Exercise 6: Distinguish a missing number from a forbidden outcome

Take n=2, ε=10^280 J and a finite positive temperature chosen so t≈800. Can an unrepresentable excitation probability coexist with representable U and Var(E)? Contrast this with ε>0 at the explicit T=0 input and with ε=0.

**Answer:** At finite t, r is positive and approximately exp(−800). The leading scales U≈2×10^280 exp(−800) J and Var(E)≈2×10^560 exp(−800) J² are representable even though r is below the ordinary linear display range. At the positive-side T=0 limit with ε>0, excitation probability and energy variance are true zeros. At ε=0, the uniform preparation instead has entropy 2kB ln 2 and count variance 1/2, despite zero energy and energy variance. These are three different cases.

Before trusting a plot, check normalization, distinguish pattern and count probabilities, verify both moments, and check the fixed-Hamiltonian response relation. Label sample count separately from site count. Preserve positive tails and undefined ratios. The next particle-exchange chapter will change which quantity is fixed; interactions, continuous systems and dynamical relaxation remain further topics.

## Connect the calculation to the experiment

Select **Worked example t=ln 2**. First predict the three count probabilities (4,4,1)/9. Read the distribution table to check them, then compare Var(K)=4/9 with C/kB=(ln 2)²·4/9. Their different factors are the fluctuation-response conversion, not inconsistent answers.

| Experiment | Comparison | What to explain |
| --- | --- | --- |
| Four sites, t=1 → Same preparation, sixteen sites | Hold ε and T fixed; multiply n by four | Mean and variance grow fourfold, but relative fluctuation halves |
| Positive-side zero-temperature limit → Exact zero gap | Compare ε>0,T=0 with ε=0,T=0 | A unique occupied energy minimum and a degenerate uniform preparation have different count uncertainty |
| Energy rescued beyond probability range | Inspect probability, U, Var(E), units and log columns together | A positive tail may be unrepresentable while its energy-weighted moments remain representable |

The controls change n,ε,T and show the theoretical count law. They do not draw M random systems or expose a sample-count or random-seed control. The M-repetition standard error in Exercise 4 is a separate calculation from Var(K): changing M changes the uncertainty of an estimated mean, whereas changing n changes the system itself. Comparison playback switches theoretical preparations and is not a simulation of physical relaxation.

For your own check, hold ε fixed and change T slightly near a finite nonzero reference temperature. Compare the resulting change in U with C times the temperature change, including their units. Use progressively smaller changes only while numerical resolution allows it. This local response test does not apply across T=0, and changing ε at the same time measures a different protocol.
