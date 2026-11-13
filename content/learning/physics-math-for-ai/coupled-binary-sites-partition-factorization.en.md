## Two switches can reveal the meaning of independence

Imagine two switches, each pointing down or up. We can record four possible joint configurations, and we can count how frequently each switch points up. Does knowing the two individual frequencies tell us the joint frequencies? Only if the switches are independent. A shared preference for agreeing, or a rule that permits only opposite settings, changes that answer.

This chapter uses two labeled binary sites as a complete, small statistical system. Four states are enough to distinguish interaction, a support constraint, correlation, information and response. We enumerate every state, so no sampling approximation hides the mechanism. The model is a classical Ising pair; it does not derive quantum spin or describe a large lattice transition. See [Tong's introduction to the Ising model](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S5.html#S5.S2) for the broader physical setting. All pair calculations and counterexamples below are derived from our four-state preparation.

## Write the Hamiltonian and count the states

Let each spin be either −1 or +1. Define alignment A and total magnetization M. The coupling J and field h are energies in joules; these spin variables and their sums are dimensionless. Start with finite positive temperature and the full Cartesian product of the two state sets:

$$
\sigma_1,\sigma_2\in\{-1,+1\},\qquad A=\sigma_1\sigma_2,\qquad M=\sigma_1+\sigma_2,\qquad E=-JA-hM,\qquad \beta=\frac1{k_BT}.
$$

| Configuration | Alignment A | Magnetization M | Energy E |
| --- | --- | --- | --- |
| (−1,−1) | +1 | −2 | −J+2h |
| (−1,+1) | −1 | 0 | +J |
| (+1,−1) | −1 | 0 | +J |
| (+1,+1) | +1 | +2 | −J−2h |

The middle rows have equal energies but are different labeled configurations. Both contribute to the sum. At positive temperature, J>0 rewards agreement because agreeing lowers the coupling energy. Positive h rewards up spins. A negative J rewards disagreement. These are statements about relative weights under the specified preparation, not a force law that tells us how quickly a spin flips.

## Add weights before interpreting probabilities

Adding the four Boltzmann weights gives a closed expression for the normalizer. The hyperbolic cosine simply pairs two exponentials; it is not an approximation:

$$
Z=2e^{\beta J}\cosh(2\beta h)+2e^{-\beta J},\qquad
p(\sigma_1,\sigma_2)=\frac{e^{\beta JA+\beta hM}}{Z}.
$$

Every allowed configuration has positive probability at finite nonzero temperature. A very small displayed probability is therefore different from a forbidden configuration. The field affects the agreeing states differently, while both disagreeing states retain the same weight. Exchanging the two site labels leaves this preparation unchanged, so their marginal means agree.

For h=0, global spin reversal pairs equally weighted configurations. Both mean spins vanish, yet agreement and disagreement can still have different total probabilities. A zero average arrow does not tell us whether the two arrows coordinate.

## Derive factorization from both energy and support

Set J=0 while keeping all four states. The energy is now a sum of one-site terms. Because each first-site value can occur with each second-site value, the double sum splits:

$$
Z\big|_{J=0}=\left(\sum_{\sigma_1=\pm1}e^{\beta h\sigma_1}\right)
\left(\sum_{\sigma_2=\pm1}e^{\beta h\sigma_2}\right)
=[2\cosh(\beta h)]^2=Z_1^2,\qquad p=p_1p_2.
$$

There are two ingredients: additive energy and product support with the specified counting measure. Factoring the exponential alone does not justify separating a sum over a constrained set. For example, the diagonal of a square is not the whole square even when a function on it can be written as a product.

The equality of one scalar normalizer at one parameter setting is also weaker than independence. Independence is a property of the normalized joint table. To diagnose it, compare the table with the product of its own marginals, rather than relying on an accidental equality between two sums.

## A hand calculation with correlated, unbiased spins

Choose h=0 and βJ=ln2. The agreeing states have weight 2 each; the disagreeing states have weight 1/2 each. Normalize the individual rows:

$$
Z=5,\qquad (p_{--},p_{-+},p_{+-},p_{++})=(2/5,1/10,1/10,2/5),\qquad
\langle A\rangle=3/5,\quad\langle M\rangle=0.
$$

Each spin is up half the time. If they were independent, every joint state would have probability 1/4. Instead, observing the first spin makes the matching value of the second more likely: conditional agreement has probability 4/5. Their covariance is 3/5, because the individual means are zero.

The ideal βJ=ln2 relation is a hand specification. A stored binary64 J equal to the computed product kB·ln2 is only a nearby numerical preparation. Our independent reference evaluates the actual stored values, and the ideal fractions illustrate the derivation rather than replace that reference.

## A constraint can create dependence without coupling

Now permit only configurations satisfying M=0. Even at J=0, the two surviving states are opposites. More generally, both have energy J, so the field has no effect at any finite temperature:

$$
\Omega_0=\{(-1,+1),(+1,-1)\},\qquad
Z_{\Omega_0}=2e^{-\beta J},\qquad p_{-+}=p_{+-}=1/2,\qquad
\langle A\rangle=-1,\quad\operatorname{Cov}(\sigma_1,\sigma_2)=-1.
$$

| Preparation at J=h=0 | Allowed configurations | Normalizer | Spin covariance |
| --- | --- | --- | --- |
| Full product | All four | 4 | 0 |
| Fixed zero sum | Only the two opposite pairs | 2 | −1 |
| Product of constrained marginals | All four | A normalized comparison law | 0 |

Both constrained marginals are uniform, but their product puts probability on configurations the actual system forbids. This is why constraints must accompany the Hamiltonian in the model description. Changing support is a change of preparation, not a small numerical perturbation of the energy.

## The log normalizer generates feature averages

Introduce dimensionless sources θ1=βJ and θ2=βh. At fixed support, differentiate the finite sum with respect to either source. The quotient rule then identifies the gradient and Hessian:

$$
\theta=(\beta J,\beta h),\quad f=(A,M),\qquad
\partial_{\theta_a}\ln Z=\langle f_a\rangle,\qquad
\partial_{\theta_a}\partial_{\theta_b}\ln Z=\operatorname{Cov}(f_a,f_b).
$$

For example, differentiating a normalized average brings down a feature from the numerator and subtracts the change of normalization. Those are exactly the two terms of covariance. This illustrates the exponential-family differential identities discussed in [Fithian's course reader](https://stat210a.berkeley.edu/fall-2024/reader/exponential-families.html#differential-identities).

The Hessian is positive semidefinite because a quadratic form in it is the variance of the corresponding linear combination of A and M. Its off-diagonal entry may be negative. Positive semidefinite does not mean every matrix entry is nonnegative. On the constrained support, A=−1 and M=0 are constants, so this entire source Hessian is zero although the spins are perfectly dependent. The choice of features matters.

## Translate source curvature into physical response

Vary J or h while holding temperature and support fixed. The chain rule contributes a factor β. For the ideal hand example, the source Hessian is particularly simple:

$$
\left.\frac{\partial\langle A\rangle}{\partial J}\right|_{T,h}=\beta\operatorname{Var}(A),\qquad
\left.\frac{\partial\langle M\rangle}{\partial h}\right|_{T,J}=\beta\operatorname{Var}(M),\qquad
\left.\frac{\partial\langle A\rangle}{\partial h}\right|_{T,J}=\beta\operatorname{Cov}(A,M).
$$

$$
\left.\nabla_\theta^2\ln Z\right|_{\beta J=\ln2,\,h=0}
=\begin{pmatrix}16/25&0\\0&16/5\end{pmatrix}.
$$

To check the first entry, A²=1 and its mean is 3/5, giving 1−9/25. To check the second, M²=4 in the agreeing rows and zero elsewhere; the total agreement probability is 4/5. The cross moment vanishes by spin reversal.

These responses have inverse-energy units, since the observables are dimensionless and J,h are energies. They describe changes between equilibrium preparations. They do not specify a dynamical susceptibility at a driving frequency, a transition rate or a mixing time.

## Compare factorization error with mutual information

A useful comparison is the signed log-normalizer difference Δ=ln Z−2ln Z1, where Z1 uses the same field and temperature with no coupling. Mutual information asks a different question: how far is the actual joint law from the product of its own marginals?

$$
\Delta=\ln Z-2\ln[2\cosh(\beta h)],\qquad
I(\sigma_1;\sigma_2)=\sum_{\sigma_1,\sigma_2}p\ln\frac{p}{p_1p_2}
=H(p_1)+H(p_2)-H(p).
$$

Natural logarithms give information in nats. Mutual information is nonnegative, and vanishes precisely for independence. The comparison Δ can have either sign, and changes when the energy reference of the interacting system changes. A common energy shift leaves p and its mutual information unchanged. Thus Δ is not an information divergence or a general-purpose dependence score.

For the unbiased hand example, the product of the actual marginals is uniform. Substitution produces two distinct numbers:

$$
\Delta=\ln(5/4)\approx0.22314355,\qquad
I=\frac45\ln\frac85+\frac15\ln\frac25\approx0.19274476.
$$

Under the fixed-zero-sum constraint at J=h=0, Δ=−ln2 while I=ln2. A negative Δ is therefore perfectly compatible with strong dependence. In that constrained law, one observed spin determines the other exactly, accounting for one binary unit, or ln2 nats, of shared information.

## Use the joint table to determine the correlation sign

For a positive two-by-two probability table, expanding covariance gives four times the difference between its diagonal products. Our energies also yield a simple odds ratio:

$$
\operatorname{Cov}(\sigma_1,\sigma_2)=4(p_{++}p_{--}-p_{+-}p_{-+}),\qquad
\frac{p_{++}p_{--}}{p_{+-}p_{-+}}=e^{4\beta J}.
$$

The common field cancels in this ratio. On full support, the covariance sign is therefore the sign of βJ. At positive temperature it follows J; at negative temperature it reverses. Covariance can become extremely small under a strong field without becoming exactly zero. Replacing a small signed value by zero would erase a real distinction between weak dependence and independence.

Negative temperatures normalize mathematically here because the finite energy list is bounded. The covariance Hessian remains positive semidefinite, but multiplying its diagonal by a negative β reverses the corresponding physical response sign. This finite mathematical example alone does not establish how to prepare a material at negative absolute temperature.

## Treat zero temperature as a separate limiting preparation

At T=0 the model means the positive-temperature limit with fixed J,h and support. It places equal probability on the minimum-energy labeled configurations. This is a specified equilibrium limiting convention, not a guarantee that any cooling protocol mixes between degenerate ground states.

For J>0,h=0, two agreeing ground states remain and I=ln2. For J<0,h=0, two opposite ground states remain and again I=ln2. A unique ground configuration instead gives I=0: two deterministic variables have no uncertainty left to share. Neither statement requires evaluating an infinite β as an ordinary floating-point input.

At J=−a,h=a with a>0, three states share the ground energy. Their source covariance limit is

$$
(p_{--},p_{-+},p_{+-},p_{++})=(0,1/3,1/3,1/3),\qquad
\langle A\rangle=-1/3,\quad\langle M\rangle=2/3,\qquad
\operatorname{Cov}(A,M)=\operatorname{Var}(A)=\operatorname{Var}(M)=8/9.
$$

Changing the field across this crossing selects different ground sets, so the zero-temperature magnetization is not differentiable there. The finite source covariance limit is not a finite physical derivative. There is a subtler warning at J=h=0: Cov(A,M)=0 in the uniform ground law, but any nonzero field selects an agreeing state and changes mean alignment from zero to one. Multiplying the zero covariance by an infinite β cannot define that derivative.

## Separate exact zeros, tiny positives and unresolved values

The numerical model assembles energies from J and h as exact binary combinations. It can preserve an input contribution even when computing a floating-point 2h first would overflow. It enumerates only the allowed states, then marks the forbidden rows separately. A forbidden row and a finite-temperature rare row must never share the same mathematical status.

Directed exponential and logarithm intervals propagate to means, covariances and marginal entropies. Covariance uses a state-pair identity, and equal exact energy costs are grouped before summing signed coefficients. This can prove a symmetry zero without an arbitrary tolerance:

$$
\operatorname{Cov}(f,g)=\frac{1}{Z^2}\sum_{i<j}w_iw_j(f_i-f_j)(g_i-g_j),\qquad w_i=e^{-\beta E_i}.
$$

| Result status | Meaning | Interpretation |
| --- | --- | --- |
| Structural zero | An exact identity or support argument proves zero | Independence requires additional conditions |
| Nonzero outside display range | Sign or logarithmic magnitude survives, ordinary value does not | Do not display a mathematical zero |
| Interval resolved | A directed interval certifies the sign and target relative width | Projection into binary64 is a separate operation |
| Precision ceiling unresolved | The requested relative accuracy was not established | Keep uncertainty visible |

Entropy differences can nearly cancel when dependence is weak. The model raises precision up to its explicit ceiling rather than subtracting displayed entropies or clipping a negative residual to zero. KL nonnegativity can constrain an enclosure, but an enclosure touching zero does not establish a positive value to the requested relative precision. Extremely cold leading-moment approximations retain their separate approximation label.

## Connect the pair to learning an energy model

Suppose observations are configurations and the trainable scores are θ1A+θ2M. The log likelihood for one observation is a score minus the log normalizer. Its source gradient is the observed feature minus the model mean; its negative Hessian is the feature covariance:

$$
\ell(\theta;\sigma)=\theta\cdot f(\sigma)-\ln Z(\theta),\qquad
\nabla\ell=f(\sigma)-\langle f\rangle,\qquad
-\nabla^2\ell=\operatorname{Cov}(f).
$$

For many observations, average the observed features. Training then compares empirical agreement and magnetization with the model predictions. On fixed-zero-sum data with that same constrained model, both features are constants and their source curvature is zero. Those parameters cannot be identified from the normalized distribution: changing J changes a common energy but not either probability, and h multiplies an always-zero feature.

This is a useful research diagnostic before applying a larger optimizer. Ask which configurations are available, which features vary on that support, and which parameter combinations change the law. A static normalizer still does not provide a sampling algorithm. In a large model, computing the expectation and obtaining representative samples become additional tasks; the four-state example isolates their algebraic target.

## Research extension: sum out one spin without losing its thermodynamic contribution

Hiding a variable does not remove its possible fluctuations. Think of observing only the two ends of a short chain while the middle spin remains free to move. We can predict the visible joint law exactly, yet an energy variance computed from visible labels alone can miss uncertainty inside each visible state.

Add a separate comparison system to the original two-spin lesson: an open three-spin chain, no field, equal positive fixed coupling J, and positive finite inverse temperature β. This is an eight-state equilibrium calculation, not a new reader control, a time evolution or a thermodynamic-limit phase-transition claim.

J is in joules and b is dimensionless. Both bonds share the middle spin, and there is no direct endpoint bond in the original Hamiltonian. We retain the endpoint observation y and sum over the hidden middle label.

$$
\sigma_i\in\{-1,+1\},\quad H=-J(\sigma_1\sigma_2+\sigma_2\sigma_3),\quad b=\beta J,\qquad y=(\sigma_1,\sigma_3).
$$

The visible weight counts both hidden completions. Aligned endpoints have weight 2 cosh(2b); opposite endpoints have weight 2. This exact marginalization already creates an apparent endpoint preference.

$$
W_y(\beta)=\sum_{\sigma_2=\pm1}e^{\beta J\sigma_2(\sigma_1+\sigma_3)}=2\cosh[b(\sigma_1+\sigma_3)].
$$

Match the aligned and opposite weights to solve for both the induced coupling k and the additive log-weight a. The common factor cancels in normalized endpoint probabilities, but it must be retained to reproduce the physical partition function and its derivatives.

$$
W_y=e^{a+k\sigma_1\sigma_3},\qquad k=\tfrac12\log\cosh(2b),\quad a=\log2+k.
$$

Summing four endpoint weights reproduces all eight original states. Endpoint correlation is nonzero for b>0 even though no direct endpoint interaction was specified. An effective statistical coupling is not a newly discovered microscopic bond.

$$
Z=\sum_yW_y=4e^a\cosh k=8\cosh^2b,\qquad\langle\sigma_1\sigma_3\rangle=\tanh k=\tanh^2b.
$$

Fᵧ is a conditional free-energy-like potential, with a temperature dependence caused by the hidden sum. Its value is not generally the conditional mean physical energy. Differentiating βFᵧ, rather than treating Fᵧ as a fixed microscopic energy, restores the correct observable.

$$
F_y(\beta)=-\beta^{-1}\log W_y,\qquad\mathbb E[H\mid y]=\partial_\beta(\beta F_y)=-\partial_\beta\log W_y.
$$

Condition on the endpoints and average the two hidden spin values. Opposite endpoints force H=0 because the bond energies cancel. Aligned endpoints still leave two different energies at finite temperature, so their hidden conditional variance is positive.

$$
\bar H_y=-J(\sigma_1+\sigma_3)\tanh[b(\sigma_1+\sigma_3)],\quad\operatorname{Var}(H\mid y)=\frac{J^2(\sigma_1+\sigma_3)^2}{\cosh^2[b(\sigma_1+\sigma_3)]}.
$$

This is the law of total variance. The first term describes variation between observed endpoint states; the second describes energy uncertainty remaining inside them. Both refer to the original physical Hamiltonian, not the variance of an arbitrarily chosen effective endpoint energy.

$$
\operatorname{Var}(H)=\operatorname{Var}_{y}(\bar H_y)+\mathbb E_y[\operatorname{Var}(H\mid y)].
$$

The aligned class has conditional mean −2J tanh(2b), while the opposite class has mean zero. The variance of this two-valued conditional mean yields the displayed between-class term. At b=0 the conditional mean is zero in both classes.

$$
P_+=\frac{\cosh(2b)}{1+\cosh(2b)},\quad P_-=1-P_+,\qquad\operatorname{Var}_y(\bar H_y)=4J^2\tanh^2(2b)P_+P_-.
$$

Add the hidden term to recover the full energy variance and fixed-Hamiltonian heat capacity. An endpoint histogram can be exact while an energy-response calculation that drops the hidden contribution is wrong. At b=0 the entire energy variance is hidden from the conditional mean.

$$
\mathbb E_y[\operatorname{Var}(H\mid y)]=\frac{4J^2P_+}{\cosh^2(2b)},\qquad\operatorname{Var}(H)=\frac{2J^2}{\cosh^2b},\quad C/k_B=\beta^2\operatorname{Var}(H).
$$

For estimating β with fixed known J, the full-state score is negative centered energy; the endpoint score is its conditional expectation. Marginalization loses Fisher information in this finite model. This compares one exact independent preparation in each observation model and is not an estimator-efficiency guarantee.

$$
\mathcal I_{\rm full}(\beta)=\operatorname{Var}(H),\qquad\mathcal I_{\rm ends}(\beta)=\operatorname{Var}_y(\bar H_y),\qquad\mathcal I_{\rm full}-\mathcal I_{\rm ends}=\mathbb E_y[\operatorname{Var}(H\mid y)]>0.
$$

### Read the comparison

The upper panel compares the induced dimensionless coupling and endpoint correlation. The lower panel divides energy variance into visible conditional-mean variation and hidden conditional variance. Their sum, not either curve alone, gives the original equilibrium response.

![Exact hidden-spin marginalization creates endpoint dependence while the original energy variance retains a positive hidden contribution.](/assets/learning/partition-research-hidden-spin-marginal.en.svg)

### Reproduce the example

Physical energies and variances have joule and squared-joule units. The table divides variances by the squared coupling J to make them dimensionless. The b=0 row is the infinite-temperature limit; Fᵧ itself is not evaluated there. All numerical hyperbolic functions are approximations.

| b | Induced k | Endpoint correlation | Between / J squared | Hidden / J squared | Total / J squared |
| --- | --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 2.00000000 | 2.00000000 |
| 0.25000000 | 0.06005725 | 0.05998515 | 0.21278386 | 1.66724584 | 1.88002970 |
| 0.50000000 | 0.21689042 | 0.21355227 | 0.55357384 | 1.01932163 | 1.57289547 |
| 1.00000000 | 0.66250137 | 0.58002566 | 0.61668845 | 0.22326023 | 0.83994868 |

```python
from math import cosh,tanh,log
for b in [0,0.25,0.5,1]:
    k = log(cosh(2*b))/2
    correlation = tanh(b)**2
    same = cosh(2*b)/(1+cosh(2*b))
    between = 4*tanh(2*b)**2*same*(1-same)
    hidden = 4*same/cosh(2*b)**2
    total = 2/cosh(b)**2
    print(f"{b:.8f} {k:.8f} {correlation:.8f} "
          f"{between:.8f} {hidden:.8f} {total:.8f}")
```

### Three research questions with explanations

#### 1. May the common log-weight a be dropped?

**Explanation.** It cancels in normalized endpoint probabilities, but dropping it changes the physical partition function. Its β derivatives contain energy information from the hidden spin. Preserving a normalized histogram is weaker than preserving thermodynamic derivatives.

#### 2. Is the effective potential a fixed microscopic Hamiltonian?

**Explanation.** No. Summing hidden states makes it temperature dependent. The conditional mean energy is ∂β(βFᵧ), and energy variance additionally includes conditional fluctuations. Applying a fixed-level fluctuation identity directly to Fᵧ omits those terms.

#### 3. Does mediated endpoint dependence prove a direct endpoint bond?

**Explanation.** No. The specified microscopic chain has only two adjacent bonds. The endpoint statistical coupling appears after marginalizing their common neighbor. This is a finite coarse-observation example, not a claim about a microscopic force or a universal critical exponent.

### Source and research boundary

The Ising Hamiltonian background is [Tong, section 5.2](https://davidtong.org/pdfs/teaching/statistical-physics/statphys5.pdf). The open three-spin sum, conditional variance split and Fisher comparison are derived here by enumerating eight states. No renormalization-group limit or physical spin-flip clock is inferred.

## Practice with explanations

### 1. Count before normalizing

At J=h=0, compare full support with fixed zero sum. What are Z and the marginal probability of an up spin?

**Answer.** Full support has four unit weights, so Z=4 and each marginal up probability is 1/2. Constrained support has two unit weights, so Z=2 and the same marginals. The joint tables differ even though the marginal probabilities agree.

### 2. Explain zero mean with nonzero covariance

Use βJ=ln2,h=0. Compute agreement probability and the conditional probability that the second spin is up given that the first is up.

**Answer.** Agreement probability is 2/5+2/5=4/5. The conditional probability is (2/5)/(1/2)=4/5. Each spin has zero mean, yet their covariance is 3/5. Zero mean concerns each individual distribution; dependence concerns their joint arrangement.

### 3. Distinguish two measures

Is ln(5/4) the mutual information of the hand example?

**Answer.** No. It is Δ, the signed log-normalizer comparison with uncoupled spins. Mutual information is the probability-weighted logarithm of p divided by its marginal product, giving about 0.19274476 nats rather than 0.22314355. Both require stating their comparison object.

### 4. Check physical response units

For the hand example, find the fixed-temperature magnetization response to h.

**Answer.** The variance of M is 16/5. Multiply it by β, so the response is 16β/5 in inverse joules. The dimensionless source curvature 16/5 alone omits the chain-rule factor. A negative temperature would reverse this physical response sign.

### 5. Test a proposed zero-temperature rule

A colleague proposes that zero cross covariance always means zero response at T=0. Give a counterexample.

**Answer.** At J=h=0 the four ground states are uniform and Cov(A,M)=0. Any nonzero h selects an agreeing ground state, so mean alignment jumps from zero to one. Its derivative at h=0 does not exist. Ground-set selection under both signs of a perturbation must be examined directly.

### 6. Diagnose parameter identifiability

With support restricted to M=0, can configuration samples identify J or h at fixed temperature?

**Answer.** Neither is identified from the normalized configuration law. A=−1 makes the J contribution common to both states; M=0 removes the h contribution. The probabilities remain 1/2 and 1/2, and the source Hessian is zero. This does not imply that the spins are independent: their covariance is −1 and their mutual information is ln2.

## Connect the four-state calculation to the experiment

Begin with **Correlated unbiased pair**. The two-by-two diagram shows a joint table: the first spin selects a row and the second a column. Read all four probabilities before looking at the marginal means. The equal marginal frequencies coexist with a preference for agreement; a mean of zero is not an independence test.

| Experiment | What to compare | What to explain |
| --- | --- | --- |
| Uncoupled full support → Zero-sum constraint without coupling | Keep J=h=0 and change the allowed state space | Both marginals stay uniform, but covariance changes from0 to−1 and mutual information from0 to ln2 |
| Correlated unbiased pair → Preference for disagreement | Reverse J at the same positive T and h=0 | The sign of spin covariance changes; dependence need not disappear |
| Zero covariance, absent derivative → Three-state crossing at zero temperature | Inspect source Hessian and physical-response rows separately | Finite source covariance at a ground crossing does not supply a finite physical derivative |

The field input h already includes the magnetic-moment factor needed to make it an energy in joules. A derivative with respect to this h therefore has inverse-joule units; it is not automatically a derivative with respect to a laboratory field in tesla. At finite nonzero T, keep T, support and the other energy parameter fixed when perturbing J or h to test a response.

A forbidden cell is marked separately from a rare allowed cell. Switching support actually removes configurations, whereas increasing an energy only suppresses their probability at finite temperature. Compare the signed normalizer difference Δ with mutual information I in their own rows: the zero-sum example gives Δ=−ln2 and I=ln2. A color shade is a visual cue, not a replacement for the numerical table or a measure of transition speed.
