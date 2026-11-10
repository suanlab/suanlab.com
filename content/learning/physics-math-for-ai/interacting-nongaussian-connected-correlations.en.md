## 1. What a covariance matrix cannot tell us

Imagine four binary detectors that each report either +1 or −1. Every detector has mean zero, and every pair looks uncorrelated. Does that prove the four reports are independent? No: their product can still be biased toward +1. A covariance matrix describes second-order structure, not every possible dependence.

We will build two strictly positive distributions with the same means and covariance matrix but different fourth connected correlations. We also study ordinary neighboring spin interactions, where changing a source changes the covariance itself. Both observations break assumptions that were valid for the previous quadratic Gaussian family.

These are finite classical equilibrium distributions. A site labels a position or a component, not an instant along a trajectory. Statistical dependence is not a causal explanation, and this calculation does not estimate a causal effect from observational data.

## 2. Declare the state space and interactions

Use n=4,…,8 spins, with sᵢ∈{−1,+1}. An open chain contains edges (0,1),…,(n−2,n−1). A periodic ring adds (n−1,0). Write S(s) for the sum of neighboring spin products and R(s) for the product at the first four sites only:

$$
S(s)=\sum_{(i,j)\in E}s_i s_j,\qquad
R(s)=s_0s_1s_2s_3,\qquad
U(s)=-J S(s)-gR(s)-\sum_i h_i s_i.
$$

The four-body term is one specified interaction, not a sum over all groups of four or all translated windows. At n>4 it still acts only on sites 0,1,2,3. This explicit term gives us a clean higher-order counterexample without changing the state space.

Positive J favors aligned neighbors and negative J favors opposite neighbors. Positive g favors an even number of down spins among the first four. Neither statement forces one configuration: at finite coefficients every configuration remains possible.

## 3. Input factors are not physical energies

Set Boltzmann's constant to one. Instead of asking a decimal exponential routine to evaluate the weights, supply positive rational factors:

$$
q=e^{J/T},\qquad r=e^{g/T},\qquad p_i=e^{h_i/T},\qquad
w(s)=q^{S(s)}r^{R(s)}\prod_i p_i^{s_i}.
$$

Thus q=2 means J=T log 2; it does not mean J=2. A source factor pᵢ=1 means hᵢ=0, not hᵢ=1. Factors below one represent negative physical coefficients. Never insert a signed factor into this probability model.

The calculator accepts q,r,pᵢ∈[1/16,16] and 0<T≤16. Numerators and denominators contain at most twelve digits; zero, decimal notation, exponents, leading zeros and invalid dimensions are rejected. Equivalent fraction spelling is retained in the input while arithmetic uses reduced fractions.

## 4. Normalize using counting measure

Every one of the 2ⁿ configurations is counted once:

$$
Z=\sum_{s\in\{-1,+1\}^n}w(s),\qquad
P(s)=\frac{w(s)}{Z},\qquad
\langle A\rangle=\sum_s P(s)A(s).
$$

There is no integration-cell width, continuous measure, rejected configuration or Monte Carlo burn-in. Positive finite factors make Z positive and finite. The calculation reports every state probability and a magnetization histogram, so normalization can be checked directly.

A finite sum of positive exponentials is smooth in the physical coefficients. A sharp-looking finite histogram is not a thermodynamic singularity. The importance of taking an infinite-system limit before interpreting spontaneous symmetry breaking is explained in [Tong, Statistical Field Theory §1.2.1](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S1).

## 5. Why exact integer weights are possible

For one reduced factor a/b appearing with exponent σ=±1, multiply its contribution by ab. The result is a² for σ=+1 and b² for σ=−1. Apply this to each site factor, each edge factor and the single four-body factor:

$$
D=\prod_f a_f b_f,\qquad W(s)=D w(s)\in\mathbb N,
\qquad P(s)=\frac{W(s)}{\sum_{s'}W(s')}.
$$

This common D need not be the smallest denominator. It cancels from probabilities, while Z=(∑W)/D remains exact. Large numerators are a computational cost, not a sign of physical divergence.

Encode a down spin with a set bit. Products of spins are parity characters of this bit string. The implementation uses a Walsh transform of the integer weights to obtain all subset moments. Applying that transform twice must reproduce 2ⁿW. The independent reference instead multiplies rational factors for every state and directly sums spin products; it does not reuse the transform.

## 6. Repeated insertions still matter

Define μᵢ=⟨sᵢ⟩ and Cᵢⱼ=⟨sᵢsⱼ⟩−μᵢμⱼ. Because a spin satisfies sᵢ²=1, repeated powers simplify raw moments:

$$
\langle s_i^2\rangle=1,\qquad
\langle s_i^3\rangle=\mu_i,\qquad
C_{ii}=1-\mu_i^2.
$$

A centered power is different: (sᵢ−μᵢ)² is not identically one. Simplifying the raw product before centering is safe; replacing every centered repeated insertion with one is not.

The four probe entries are labeled insertion slots and may repeat. Their multiplicities must remain when forming partitions or Wick pairings. Four identical labels still have three pairings.

## 7. Sources generate moments and connected correlations

Introduce a dimensionless auxiliary source t, separate from the physical energy source h, and define M(t)=⟨exp(tᵀs)⟩ at the current input. Then M(0)=1. Derivatives of M generate raw moments; derivatives of log M generate cumulants.

$$
\partial_{t_i}\log M(0)=\mu_i,\qquad
\partial_{t_i}\partial_{t_j}\log M(0)=C_{ij}.
$$

For independent groups M factorizes and log M adds. Mixed derivatives spanning independent groups therefore vanish. This statement does not say that a single non-Gaussian variable has no higher cumulants. The definitions and labeled-insertion partition construction are developed in [Helias and Dahmen, §§II.3–II.4](https://arxiv.org/html/1901.10416v1).

For the physical source, differentiation changes pᵢ=exp(hᵢ/T), holding T,J,g and other physical sources fixed. Each derivative inserts 1/T. This local derivative is defined mathematically even though the input interface samples rational factors.

## 8. Third cumulants describe a first nonlinear response

The third cumulant is the centered third moment:

$$
\kappa_{ijk}=\langle s_i s_j s_k\rangle
-\mu_i\langle s_j s_k\rangle-\mu_j\langle s_i s_k\rangle
-\mu_k\langle s_i s_j\rangle+2\mu_i\mu_j\mu_k.
$$

Its sign is not determined by the signs of pair covariances. At a globally spin-flip symmetric input, every odd raw moment vanishes, and so does κᵢⱼₖ. A nonzero local source can destroy that symmetry.

Susceptibility is χᵢⱼ=Cᵢⱼ/T. Unlike a fixed quadratic Gaussian family, this C can depend on h. Its source derivative gives ∂χᵢⱼ/∂hₖ=κᵢⱼₖ/T². Thus a third cumulant measures the first change in linear response, rather than another name for covariance.

## 9. Fourth cumulants remove three covariance pairings

Let ηᵢ=sᵢ−μᵢ. The fourth cumulant is

$$
\kappa_{ijkl}=\langle\eta_i\eta_j\eta_k\eta_l\rangle
-C_{ij}C_{kl}-C_{ik}C_{jl}-C_{il}C_{jk}.
$$

This identity applies to the present finite distribution without assuming Gaussianity. Wick's Gaussian rule would assert that the centered fourth moment equals the three pairings, making κ₄ zero. Here that equality is a hypothesis to test, not a rule to impose.

The direct raw-moment partition formula has one four-block insertion term, four triple-plus-single terms, three pair-plus-pair terms, six pair-plus-two-single terms with coefficient +2, and the four-single term with coefficient −6. This accounts for all fifteen partitions of four labeled slots. It remains valid for repeated sites.

## 10. Independence does not mean Gaussianity

Set q=r=pᵢ=1. The spins are independent and unbiased. At four distinct sites, the fourth mixed cumulant is zero because the four variables are independent. At one site repeated four times, however,

$$
\langle s_0^4\rangle=1,\qquad C_{00}=1,\qquad
\kappa_{0000}=1-3=-2.
$$

An independent Rademacher spin is not Gaussian. A Gaussian variable with variance one instead has fourth moment three. Distinct-site independence and same-site distribution shape answer different questions.

This also shows why a nonzero fourth cumulant alone does not prove interaction between distinct sites. Always report the probe indices and the reference family being rejected.

## 11. Same mean and covariance, different fourth structure

Now set q=pᵢ=1 and vary only r. Since R=±1, write rᴿ=(r+r⁻¹)/2+R(r−r⁻¹)/2. Summing over spins gives

$$
Z=2^{n-1}(r+r^{-1}),\qquad \mu=0,\qquad C=I,
\qquad \langle R\rangle=\kappa_{0123}=\frac{r^2-1}{r^2+1}.
$$

Every proper nonempty subset of the first four spins remains uniform: a flip of an unobserved spin cancels its parity contribution. Hence all pair covariances are unchanged even though four-way dependence changes.

At n=4,r=2, the eight R=+1 states each have probability 1/10; the eight R=−1 states each have probability 1/40. Their total probabilities are 4/5 and 1/5. The fourth cumulant is 3/5, while the r=1 independent reference has the same μ,C and fourth cumulant zero. Neither distribution has zero-probability states.

## 12. A biased spin exposes raw-subtraction errors

Take q=r=1, p₀=2 and all other pᵢ=1. The first spin has probabilities 4/5 and 1/5, so μ₀=3/5. Repeating probe 0 yields:

| Quantity | Exact value |
|---|---:|
| Variance | 16/25 |
| Centered third moment / third cumulant | −96/125 |
| Raw fourth moment | 1 |
| Centered fourth moment | 832/625 |
| Three covariance pairings | 768/625 |
| Correct fourth cumulant | 64/625 |
| Incorrect raw fourth minus three raw pair products | −2 |

The wrong expression is not even of the same sign. It discards necessary mean and third-moment contributions. Changing the source changes this spin's covariance and higher response, even though it stays independent of the other spins.

## 13. Neighbor interactions give a separate non-Gaussian example

For the default four-site ring, q=2,r=pᵢ=T=1. Count two fully aligned states with weight16, twelve states with two disagreeing edges and weight1, and two alternating states with weight1/16. Therefore Z=353/8.

The first covariance row is (1,255/353,225/353,255/353). At the four distinct probes, the raw and centered fourth moments are both 225/353. The three Wick products have numerators 65025,50625,65025 over 124609. Consequently κ₀₁₂₃=−101250/124609.

The four-body factor is one in this example. Pairwise energy interactions can already produce nonzero higher connected correlations. Conversely, a vanishing particular fourth cumulant would not prove that an entire distribution is Gaussian.

## 14. Finite-volume symmetry is a check, not a phase diagnosis

When every pᵢ=1, global spin reversal s↦−s leaves S and R unchanged. Every state and its reversed state have equal weight. All odd subset moments vanish exactly, regardless of how strongly the even interactions favor alignment.

With arbitrary fields, replacing every pᵢ by its reciprocal reverses the distribution. Means and odd cumulants change sign; covariance and fourth cumulants are preserved. This is a useful exact check that does not require a numerical derivative.

A symmetric distribution may have a bimodal total-magnetization histogram. Mean zero then coexists with large fluctuations. It does not mean the spins are independent, nor does it demonstrate spontaneous symmetry breaking in this finite system.

## 15. Boundaries and frustration change the distribution

At q=2,r=pᵢ=T=1, the open four-site chain has Z=125/4 and first covariance row (1,3/5,9/25,27/125). Closing it into a ring changes the graph, partition function and correlations; it is not a change of drawing style.

At q<1, an even ring can alternate every neighboring spin. An odd ring cannot: the product of all edge products is always +1, while making every one of an odd number of edges negative would give −1. At least one edge must frustrate the preferred pattern.

This combinatorial obstruction is exact. A plot of the five-site q=1/2 ring illustrates it, but the plot is not the proof. We do not infer an infinite-volume correlation length by fitting these few points.

## 16. Interaction parameters have their own response statistics

Write θ=log q and γ=log r. Differentiating these dimensionless coefficients inserts S and R, respectively:

$$
\partial_\theta\log Z=\langle S\rangle,\quad
\partial_\theta^2\log Z=\operatorname{Var}(S),\quad
\partial_\gamma\log Z=\langle R\rangle,\quad
\partial_\theta\partial_\gamma\log Z=\operatorname{Cov}(S,R).
$$

The implementation reports these sufficient-statistic moments, including Var(R)=1−⟨R⟩². Physical derivatives with respect to J or g require the corresponding factors of 1/T. A derivative with respect to q itself is different again: ∂q log Z=⟨S⟩/q.

This distinction is useful in exponential-family learning. A parameter gradient is tied to the observable conjugate to that parameter. Replacing a four-body statistic with a pair statistic silently changes the fitted model.

## 17. Temperature changes require a declared path

Keeping q,r,p fixed while doubling T preserves the whole distribution: J,g,h all double because they equal T times the logarithms of those factors. Covariance is unchanged, but susceptibility C/T is halved. This is a rescaling of physical energies, not heating the same Hamiltonian.

For a true fixed-energy comparison from T=1 to T=2, use q=4→2, r=9→3 and p₀=4→2, keeping other source factors one. The factors become square roots, so the physical J=log4, g=log9 and h₀=log4 stay fixed. These specially chosen roots remain rational and are represented exactly.

At fixed T the physical-source derivative hierarchy is

$$
\partial_{h_i}\log Z=\mu_i/T,\quad
\partial_{h_i}\partial_{h_j}\log Z=C_{ij}/T^2,\quad
\partial_h^3\log Z=\kappa_3/T^3,\quad
\partial_h^4\log Z=\kappa_4/T^4.
$$

The selected second and third derivatives of a mean are κ₃/T² and κ₄/T³, respectively. The probe slots specify which component is measured and which sources are perturbed.

## 18. What the exact computation establishes

All probabilities, moments, cumulants and response values are reduced rational numbers. Logarithms defining physical energies remain analytic expressions; no claim is made that log2 is rational. At most256 configurations are enumerated, so the method is deliberately finite and exponential in n.

The producer forms common-denominator integers and a Walsh transform. A separate Python Fraction reference directly sums each rational weight, multiplies spin products, and constructs cumulants from labeled set partitions. It also computes centered products directly. Agreement is checked across named examples, seeded inputs and extreme factors.

Normalization, positivity, spin diagonals, spin reversal, repeated indices and transform inversion are additional checks. Passing a finite reference collection is not a formal proof for every admissible input. Browser Workers and reader integration require separate transport, lifecycle and accessibility checks.

## 19. How to use the idea in AI research

A pairwise feature representation cannot distinguish the independent reference from the pure four-body example using population means and covariances alone. Adding the parity feature s₀s₁s₂s₃ separates them. This is an exact representational counterexample, not a claim that every neural network needs a particular architecture.

For an energy-based model, compare sufficient statistics and higher held-out moments in addition to matching training pair correlations. In finite data, estimating a fourth cumulant also introduces sampling uncertainty and sensitivity to preprocessing. This calculator supplies population values, so those empirical uncertainties require a separate experiment.

The [spatial-spectrum chapter](/book/online/physics-math-for-ai/spatial-spectra-decay-and-correlation-length/) asks a different question: how covariance varies with separation and wave number. Higher-order dependence and spatial decay should be studied together without treating either one as a substitute for the other.

## 20. Worked review questions

1. **Does zero pair covariance prove independence?** No. At q=pᵢ=1,r=2, μ=0 and C=I, yet κ₀₁₂₃=3/5. A fourth-order observation separates the distribution from independent spins.
2. **Does a nonzero fourth cumulant prove interaction?** No. An unbiased independent spin repeated four times has κ₀₀₀₀=−2. Probe multiplicities and same-site distribution shape matter.
3. **What is the partition function of the four-site pure four-body example at r=2?** Eight states have weight2 and eight have weight1/2, giving Z=20. Their individual probabilities are1/10 and1/40.
4. **Why is raw4−three raw2 products wrong at nonzero mean?** It omits mean and third-order terms. For the biased independent spin it gives−2, while proper centering and covariance subtraction give64/625.
5. **What happens when T doubles at fixed q,r,p?** The distribution and covariance remain unchanged; physical energies double and susceptibility halves. This is not fixed-energy heating.
6. **Why are odd moments zero at zero physical fields?** Global spin reversal pairs equal-weight states with opposite odd observables. Even pair and four-body interactions preserve this symmetry.
7. **Can an odd antiferromagnetic ring satisfy every edge?** No. Multiplying all edge products must give+1, contradicting an odd number of preferred negative products.
8. **Which cumulant determines a third source derivative of a mean?** The fourth cumulant divided by T³, with the first probe slot labeling the mean and the other three labeling source perturbations.

## 21. Reading and next steps

The finite Ising starting point and the distinction between finite and infinite systems are developed in [Tong, §1](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S1). For the source convention, compare [Tong, §2.2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2). For moments, cumulants and the special Gaussian Wick rule, consult [Helias and Dahmen, §§II–III](https://arxiv.org/html/1901.10416v1).

The rational-factor parameterization, bounded implementation, pure four-body teaching counterexample and exact reference checks are derived here for this course. They are not presented as a quoted construction from those sources. General interacting continuum fields, renormalization, critical exponents and quantum time-ordered correlators remain separate material.
