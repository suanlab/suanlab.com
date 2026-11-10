## Why the most likely energy need not be the lowest energy

Imagine three floors in a building. The higher floors cost more energy to reach, but the middle floor has two rooms while each other floor has one. A law that assigns a probability to each room and a law that assigns a probability to each floor describe different outcomes. The middle floor can win by offering more rooms, even though each of its rooms is less likely than the ground-floor room.

The same distinction appears in statistical mechanics, Bayesian inference and energy-based machine learning. An exponential penalty says how energy changes a weight. It does not say how many outcomes are available or what reference weight each outcome had before applying that penalty. This chapter makes those choices explicit, then asks which of them can be recovered from observed probabilities.

You should already know how to normalize a finite sum and how a partition function generates mean energy. The canonical starting point follows [Tong's treatment of the canonical ensemble](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S1.html#S1.S3). The weighted examples and identifiability arguments below are derived for the preparations stated here. They do not assert that every AI probability model is a physical equilibrium system.

## Specify the labels, support, weights and units

There are finitely many labels i. Each has an energy E_i in joules and a nonnegative, dimensionless reference weight m_i. At least one weight is positive. Temperature T is in kelvin; kB has units J/K. For positive finite temperature,

$$
\beta=\frac{1}{k_BT},\qquad
Z_m=\sum_{i:m_i>0}m_i e^{-\beta E_i},\qquad
p_i=\frac{m_i e^{-\beta E_i}}{Z_m}.
$$

A zero reference weight excludes a label. Its probability is exactly zero at every temperature under this fixed support. By contrast, every allowed finite-energy label has strictly positive probability at every finite nonzero temperature. A tiny positive value that the screen cannot display has not become a forbidden event.

There are three common interpretations of m. They share the same normalization algebra, but their entropy interpretations differ.

| Reference | What m means | What a label represents |
| --- | --- | --- |
| Counting measure, m=1 | One counted outcome | One explicit microstate |
| Integer degeneracy, m=g | g distinct outcomes with the same energy | An energy level grouping microstates |
| Positive real reference weight | A specified base measure, possibly a prior | A model label; no automatic microstate count |

The reference need not sum to one. When it does, call it a normalized prior only if that interpretation is intended. A floating-point input such as 0.1 is also a particular binary approximation; the numerical checks use that actual input rather than silently replacing it with the rational 1/10.

## Derive the weighted odds before normalizing

Divide the probabilities of two allowed labels. The common partition function cancels:

$$
\frac{p_i}{p_j}=\frac{m_i}{m_j}\exp[-\beta(E_i-E_j)],\qquad
\ln\frac{p_i}{p_j}=\ln\frac{m_i}{m_j}-\beta(E_i-E_j).
$$

The log odds consist of a reference advantage and an energy penalty. At positive temperature, a higher-energy label i is more likely than j precisely when its logarithmic reference advantage exceeds its dimensionless energy gap. “Lower energy is more likely” therefore requires equal reference weights, or a comparison between individual equally counted microstates.

For a visual explanation, picture one column per energy level. Its vertical position gives energy, its width gives reference weight, and its filled area after applying the exponential penalty gives unnormalized mass. Normalization rescales the filled areas to sum to one. A column with zero width is absent; a very thin filled area is still present. When examining a plot, check whether its bars show mass per label or probability per microscopic room.

## Work through three levels and four microstates

Set the energies to 0, ε and 2ε, choose integer degeneracies 1, 2 and 1, and impose the ideal relation βε=ln 2. Then

$$
(m_i e^{-\beta E_i})=(1,1,1/4),\qquad Z_m=9/4,\qquad
(p_0,p_1,p_2)=(4/9,4/9,1/9).
$$

| Level | Degeneracy | Probability per microstate | Probability of the level |
| --- | --- | --- | --- |
| 0 | 1 | 4/9 | 4/9 |
| ε | 2 | 2/9 | 4/9 |
| 2ε | 1 | 1/9 | 1/9 |

To check the grouping, explicitly list four microstates with energies 0, ε, ε and 2ε. Their probabilities are 4/9, 2/9, 2/9 and 1/9. Adding the two middle probabilities recovers the level probability 4/9. If you list both middle microstates and give each a degeneracy of two, you count those rooms twice and produce a different model.

Writing K=E/ε gives a useful moment check:

$$
\langle K\rangle=\frac23,\qquad
\langle K^2\rangle=\frac89,\qquad
\operatorname{Var}(K)=\frac49.
$$

Consequently U=2ε/3 and Var(E)=4ε²/9. The ideal heat capacity is C/kB=4(ln 2)²/9. This hand calculation assumes an exact logarithm relation; a rounded computer preparation has its own independent reference. An ideal fraction is a teaching check, not an exact oracle for rounded inputs.

## Separate label entropy from entropy relative to the reference

Label entropy asks how uncertain the level label is. Relative entropy here uses the chosen base weights; this signed convention is not the nonnegative KL divergence:

$$
H(p)=-\sum_{i:p_i>0}p_i\ln p_i,\qquad
S_m/k_B=-\sum_{i:p_i>0}p_i\ln\frac{p_i}{m_i}
=H(p)+\sum_i p_i\ln m_i.
$$

For integer degeneracies, the canonical probabilities within one level are equal. The conditional entropy of the microstate inside level i is ln g_i. The ordinary entropy chain rule then gives the full microstate entropy as label entropy plus the mean conditional entropy. That is why S_m is the thermodynamic counting entropy for this particular interpretation.

In the worked example, the missing conditional information is (4/9)ln 2. The label entropy is ln 9−(8/9)ln 4, while the microstate entropy is ln 9−(4/3)ln 2. Resolving the two middle rooms increases the entropy. A diagram that groups rooms can hide that information without changing the energy distribution.

If instead m is a normalized prior q, the same formula becomes

$$
S_q/k_B=-D_{\mathrm{KL}}(p\Vert q)\leq0.
$$

A negative value is expected and does not mean label entropy is negative. It measures deviation from that prior using the explicitly signed convention. For an unnormalized reference m, define M=Σm and q=m/M. Then S_m/kB=ln M−KL(p||q). There is no general nonnegativity requirement for arbitrary real base weights.

## Change reference scale without changing the probability law

Replace each reference weight by λm_i for fixed λ>0. The unnormalized masses and Z both acquire the same factor. Hence

$$
p_i'=p_i,\qquad Z_m'=\lambda Z_m,\qquad
S_m'/k_B=S_m/k_B+\ln\lambda,\qquad
F_m'=F_m-k_BT\ln\lambda,
$$

where F_m=−kBT ln Z_m. Label entropy, mean energy, variance and heat capacity do not change. The reference-dependent entropy and free energy do change. A probability vector by itself therefore does not determine the normalization of the reference measure.

For example, divide the degeneracies (1,2,1) by four to obtain the prior (1/4,1/2,1/4). The probabilities remain (4,4,1)/9, but Z changes from 9/4 to 9/16. The counting entropy is about 1.27303 kB, label entropy is about 0.96496 nats, and prior-relative entropy is about −0.11327 kB. Reporting these three values with one generic “entropy” label would conceal different questions.

An informative experiment keeps the energy bars and probability bars stationary while rescaling the reference weights. Only the partition-function, reference-entropy and free-energy readouts should move. The reference convention must remain visible beside those readouts.

## Keep the Hamiltonian and reference fixed during derivatives

Take the logarithm of the probability formula and average it. Differentiating the finite partition sum at fixed energies and fixed reference weights gives

$$
S_m/k_B=\ln Z_m+\beta U,\qquad
-\frac{\partial\ln Z_m}{\partial\beta}=U,\qquad
\frac{\partial^2\ln Z_m}{\partial\beta^2}=\operatorname{Var}(E),\qquad
\frac{C}{k_B}=\frac{\operatorname{Var}(E)}{(k_BT)^2}.
$$

The last identity follows by the chain rule dβ/dT=−1/(kBT²). It remains a statement about the specified equilibrium family. It does not measure how quickly a physical system or sampler reaches equilibrium.

You can absorb positive m into an effective energy E_eff,i=E_i−kBT ln m_i. This reproduces the probabilities at a chosen temperature. But that effective energy depends on temperature even when E and m are fixed. Differentiating it as though it were a fixed Hamiltonian loses the derivative of the reference term. Likewise, if a learning algorithm changes the prior as temperature changes, the derivative includes that change and is no longer the fixed-reference heat capacity above.

## Distinguish two different energy ambiguities

First, adding a common constant c to every energy leaves every energy difference unchanged. At fixed reference and temperature,

$$
p_i'=p_i,\quad Z_m'=e^{-\beta c}Z_m,\quad U'=U+c,\quad F_m'=F_m+c.
$$

Variance, label entropy, reference entropy and heat capacity are unchanged. This is an energy-origin ambiguity: normalized probabilities cannot determine the common zero of energy. It differs from reference rescaling, which changes reference entropy while preserving the same probability law.

Second, multiply all energies and temperature by the same positive factor a, holding kB and m fixed. Every βE stays the same, so the probabilities stay the same. Mean energy scales by a and energy variance by a², while C/kB is unchanged. Observing p does not jointly identify the absolute energy scale and temperature. Additional calibration, a known temperature or a specified energy unit is necessary.

With known m and nonzero β, the odds equation can recover energy differences. With unknown m, the observed log odds can be explained by different allocations between reference weight and energy. An inverse problem must state what is fixed before claiming that its parameters have been learned uniquely.

## Connect to softmax with an explicit scale

For dimensionless logits s_i and positive dimensionless softmax temperature α, a weighted softmax is

$$
p_i=\frac{m_i e^{s_i/\alpha}}{\sum_jm_j e^{s_j/\alpha}}.
$$

To connect this to a physical energy, choose an energy scale E_* in joules and set E_i=−E_*s_i and kBT=αE_*. Without E_*, α is not a temperature in kelvin. An equal additive shift of all logits cancels, and scaling logits and α together also preserves the law. These familiar softmax symmetries are the normalized-law versions of the energy ambiguities above.

Under the uniform reference, high positive α flattens a finite logit distribution. Under a nonuniform reference, the α→∞ limit is m/Σm, generally not uniform. At small positive α, mass concentrates on the largest logits; among exactly tied maxima, probabilities remain proportional to m. A visualization should show that residual reference preference rather than drawing equally high bars for all tied labels.

## Treat zero temperature and negative temperature separately

At T→0 from above, let G be the allowed labels of minimum energy and M_G=Σ_Gm. Then

$$
p_i\longrightarrow
\begin{cases}m_i/M_G,&i\in G,\\0,&i\notin G,\end{cases}
\qquad S_m/k_B\longrightarrow\ln M_G.
$$

For two tied ground labels with weights 1 and 3, their limiting probabilities are 1/4 and 3/4. Label entropy is about 0.56234 nats, whereas reference entropy is ln 4. For a finite gapped spectrum, variance and heat capacity tend to zero. These are finite-spectrum limits; they do not establish a universal classical or quantum third-law statement.

A finite bounded list also normalizes for negative finite T. Its Boltzmann law favors higher energies relative to the reference, so the stable numerical anchor is the largest allowed energy. The variance formula still gives nonnegative heat capacity. This mathematical preparation does not imply that a system with an unbounded energy spectrum admits a negative-temperature equilibrium.

At the positive-side zero-temperature limit, the behavior of Z itself depends on the ground energy: it tends to zero for positive ground energy, diverges for negative ground energy, and tends to M_G for zero ground energy. The normalized ground-state law is well defined in all three cases. Infinite log Z in this limit must be distinguished from a finite nonzero partition function that merely exceeds the display range.

## Read numerical observations without erasing rare events

The calculation accepts 1–32 finite levels, a separate common offset and nonnegative finite weights. A zero weight is removed before selecting the low- or high-energy anchor. Exact binary sums preserve a relative gap even when adding a large offset would make displayed energy values identical.

Directed rational bounds enclose the exponentials, logarithms, normalized probabilities and moments. Precision grows from 128 to at most 8192 bits; a nonzero observation is resolved only when its interval has a certified sign and relative width at most 2^-48. If that cannot be established, the result stays unresolved. An interval containing zero is not a proof of zero. Nonzero values outside the binary64 display range have a null linear value with a retained sign; their logarithmic information is reported separately when available.

Variance uses the pair identity Σ_{i<j}p_i p_j(E_i−E_j)². This avoids subtracting two nearly equal large raw moments and preserves a physically representable variance even when a contributing probability is too small to display. A probability that rounds to one can still have a negative nonzero log probability, revealing its tiny complement.

Independent Decimal references use the actual binary64 inputs and the stored numerical kB. They are recomputed at 400–1600 decimal digits and confirmed with 80 additional digits. These tests include forbidden labels, real reference weights, large reference rescaling, rare-event moments and both temperature boundaries. They verify this finite numerical model; these numerical checks alone do not prove that a renderer, sampling animation or physical experiment implements it correctly.

## Exercises with explained answers

1. **Which middle-room probability is correct?** In the ideal three-level example, a friend assigns probability 4/9 to each of the two middle microstates. Explain the error. **Answer:** 4/9 belongs to the grouped level. Each equally counted room has probability 2/9, so their sum is 4/9. Giving each 4/9 changes normalization and double-counts multiplicity.

2. **Can the higher level win?** Compare a lower label with m=1 to a label one gap ε higher with m=3, at βε=ln 2. **Answer:** Their probability ratio is 3 exp(−ln 2)=3/2. The higher label is more likely because the reference advantage exceeds the energy penalty. Normalizing the two-label example gives probabilities 2/5 and 3/5.

3. **Why is the entropy negative?** Replace the degeneracies by the normalized prior (1/4,1/2,1/4) in the worked example. **Answer:** p is unchanged, but S_q/kB equals −KL(p||q), approximately −0.11327. Label entropy remains approximately 0.96496 nats. The sign reflects the reference-relative definition, not a failure of Shannon entropy.

4. **What can one observation identify?** Suppose m and T are known and two positive probabilities are measured. Can you recover both absolute energies? **Answer:** The log odds recover E_i−E_j=−kBT ln[(p_i m_j)/(p_j m_i)]. A common energy shift remains arbitrary. If T is also unknown, the dimensionless gap is determined, but the absolute gap and temperature are not separately identified.

5. **Are tied ground labels uniform?** Use energies (0,0,ε), weights (1,3,1/4) and the positive-side zero-temperature limit. **Answer:** The limiting law is (1/4,3/4,0), because only the allowed ground weights survive. Its label entropy is −(1/4)ln(1/4)−(3/4)ln(3/4); its reference entropy is ln 4. Uniformity requires equal ground reference weights.

6. **Has an underflow removed a state?** A finite-temperature allowed excited label has a null linear probability and positive sign, while a forbidden label has probability zero. May both be deleted before computing variance? **Answer:** Only the forbidden label is exactly absent. The allowed probability may combine with a huge squared energy gap to yield a representable variance. Multiply using retained exact bounds before projecting; if precision is insufficient, report uncertainty rather than zero.

## Continue from equilibrium laws to update rules

The distribution specifies normalized masses, energy fluctuations and reference dependence. It does not specify a trajectory. Next, finite Gibbs conditionals will build update kernels that preserve a target law, while showing why stationary, reversible and mixing describe different properties. The continuous Gaussian case will then replace the finite reference weights by a stated Lebesgue measure and make normalizability an explicit question.

## Connect reference weights to the experiment

Start with **Integer microstate multiplicities**. Predict the grouped law (4,4,1)/9, then compare the level bars with the small room diagram. A room label i:j denotes one microstate within level i. The level bar is the sum over those rooms, rather than the probability of each room.

| Experiment | Keep fixed or compare | What to explain |
| --- | --- | --- |
| Normalized prior, same probabilities | Replace (1,2,1) by (1/4,1/2,1/4) at the same energies and T | p and H(label) stay the same while ln Z, reference entropy and free energy change |
| Large common offset, same gaps → Energy and temperature scaled by eight | Compare an origin change with a joint scale change | The first shifts U; the second scales U and Var(E), despite preserving p |
| Very low forbidden level → Weighted ground-state zero-K limit | Read reference support before comparing energy minima | A zero-weight label is absent, and tied allowed ground labels need not be uniform |

The two list inputs must have the same length. Edit both and press **Apply energy and reference lists**; an incomplete list or all-zero reference is rejected. This differs from the preceding spectrum's add-state button. The room diagram is displayed only for integer counts totaling at most32. An absent room diagram does not mean that the probability law or its entropy is absent; arbitrary real weights need their stated reference interpretation.

Read H(label), E[ln m] and Sₘ/kB=H+E[ln m] together. A negative prior-relative value is expected under its signed definition. Keep T, the energy list and the common offset fixed when testing reference rescaling. Keep the reference weights fixed when testing a temperature derivative. These operations answer different questions even when the probability bars look identical.

Commit each scalar field with Enter or by moving focus away from it.
