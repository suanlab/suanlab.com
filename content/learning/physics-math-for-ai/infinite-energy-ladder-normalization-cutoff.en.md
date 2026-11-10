## A normalized short list can still miss most of a distribution

Suppose a building has infinitely many floors and each higher floor costs the same extra energy. At low temperature, visits concentrate near the bottom. At high temperature, many floors matter. A computer can list only finitely many floors, but making that short list sum to one does not establish that the omitted floors were unimportant.

This chapter separates three questions: does an infinite equilibrium law exist, what law does a finite cutoff define, and how accurately does that finite law represent the infinite one? We use a spectrum specified in advance. The oscillator example in [Tong's discussion of internal modes](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S2.html#S2.S4) motivates the setting; the cutoff calculations and numerical contracts below follow from our stated ladder. We are not deriving quantization, transition rates or a thermalization mechanism.

## State the spectrum and normalization conditions

There is one labeled state for every nonnegative integer n. The level spacing ε is nonnegative and measured in joules. For the infinite equilibrium calculation, assume ε>0 and finite T>0:

$$
E_n=\varepsilon(n+1/2),\qquad n=0,1,\ldots,\qquad
x=\frac{\varepsilon}{k_BT},\qquad q=e^{-x},\qquad 0<q<1.
$$

The half-spacing is a common energy offset. It shifts mean energy and free energy, but not the normalized probabilities or energy variance. One state per index is part of the counting convention. Adding a degeneracy that grows with n would change the convergence problem and the resulting distribution.

For ε=0, every state has the same weight and the infinite sum diverges. For ε>0,T<0, the weights grow with n and again cannot be normalized. A finite truncation still normalizes in both cases because it has only finitely many finite weights. Existence of that finite law is not evidence for an infinite law.

## Sum the geometric series before assigning probabilities

The common half-spacing factor can be taken outside the sum. Multiplying a geometric sum by q shifts every term and leaves only the first term after subtraction:

$$
Z_\infty=e^{-x/2}\sum_{n=0}^\infty q^n
=\frac{e^{-x/2}}{1-q}=\frac1{2\sinh(x/2)},\qquad
p_\infty(n)=(1-q)q^n.
$$

The index has a geometric distribution on zero and the positive integers. Some texts instead count trials starting at one; that convention changes the mean index by one. Here n=0 is the ground level. The probability distribution can also be viewed through the repeated-trial construction in [Taboga's geometric distribution notes](https://www.statlect.com/probability-distributions/geometric-distribution). This is a mathematical identification of a law, not a claim that the energy system literally performs independent coin flips.

The series converges because q is strictly below one as a mathematical number. A rounded screen value of q=1 is a numerical representation issue; it must not overwrite the exact input condition x>0.

## Derive occupation moments and physical observables

Differentiate the geometric sum with respect to q. Multiplying the first derivative by q generates n, and applying that operation twice generates n². Normalizing and centering yields

$$
\langle n\rangle_\infty=\frac{q}{1-q},\qquad
\operatorname{Var}_\infty(n)=\frac{q}{(1-q)^2},\qquad
U_\infty=\varepsilon\left(\frac12+\frac{q}{1-q}\right),\qquad
\operatorname{Var}_\infty(E)=\varepsilon^2\frac{q}{(1-q)^2}.
$$

The mean index is dimensionless, the mean energy is in joules, and energy variance is in squared joules. A huge index can coexist with a modest physical energy when the level spacing is tiny. Multiplying a displayed infinite index by a displayed zero spacing would lose that distinction; the exact input factors must be combined before projecting the result.

The fixed-spectrum canonical identities give heat capacity, entropy and free energy:

$$
\frac{C_\infty}{k_B}=x^2\frac{q}{(1-q)^2},\qquad
\frac{S_\infty}{k_B}=-\ln(1-q)+x\frac{q}{1-q},\qquad
F_\infty=\frac{\varepsilon}{2}+k_BT\ln(1-q).
$$

Keep ε fixed when differentiating with respect to temperature. If ε itself depends on T, the derivative of U includes that dependence and is no longer given by the displayed fluctuation formula alone. The half-spacing contributes no heat capacity under the fixed-spectrum assumption.

## A finite cutoff defines a conditional distribution

Allow only n=0 through L and write m=L+1 for the number of retained states. A finite geometric sum gives

$$
Z_L=e^{-x/2}\frac{1-q^m}{1-q},\qquad
p_L(n)=\frac{(1-q)q^n}{1-q^m}\quad(0\le n\le L),\qquad
p_L(n)=0\quad(n>L).
$$

For the normalizable infinite preparation, this is exactly its distribution conditioned on n≤L. A conditioning operation increases every retained probability by the same factor. It does not mean that the physical system was unable to occupy higher levels. That interpretation requires a real support constraint in the preparation.

The finite normalizer is positive even if q=1 or q>1 mathematically; then evaluate the finite sum directly rather than using an indeterminate geometric quotient. At ε=0 it is simply m. For negative temperature, anchoring at the highest retained energy makes the deficit L−n have decreasing weights, although the physical ratio q between successive n weights exceeds one.

## Measure what the cutoff removed

The tail probability is another geometric series. It is also the total variation distance between the infinite law and the finite law extended by zero outside the cutoff:

$$
\delta_L=P_\infty(n>L)=q^m,\qquad
\frac{Z_L}{Z_\infty}=1-\delta_L,\qquad
\lVert p_L-p_\infty\rVert_{\mathrm{TV}}=\delta_L.
$$

For retained states, the finite law gains total probability δ_L by renormalization. Outside, it loses that same amount. Half the sum of the absolute differences is therefore δ_L. This derivation uses the special conditional structure; a general finite approximation need not have this formula.

The direction of a KL comparison matters:

$$
D_{\mathrm{KL}}(p_L\Vert p_\infty)=-\ln(1-\delta_L),\qquad
D_{\mathrm{KL}}(p_\infty\Vert p_L)=+\infty\quad(T>0,\ L<\infty).
$$

The forward ratio is constant on retained states. The reverse comparison encounters positive infinite-law probability where the finite law has exact zero. Even a tiny tail causes this support mismatch. This mathematical infinity is different from a finite numerical quantity outside the display range.

## Quantify mean and variance bias separately from missing mass

Differentiate the finite geometric normalizer, or subtract the conditional moments from their infinite counterparts. The resulting positive biases are

$$
\langle n\rangle_L=\frac{q}{1-q}-\frac{m q^m}{1-q^m},\qquad
\operatorname{Var}_L(n)=\frac{q}{(1-q)^2}-\frac{m^2q^m}{(1-q^m)^2}.
$$

The energy bias is ε times the mean-index bias. Tiny missing mass does not automatically mean a negligible error for an unbounded observable: rare high indices can carry a large contribution. In this ladder, the exact formulas quantify it instead of relying on mass alone. For a bounded observable, total variation supplies a bound involving its range; the index and energy on the infinite ladder are not bounded.

These subtraction formulas are good for derivation but can be poor numerical recipes. Near high temperature, both terms in a variance difference can be enormous while the finite result is modest. The implementation evaluates positive finite sums for the first two raw moments and then forms a directed enclosure for the variance, rather than subtracting two displayed infinite quantities.

## Work through q equal to one half

Take x=ln2 and L=2. The three retained relative weights are 1,1/2,1/4. Their normalized probabilities are (4,2,1)/7, but the infinite preparation also includes every higher n.

| Quantity | Infinite ladder | Cutoff L=2 |
| --- | --- | --- |
| Mean index | 1 | 4/7 |
| Index variance | 2 | 26/49 |
| Mean energy divided by ε | 3/2 | 15/14 |
| Total retained probability in the infinite law | 7/8 | Conditioned to 1 |

$$
\delta_2=1/8,\qquad
\langle n\rangle_\infty-\langle n\rangle_2=3/7,\qquad
\operatorname{Var}_\infty(n)-\operatorname{Var}_2(n)=72/49,\qquad
D_{\mathrm{KL}}(p_2\Vert p_\infty)=\ln(8/7).
$$

This connects to the earlier three-level example, now with the common half-spacing included. The two normalized lists both sum to one, yet their means and variances differ. The ideal relation x=ln2 defines the hand calculation; independent numerical references use the actual binary64 ε,T and stored kB instead of treating computed exponentials as exact fractions.

## Set a cutoff using the intended error criterion

For a requested missing-mass tolerance η between zero and one, the requirement is

$$
\delta_L=e^{-mx}\le\eta
\quad\Longleftrightarrow\quad
m x\ge\ln(1/\eta),\qquad m=L+1.
$$

At x=ln2 and η=1/100, m must be at least seven, so L≥6. If the goal is mean energy accuracy, use the energy-bias formula as well; a mass target alone answers a different question. Numerical rounding near an integer boundary should be checked against the inequality rather than trusting a rounded ceiling formula.

The calculator's supported cutoff range is 0 through 4096. That is an implementation resource budget, not a theorem that higher levels do not matter. If the requested tolerance requires more states than the interface supports, report that fact. Do not relabel the largest supported cutoff as adequate.

## High temperature and cutoff removal need not commute

With fixed positive ε, high temperature means x→0. On the infinite ladder, typical n grows like 1/x and the heat capacity tends to kB. On any fixed finite list, the distribution tends to uniform, the mean energy stays bounded, and the heat capacity tends to zero:

$$
\lim_{T\to\infty}\frac{C_\infty}{k_B}=1,\qquad
\lim_{T\to\infty}\frac{C_L}{k_B}=0\quad(L\text{ fixed}),\qquad
\lim_{T\to\infty}\langle n\rangle_L=\frac L2.
$$

Taking L→∞ after the fixed-cutoff high-temperature limit gives a heat-capacity limit of zero. Removing the cutoff first and then raising temperature gives one in units of kB. To retain most probability while increasing temperature, the product (L+1)x must grow sufficiently large; merely choosing a fixed large L is not a uniform guarantee.

This is removal of a level-space cutoff for a single specified spectrum. It is not the thermodynamic limit of particle number and volume. The example teaches why the order of approximations and limits must be stated, without identifying physically different limits.

## Distinguish zero temperature from a very cold finite preparation

For ε>0, the positive-side T=0 limit puts all probability at n=0. Its energy and free energy equal ε/2, and its entropy and heat capacity vanish:

$$
U_0=F_0=\varepsilon/2,\qquad S_0=C_0=0,\qquad
p_0(n)=\mathbf1_{\{n=0\}},\qquad
Z\longrightarrow0,\quad\ln Z\longrightarrow-\infty.
$$

The vanishing raw normalizer does not mean probabilities vanished; normalization is taken before the limit. At every finite positive temperature, the tail is still strictly positive and the reverse KL against a finite cutoff is infinite. At the zero-temperature limit, both laws become the same point mass and both KL directions are zero. That support change must be explicit.

At ε=0 there are infinitely many equal-energy states. There is no uniform probability law on that infinite countable set, including under the stated zero-temperature convention. Selecting one state arbitrarily would add a new preparation assumption, not repair the divergent partition sum.

## Preserve numerical meaning near zero and infinity

For tiny x, direct floating-point subtraction 1−exp(−x) can return zero. Directed intervals and exact binary input ratios preserve the positive denominator. For example, a subnormal gap at the largest finite temperature can produce a mean index outside binary64 range while U remains finite and C/kB is close to one.

For x>4096, the model uses separately labeled leading logarithmic moments. Their physical omission is bounded because q<2^-4096. For a nontrivial finite ladder and the infinite law, the leading mean and variance are q, while the leading entropy in units of kB is (1+x)q. Omitting the one in that entropy would give an error of order 1/x, which is much larger than the stated tail correction.

$$
\langle n\rangle_\infty\sim q,\qquad
\operatorname{Var}_\infty(n)\sim q,\qquad
C_\infty/k_B\sim x^2q,\qquad S_\infty/k_B\sim(1+x)q\quad(x\to\infty).
$$

| Situation | Mathematical meaning | Numerical treatment |
| --- | --- | --- |
| Zero gap in the infinite ladder | Partition sum diverges | No invented normalized probabilities |
| Finite T, tiny positive tail | Positive probability | Keep sign and logarithmic magnitude |
| T=0 positive-side limit | True zero above the ground state | Exact support change |
| Precision ceiling reached | Accuracy was not established | Preserve an unresolved result |

The cold omission bound does not certify rounding errors of floating-point logarithms and exponentials. Ordinary interval-resolved observations and logarithmic asymptotics have distinct labels. Large ladders also return selected rows for plotting, using their original probabilities; those rows are not renormalized to form a new distribution.

## Connect truncation to energy-based learning

A model over a very large output set may replace the true normalizer by a candidate subset. In this ladder, we can calculate exactly what such a replacement does: it conditions the distribution, changes feature expectations and introduces a support mismatch. A normalized candidate list is therefore not by itself a certificate of likelihood or gradient accuracy for the original model.

For the source −x multiplying n, log-normalizer derivatives generate the mean index and variance. Replacing the infinite normalizer with its truncated version replaces those moments with conditional moments. The bias is controlled by the formulas above, not by whether an optimizer converged. For more general models, a geometric-tail formula may be unavailable, so a separate tail estimate or other justified approximation is required.

The ladder supplies a transparent benchmark: specify the support, compare a known infinite target with finite approximations, choose the observable and error criterion, and inspect how the required cutoff changes with temperature. These are static distribution checks. They do not determine how long a sampler takes to reach equilibrium.

## Practice with explanations

### 1. Check existence before calculation

Does a normalized infinite law exist for ε=0,T>0, or for ε>0,T<0?

**Answer.** Neither case normalizes. Equal unit weights give an infinite sum in the first case; increasing weights fail even to approach zero in the second. Finite cutoffs normalize in both cases, but do not create infinite equilibria.

### 2. Recover the omitted mass

For q=1/2,L=2, why is the missing probability 1/8 although the displayed finite probabilities sum to one?

**Answer.** The infinite tail starts at n=3 and sums to q³=1/8. Conditioning rescales the retained mass 7/8 to one. That rescaling explains the finite normalization without eliminating the original tail.

### 3. Compare two error criteria

At q=1/2,L=2, calculate the mean-index and variance biases.

**Answer.** The mean bias is 1−4/7=3/7. The variance bias is 2−26/49=72/49. Neither equals the missing mass 1/8, because moment errors weight the positions and spread of the omitted levels.

### 4. Choose a mass cutoff

At x=ln2, what cutoff guarantees missing mass at most one percent?

**Answer.** Require 2^−m≤0.01. The smallest integer m is seven, giving L=6 and missing mass 1/128. This is a mass guarantee; energy accuracy requires checking its own bias.

### 5. Explain the two KL directions

Why can the forward KL be small while the reverse KL is infinite?

**Answer.** On the retained support, p_L/p_infinity is the finite constant 1/(1−δ). Outside it, p_infinity is positive but p_L is exactly zero, making the reverse logarithmic ratio infinite. At the stated zero-temperature limit that positive tail disappears and the support comparison changes.

### 6. Diagnose a numerical conclusion

A program reports q=1 and therefore declares the positive-gap, positive-temperature infinite ladder nonnormalizable. What should be checked?

**Answer.** Check the exact supplied ε/(kBT), then compute the positive difference 1−q with a stable method or directed precision. A rounded q=1 can coexist with positive x and a normalizable law. Evaluate physical energy and heat capacity before projecting enormous occupation numbers, and report numerical accuracy limits separately from mathematical divergence.

## Connect the tail calculation to the experiment

Select **q=1/2, three-state cutoff**. The input L is the last retained index, so L=2 keeps three states. Compare the finite and infinite bars at n=0,1,2, and inspect the first outside row n=3. The finite probability there is an actual zero while the infinite probability is positive. Sum only the first three infinite probabilities to recover retained mass7/8; the finite law renormalizes that mass to one.

| Experiment | What to compare | What to explain |
| --- | --- | --- |
| Missing mass below one percent | Change L from2 to6 at the same ε,T | Tail mass becomes1/128; forward KL is finite and reverse KL remains infinite |
| High temperature, small cutoff → Same temperature, larger cutoff | Keep x≈0.001 and compare L=2 with4096 | A larger supported cutoff improves accuracy without necessarily meeting a chosen tolerance |
| Zero gap, no infinite normalization → Negative temperature, finite law only | Inspect the existence notice before reading the finite table | A normalized finite list does not create a missing infinite target |

At x≈0.001 even L=4096 leaves approximately1.66% missing mass. A one-percent target would require L≥4605, outside the current supported range. Report that limit rather than treating the largest selectable cutoff as sufficient. For a mean-energy target, check the energy-bias row too; missing mass alone is a different error criterion.

The level spacing ε and temperature T determine q; q is a calculated readout, not a separate probability input. At positive finite T, an out-of-range tiny tail remains positive. At the explicit T=0 ground limit, both laws become the same point mass and both KL directions vanish. Selected plotting rows retain their original probabilities: their sum need not be one and they must not be renormalized as a third law. Comparison playback changes preparations and cutoffs, not physical time.
