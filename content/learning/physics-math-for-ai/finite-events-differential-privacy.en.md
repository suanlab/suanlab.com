## Privacy is a property of an output distribution

Imagine a small fixed-size survey containing one binary answer per person. You want an approximate count without making the released output depend too strongly on any one answer. Removing names is not a mathematical bound on that dependence. We need to specify the protected change, the released mechanism and the probabilities being compared.

This lab uses n between1 and6, known publicly. A database is a fixed vector of n bits. Neighbors differ by replacing one person's bit. This protects a bit value conditional on the stated fixed-size setting; it does not hide a roster or the public n. The add/remove convention used in many presentations of differential privacy is a different adjacency relation. Replacing a0 by a1 changes two entries of a histogram, even though it changes only one row of this vector.

No prior distribution over databases is assumed. Independence below concerns the mechanism's random coins, not the data values of different people.

## Read every quantifier in the definition

For epsilon>=0 and delta>=0, the mechanism M is (epsilon,delta)-differentially private under the declared adjacency if

$$
\forall D\sim D',\quad\forall S\subseteq\mathrm{Range}(M),\qquad
\Pr[M(D)\in S]\le e^\epsilon\Pr[M(D')\in S]+\delta.
$$

Probability is over the randomness of M on a fixed input. S can contain several outputs. The relation is checked in both directions: swapping D and D' is another ordered comparison. Identical-input comparisons are automatic for these nonnegative parameters.

Epsilon controls the multiplicative factor alpha=exp(epsilon). Delta is additive slack in the worst event inequality. It is not literally a percentage of people whose privacy is lost. At delta1 the inequality is trivial for every mechanism; such a passing claim alone conveys no useful protection.

For standard background, see [Dwork and Roth, The Algorithmic Foundations of Differential Privacy, Definition2.4 and §3.2](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf). That book's default histogram add/remove convention should be distinguished from this lab's fixed-size replace-one-bit convention. The count-channel calculations and finite certificates below use the latter explicitly.

## The randomized-response count channel

Each individual reports the true bit with probability p and flips it with probability1−p. The mechanism releases ONLY the total Y of reported ones, not the response vector, identities or random seeds. Conditional on a true count c,

$$
Y\ \overset{d}{=}\ \mathrm{Binomial}(c,p)
+\mathrm{Binomial}(n-c,1-p),
$$

where the two counts are independent. The actual bit arrangement does not matter: permuting records permutes the independent factors while leaving the reported total's law unchanged. Every neighboring database pair therefore has distributions represented by counts c and c+1, in one of the two directions. All2n nontrivial ordered count classes cover every actual neighboring pair in this model.

The code computes every output probability, not a Monte Carlo approximation. With pPercent as an integer, multiply the generating polynomials

$$
\bigl((1-p)+pz\bigr)^c\bigl(p+(1-p)z\bigr)^{n-c}.
$$

The coefficient of z^y is P_c(Y=y). All coefficients have a common denominator100^n. For n<=6 their integer weights are safe integers. Claim comparisons use BigInt cross-products when an additional denominator would exceed safe multiplication range.

## One event is a query, not the whole guarantee

The first diagram compares the two chosen distributions. The blue bars are P_D(Y=y), orange bars are P_{D'}(Y=y). Outlines select the event S. An event mask encodes membership as a sum of2^y. Mask3 means{0,1}; mask16 means{4}. Bits for impossible output indices above n are ignored and both requested/effective masks are shown.

The entered lower count clamps to n−1; the direction control chooses c→c+1 or its reverse. These controls only inspect the mechanism. Changing the claimed alpha/delta, event or neighbor query does not change its distributions.

For the chosen S, compare P_D(S) with alpha P_{D'}(S)+delta. A pass is evidence about exactly that pair and event. The definition still quantifies over every pair and every event.

## Solve the all-events maximization exactly

For a fixed ordered pair, write P_y=P_D(Y=y) and Q_y=P_{D'}(Y=y). For any S,

$$
P(S)-\alpha Q(S)=\sum_{y\in S}(P_y-\alpha Q_y).
$$

Including a negative term cannot improve the sum, while excluding a positive term loses something. Therefore

$$
\max_S[P(S)-\alpha Q(S)]
=\sum_y[P_y-\alpha Q_y]_+,
\qquad[t]_+=\max(0,t).
$$

The maximizing event contains every strictly positive atom. If the maximum is0, the empty event is a valid maximizer and does not represent a privacy violation. Maximize this value again over all ordered neighboring count pairs to obtain delta_min(alpha). The claimed guarantee holds exactly when delta>=delta_min.

This identity covers all subsets without guessing an attack. Tests additionally enumerate all finite subsets and compare their maxima with the optimizer. The certificate shows the maximizing pair and event, not merely the two distributions currently selected by the user.

At alpha1, the excess is neighboring total variation:

$$
\sum_y[P_y-Q_y]_+=\frac12\sum_y|P_y-Q_y|.
$$

For delta0, atomwise multiplicative bounds can be summed over any event. For positive delta, checking each singleton with its own delta is insufficient: the slack is allowed once for an entire event, not once per atom.

## A complete pure-privacy example

Load n4, p3/4, alpha3 and delta0. A changed individual's report has probability ratio at most3. In the full response vector, all unchanged independent factors cancel between neighbors. Summing the corresponding vector probabilities over a count-output event preserves that bound.

The bound is tight: the all-ones reported count changes its probability by p/(1−p) when one input bit flips in the appropriate direction. Thus for p<1,

$$
\epsilon_{\rm pure}=\ln\frac{p}{1-p}.
$$

At p3/4 this is ln3, and delta_min(3)=0. The global panel certifies every ordered count class and every event, including the queried{0,1} for c1→2.

The privacy cost is not n times ln3 simply because n people randomize. Only one person's record changes in the comparison, so unchanged factors cancel. Releasing several fresh responses about the SAME person is a different composition problem, addressed next.

## A passing event can miss a concrete violation

The second example uses n4, p0.9, alpha3 and claimed delta0.05. For c2→3 and event{4},

$$
P(S)=0.0081,\qquad Q(S)=0.0729,\qquad
\alpha Q(S)+\delta=0.2687.
$$

The chosen event passes easily. But for the SAME pair the event{0,1,2} has excess0.1042, already above0.05. Across all pairs the maximum is larger:

$$
D:c0,\quad D':c1,\quad S=\{0\},\qquad
0.6561-3(0.0729)=0.4374=\frac{2187}{5000}.
$$

The exact global claim fails. A claim delta0.437 is still below the threshold, while0.438 is above it. These decisions use rational inequalities rather than the rounded labels.

This channel's tight pure epsilon is ln9. It does not follow that every smaller epsilon is impossible with positive delta: the additive term allows a different point on the privacy profile. It must be computed and stated, not assumed from one favorable event.

## What delta is measuring here

When Q_y>0 define the pairwise privacy loss L(y)=ln(P_y/Q_y). The maximizing event above contains the outputs with L(y)>epsilon. However,

$$
\delta_{\rm pair}(\epsilon)
=\sum_{y:L(y)>\epsilon}P_y\bigl(1-e^{\epsilon-L(y)}\bigr),
$$

rather than simply the probability of that set. This explains why delta should not be casually equated with a probability that an individual is exposed. If Q_y=0<P_y, that atom contributes P_y. If both probabilities are0, its likelihood ratio is undefined but its event-excess contribution is0. Nothing is repaired by inserting an arbitrary zero or a tiny smoothing probability.

## Privacy and estimation error are different calculations

The expected reported total is

$$
\mathbb E[Y\mid c]=n(1-p)+(2p-1)c.
$$

For p>1/2, an unbiased count estimate is

$$
\widehat c=\frac{Y-n(1-p)}{2p-1},\qquad
\mathbb E[(\widehat c-c)^2\mid c]
=\frac{np(1-p)}{(2p-1)^2}.
$$

Each randomized bit has variance p(1−p), regardless of its true value, and the randomizations are independent. This gives the stated mean-square error after scaling. The formula is conditional on any fixed database; no random-data prior is required.

At n4,p3/4 the possible estimates are−2,0,2,4,6 and MSE is3. Estimates outside[0,n] are not clipped. Clipping would change bias and risk and would need its own analysis. At p0.9 the MSE is0.5625, an accuracy improvement with a weaker pure privacy parameter. This is a property of the specified estimator/channel, not a universal privacy–utility curve.

## Two boundary cases make the distinction clear

At p1/2 every record's report is a fair coin. All true counts have the same Binomial(n,1/2) output law. Minimum epsilon is0 and minimum delta is0. The inversion formula divides by2p−1=0 and is unavailable: Y contains no count information. An unavailable estimator is not infinite privacy loss.

At p1 the released count equals the true count. Estimation error is0. Neighboring counts have disjoint supports; for the event containing one count, P=1 and Q=0. No finite pure epsilon works, and every finite alpha requires delta>=1. The panel explicitly displays infinite pure epsilon rather than hiding it as a zero or an invalid number.

The mechanism's visible output and randomness assumptions matter. Revealing the raw record, a random seed or extra metadata would be a different release and would require its own analysis. The current finite certificate does not silently cover that program.

## Read the privacy profile

The lower graph has alpha=exp(epsilon) on the horizontal axis, NOT epsilon itself. Blue values are exact global delta requirements at all allowed alpha grid points1.0,1.1,…,10.0; connecting segments guide the eye. Orange is the claimed delta. The selected blue point must lie at or below the orange point to pass.

Changing n or p changes the channel. Changing alpha changes the privacy claim and profile query; changing delta changes the horizontal budget line. Neighbor/event controls affect inspection only. Exact numerator comparisons decide the badges; logarithms and plotted decimals are display quantities.

## Exercises and worked solutions

1. At n1,p3/4,alpha1, what delta is needed? The two distributions are(3/4,1/4) and(1/4,3/4), so the maximum excess is1/2. The channel is(0,1/2)-DP under this adjacency, while the pure guarantee is(ln3,0).
2. Can every singleton pass but their union fail when delta>0? Take P=(.4,.4,.1,.1), Q=(.1,.1,.4,.4), alpha1,delta.3. Every singleton inequality passes in both directions. Event{0,1} has excess.6, which fails. Delta cannot be spent separately on each atom.
3. In the n4,p.9 example, does increasing claimed delta from.437 to.438 change the mechanism? No. Rows, pure parameter and estimator risk stay the same. Only whether the claim covers the exact global requirement.4374 changes.
4. Why does a full-output event pass even for a deterministic count? Both event probabilities are1, so1<=alpha+delta. A single always-true event does not cover the definition; the singleton count event exposes the deterministic channel.

## Scope of the result

This is a mathematical certificate for an explicitly bounded finite channel, its replace-one adjacency and declared count output. It is not a compliance judgment or a guarantee for an arbitrary deployed data system. General mechanisms need a proof or audit covering their own full domain, randomness, release format and composition.

The next unit studies epsilon/delta composition, where repeating one sanitized output and making fresh independent releases must be distinguished.
