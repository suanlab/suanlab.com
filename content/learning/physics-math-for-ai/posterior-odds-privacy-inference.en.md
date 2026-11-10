## Privacy does not mean that every prediction becomes inaccurate

Suppose an observer already assigns probability.99 to a hypothesis. Predicting that hypothesis gives99% expected accuracy before any new release. An output completely independent of the secret cannot remove that prior knowledge; it adds no evidence, even though the observer remains accurate.

[Dwork and Roth, §2.2–2.3](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) explain why forbidding all learning is not the purpose of useful private analysis. This lesson makes a narrower statement precise: for two specified hypotheses, what evidence does a released output add, and when does a pure DP likelihood-ratio bound constrain that evidence?

We will distinguish four objects throughout: a channel's conditional output laws, the observer's prior, the posterior after one observation and an average measure of inference. None can be substituted for all the others.

## Specify the hypotheses before applying a bound

Let H be0 or1 and let pi=Pr(H1). In mode0, D0 contains n zeros and D1 differs only in its first record, which is1. These are replace-one neighboring databases. Each individual independently reports their true bit with probability p and its flip otherwise. The release is only the total number Y of reported ones.

Mode1 compares D0=all0 with D1=all1. For n>1 these databases are not one-record neighbors: they differ in n positions. They can represent a perfectly correlated population hypothesis. The mechanism still has its same per-record guarantee, but the comparison of these two endpoints needs a group bound.

Mode2 is a different single-bit mechanism. It releases an uninformative symbol bottom with probability1-q and otherwise releases the protected bit. Its output codes are0 for bottom,1 for bit0 and2 for bit1. The n and p controls are explicitly unused in this mode; q is unused in the other modes.

The prior is an observer model over these two fixed hypotheses. Changing it does not change the channel law at either hypothesis. It changes how the observer weights those laws.

## Bayes conditioning requires a possible observation

Write P0(y)=Pr(Y=y|H0) and P1(y)=Pr(Y=y|H1). The mixture law of the observed output is

$$
m(y)=(1-\pi)P_0(y)+\pi P_1(y).
$$

If m(y)>0, Bayes' rule gives

$$
\pi_y=\Pr(H_1\mid Y=y)=\frac{\pi P_1(y)}{(1-\pi)P_0(y)+\pi P_1(y)}.
$$

If m(y)=0, the posterior is undefined in this experiment. Do not silently replace it by0, by1/2 or by the prior. A crossed output in the posterior diagram marks this case.

Endpoint priors deserve particular care. If pi0 and an output occurs only under H1, that observation has probability0 under the stated prior, even though the alternative channel assigns it positive likelihood. It contradicts this complete observer model rather than yielding an ordinary conditional update. Real modeling may then require revising the assumptions; the lab does not invent a posterior for the zero-denominator formula.

## Odds separate prior belief from new evidence

For an interior prior and positive likelihoods,

$$
\frac{\pi_y}{1-\pi_y}
=\frac{\pi}{1-\pi}\frac{P_1(y)}{P_0(y)}.
$$

The last factor is the Bayes factor in favor of H1. A factor1 gives no update; a factor3 multiplies odds by3, rather than adding3 to a probability.

When P0=0<P1, the Bayes factor is infinite. With a compatible interior prior the posterior is1. When P1=0<P0 it is0 and the posterior is0. If both likelihoods are0, their ratio is undefined0/0. These cases differ from an impossible observation caused by an endpoint prior, so the readout reports the likelihood and prior-dependent possibility separately.

Prior odds are infinite at pi1. Expressions such as0 times infinity cannot be used as an update rule there. The direct mixture formula is the starting point; the odds identity is a convenient equivalent only where its operations are defined.

## Derive posterior bounds from a pure pairwise guarantee

Suppose the actual two hypothesis laws have a finite pure bound with alpha=exp(epsilon):

$$
\alpha^{-1}\le\frac{P_1(y)}{P_0(y)}\le\alpha
$$

on outputs with positive likelihood. Two-sided pure DP gives shared support, so there is no positive-versus-zero atom in such a pair. Multiply prior odds and convert back to probability:

$$
\frac{\pi}{\pi+(1-\pi)\alpha}
\le\pi_y\le
\frac{\pi\alpha}{1-\pi+\pi\alpha}.
$$

These bounds concern the stated two hypotheses. They do not assert that every conceivable sensitive attribute has an unchanged posterior under an arbitrary population prior. The database relation in the privacy theorem must match the comparison being made.

For the neighboring randomized-response count laws at p<1, the tight alpha is p/(1-p). Summing the other independent reports cannot increase this ratio, and the extreme count outputs attain it. At p1/2 the laws are identical, alpha1 and posterior equals prior. At p1 the two relevant counts are deterministic and distinct, so no finite pure bound exists.

## Work the balanced-prior guide by hand

The first guide uses n1,p3/4,pi1/2 and selects output1. The likelihoods are P0(1)=1/4 and P1(1)=3/4, giving mixture probability1/2 and Bayes factor3. Prior odds1 become posterior odds3, hence posterior probability3/4.

Pure epsilon is ln3, measured using a natural logarithm. The posterior interval for this balanced prior is[1/4,3/4]; the selected output attains its upper endpoint. Output0 instead gives posterior1/4 and attains the lower endpoint.

The optimal decision before release makes an error with probability1/2. After release, choosing the more likely hypothesis for each output gives error1/4. These are averages over the mixture, rather than statements that every particular observed prediction is correct.

## A strong prior need not indicate new leakage

The second guide uses n4,p1/2 and pi.99. Both conditional output laws are Binomial(4,1/2). Their Bayes factor is1 for every possible output, so every posterior remains.99.

The observer's optimal accuracy is.99 both before and after release. The channel provides zero additional information about the hypothesis. Prior knowledge has not disappeared, but no evidence was added.

This example also prevents the opposite mistake. A low observed attack accuracy does not by itself prove DP: it might reflect a weak attack, an unsuitable prior or a particular dataset. The DP definition quantifies over neighboring inputs and output events, independently of this observer model.

## A group hypothesis has a different comparison distance

For n6,p3/4, compare all0 with all1 and observe six reported ones. Their likelihoods are(1/4)^6 and(3/4)^6, so the Bayes factor is3^6=729. With balanced prior the posterior is729/730, approximately.998630.

The per-record pure epsilon remains ln3. But the two hypotheses differ in six rows. Insert intermediate databases changing one row at a time and multiply each pure probability bound: the endpoint ratio is at most exp(6 ln3)=729. The group posterior bound therefore permits this strong evidence.

Calling that observation a violation of a one-record bound would apply the theorem to the wrong input relation. Conversely, a small per-record epsilon does not promise to conceal every population pattern. The example is an explicit correlated-hypothesis comparison, not a claim that DP stops being valid in the presence of auxiliary information.

## Rare disclosure separates approximate guarantees from pointwise certainty

In mode2 the conditional laws on codes(bottom,bit0,bit1) are

$$
P_0=(1-q,q,0),\qquad P_1=(1-q,0,q).
$$

At epsilon0, the positive event difference is q in each direction. Therefore the channel is exactly(0,q)-DP. For q>0, a revealing symbol has zero probability under the other hypothesis, so there is no finite pure epsilon.

The third guide sets q=.01 and balanced prior. Output code2 has mixture probability.005, posterior1 and infinite Bayes factor. The output is rare, but its evidence is conclusive when it occurs under this model. The pure posterior interval is therefore marked unavailable.

The total probability of a revealing symbol is q in this particular constructed channel. It is not a general interpretation of delta in arbitrary approximate-DP mechanisms. Approximate DP constrains weighted event excess; it does not supply the same finite per-output odds bound as pure DP.

## Average posterior is preserved while information can increase

By substitution into Bayes' rule,

$$
\sum_{y:m(y)>0}m(y)\pi_y=\sum_y\pi P_1(y)=\pi.
$$

Thus the mixture-weighted mean posterior equals the prior. This identity does not mean that the observation is uninformative: some outputs move the belief up and others down. An unweighted average across output labels is generally wrong.

For zero-one decision loss, the optimal post-release error is

$$
R_Y=\sum_y\min\{(1-\pi)P_0(y),\pi P_1(y)\},\qquad
R_{\rm prior}=\min\{1-\pi,\pi\}.
$$

The observer can always ignore the output, so the optimized average error cannot increase. Tied decisions do not need an arbitrary forced hypothesis to compute this risk.

Let h(u)=-u log2(u)-(1-u)log2(1-u), with endpoint values0. The average information is

$$
I(H;Y)=h(\pi)-\sum_y m(y)h(\pi_y).
$$

This quantity uses bits, whereas epsilon uses natural-log likelihood ratios. It depends on the prior. It is not itself a replacement for a prior-free DP guarantee.

## Compute the information in rare and ordinary releases

For the first guide, the two equally weighted posteriors are1/4 and3/4. The information is1-h(3/4), approximately.1887218755 bits. For the independent second guide it is0.

For rare disclosure, bottom leaves the prior unchanged, while either revealing symbol has posterior entropy0. Therefore information is q h(pi). With balanced prior and q.01 it is.01 bits, even though the selected revealing symbol gives certainty. Bayes error becomes(1-q)min(pi,1-pi), so the guide has error.495 instead of.5.

If a prior is already certain, its entropy and mutual information are0 even for a mechanism that would reveal the bit to other observers. Low information under one selected prior cannot certify a mechanism universally private.

## Read the plots and exact fractions

The first plot shows the two channel laws and the selected output. The second shows each possible output's posterior against the orange prior line. It marks impossible outputs with a cross rather than a zero bar. The table includes their mixture weights, which are essential for average error and information.

The final plot varies the prior while keeping the selected output's likelihoods fixed. Blue shows Bayes updates only where the observation is possible; orange is unchanged belief. Open endpoint markers denote undefined endpoint updates. If both selected likelihoods are0, the blue curve is absent for every prior, rather than coinciding with the zero line.

All probability weights and prior-weight products stay within safe integer bounds in this finite model. The selected posterior has a reduced exact fraction in the caption. Logarithms, entropy and displayed decimals are approximate. Posterior odds are computed directly from joint integer weights, avoiding loss of accuracy from subtracting a rounded posterior from1.

## Exercises and worked solutions

1. With prior.2 and Bayes factor3, find the posterior. Prior odds are1/4; posterior odds are3/4 and posterior probability is3/7, not.6.
2. For balanced prior and a pure alpha3 bound, what interval is allowed? It is[1/4,3/4]. A posterior.99 would contradict that bound for the actual neighboring pair.
3. Why can the group example have posterior729/730? Its two databases differ in six records and the valid group alpha is729, not3.
4. At q.01 and balanced prior, does observing bottom give posterior.99? No; both bottom likelihoods are.99, so the Bayes factor is1 and the posterior stays.5.
5. At q0, what posterior follows a revealing symbol? None in this model: its mixture probability is0. The correct display is undefined.
6. Does expected posterior equal to prior imply zero information? No. The first guide has expected posterior.5 but positive information and improved Bayes error.
7. Why does approximate DP not bound every posterior by the pure interval? The rare channel satisfies(0,q)-DP while a possible revealing symbol has infinite likelihood ratio.

## Research connection and limitations

Membership, attribute and population inference ask different hypothesis questions. An empirical attack's performance can be useful evidence, but it must be interpreted against its prior, baseline and observed release. This lesson supplies exact finite examples and explicit observer assumptions. It does not equate a failed attack with a proof of privacy or a posterior change with a DP violation.

Next, fairness metrics and identification assumptions will separate quantities observable in evaluation data from stronger claims about a population or intervention.
