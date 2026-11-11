## Why the object to protect is an entire transcript

An analyst can compare several answers, so privacy must constrain their joint distribution. A promise about each answer considered separately does not automatically constrain that joint distribution. In this unit the database is a single protected bit x in {0,1}; neighboring inputs replace 0 by 1. We always compare the same individual across repeated releases. This differs from the preceding unit, where many individuals each randomized once and only one record changed between neighboring databases.

Think of a transcript as everything the analyst can observe: outputs, order and any disclosed choices. Counting only the number of API requests is insufficient. Copying an answer, obtaining a fresh answer and revealing a random seed can have different effects even if they all display one extra number.

The experiment offers three fully specified mechanisms. Fresh mode generates k conditionally independent randomized responses about x. Copy mode generates one response and repeats that public response k times. XOR mode releases two correlated bits (R,x XOR R) for a fresh fair bit R. The k and p controls are explicitly unused in XOR mode. The last mechanism illustrates an invalid composition argument, not a privacy-preserving protocol.

## Epsilon and delta describe a chosen claim

For transcript laws P and Q, a claim (epsilon,delta) means

$$
P(S)\le e^\epsilon Q(S)+\delta
$$

for every event S, in both neighboring directions. Write alpha=exp(epsilon). The exact smallest delta at a fixed alpha is

$$
\delta_*(\alpha)=\max\left\{\sum_z[P_z-\alpha Q_z]_+,\ \sum_z[Q_z-\alpha P_z]_+\right\}.
$$

This follows by including every positive atom excess and excluding every negative one. We do not enumerate only plausible attacks: the maximizing-event identity covers all events. Empty events and atoms with zero probability contribute zero. An atom with positive P and zero Q contributes its full P mass for every finite alpha.

The total-claim controls select a point on this profile. They change a claim rather than the mechanism. A separate per-step alpha0 control selects a different budget used to illustrate the composition theorem. Keeping those two roles separate prevents a common error: declaring a total epsilon equal to a per-query epsilon while continuing to collect fresh information.

## Derive the fresh-response law

A response equals x with probability p and its complement with probability 1-p. For x=0 the probability of a particular transcript with t reported ones is

$$
(1-p)^t p^{k-t}.
$$

There are binomial(k,t) such transcripts. Therefore the displayed count T has laws

$$
P(T=t\mid x=0)={k\choose t}(1-p)^t p^{k-t},\qquad
P(T=t\mid x=1)={k\choose t}p^t(1-p)^{k-t}.
$$

Why may a count plot stand in for the full transcript? For 1/2<=p<1, every transcript in a given count class has the same likelihood ratio. The sign of its excess also agrees throughout that class. Grouping the atoms by count thus preserves the sum of positive excesses exactly. At p=1 the two deterministic transcripts still have different counts and the same conclusion holds directly. This is a special sufficient-statistic fact for this channel; arbitrary coarsening need not preserve exact privacy loss.

For 1/2<=p<1, a transcript ratio is a power of p/(1-p), and its largest value is

$$
\left(\frac p{1-p}\right)^k.
$$

The all-zero transcript attains that ratio in the x=0 to x=1 direction; the all-one transcript does so in reverse. Hence the tight pure epsilon is k ln(p/(1-p)). At p=1/2 it is zero. At p=1 the supports are disjoint, so no finite pure epsilon exists.

## Work the two-release example by hand

Load the second guide: p=3/4, k=2, fresh mode, claimed alpha3 and delta0. The output-count distributions are

$$
P=(9,6,1)/16,\qquad Q=(1,6,9)/16.
$$

For event {0}, the excess is 9/16 minus 3 times 1/16, equal to 6/16=3/8. Other atoms have nonpositive excess in this direction. Reversing the inputs produces the same maximum with event {2}. Thus the exact required delta is 3/8 and the selected pure claim fails.

One response had tight epsilon ln3. Two fresh responses have tight epsilon ln9, approximately2.197. Selecting alpha9 makes the exact required delta zero. A new sample can strengthen evidence about the same bit; that is the information being accounted for.

With k=3 at alpha3 the laws are (27,27,9,1)/64 and their reverse. Positive excess is24/64 at output0, while output1 has exactly zero excess. Required delta is again3/8. Additional releases cannot reduce the full-transcript privacy profile, because discarding the last release is postprocessing, but the increase need not be strict at every budget.

## Copying an output is postprocessing

Now keep k=2 and p=3/4 but choose copy mode. Draw B once, then display (B,B). The count laws become

$$
P=(3/4,0,1/4),\qquad Q=(1/4,0,3/4).
$$

The likelihood ratio is still at most3, so alpha3 requires delta0. The transcript can be reconstructed from B and B can be recovered from the transcript. Their privacy profiles are exactly equal. There is one underlying query, not two independent accesses to the protected bit.

Do not generalize this observation into permission to reuse arbitrary noise with raw data. Copying a value that is already public is a function of that value alone. A second computation that also accesses x can reveal extra information. The XOR example below demonstrates the difference.

## Conditional guarantees give an additive budget

[Dwork and Roth, §3.5 and Appendix B](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) state basic composition: when each stage has a valid (epsilon_i,delta_i) guarantee for every fixed prior transcript, the full transcript has the sufficient guarantee

$$
\left(\sum_i\epsilon_i,\ \min\{1,\sum_i\delta_i\}\right).
$$

Independent fresh randomization makes the product-law calculation in this experiment valid. Independence is not a necessary assumption of the general adaptive composition theorem. Its relevant requirement is a valid conditional mechanism at every history, including how an algorithm responds to a history rather than only its unconditional output marginal.

For pure privacy, write a transcript probability as the product of conditional stage probabilities. Applying a factor exp(epsilon_i) to each conditional likelihood ratio gives exp(sum epsilon_i). Summing over any event gives the same bound. With zero-support atoms, use the corresponding probability inequalities directly rather than divide by zero.

For approximate privacy, one must avoid multiplying an unbounded slack at every branch. A finite two-stage derivation clarifies the issue. Let S_s be the second-stage section of an output event after first output s. Define

$$
h(s)=\min\{1,e^{\epsilon_2}Q_2(S_s\mid s)\}.
$$

The second-stage condition bounds P_2(S_s|s) by h(s)+delta2. The first-stage event inequality also bounds expectations of every function h in [0,1]: integrate its threshold events {h>u} over u in [0,1]. Combining these two facts gives

$$
\begin{aligned}
P(S)&\le \mathbb E_{P_1}h+\delta_2\\
&\le e^{\epsilon_1}\mathbb E_{Q_1}h+\delta_1+\delta_2\\
&\le e^{\epsilon_1+\epsilon_2}Q(S)+\delta_1+\delta_2.
\end{aligned}
$$

Repeat in reverse and induct on the number of stages. This argument treats delta as event slack, without inventing an independently labeled set of exposed people or a known failure flag.

## Compare the theorem with the exact profile

The separate alpha0 control chooses a per-response epsilon0=ln(alpha0). The model computes the exact single-response delta0 at that budget, then reports the sufficient composition pair (q epsilon0,min(1,q delta0)), where q=k for fresh mode and q=1 for copy mode. It also evaluates the actual mechanism's exact required delta at the composed epsilon. These numbers refer to the composed budget, whereas the blue/orange plot refers to the selected total claim.

For p=.9, alpha0=3 and k=2, a single response requires delta0=.9-3(.1)=.6. The theorem gives composed epsilon ln9 and delta bound1 after capping1. Direct calculation is much sharper: fresh count laws are(.81,.18,.01) and(.01,.18,.81), so required delta at alpha9 is .81-9(.01)=.72. The theorem remains valid; a sufficient worst-case bound need not be tight for a particular mechanism.

A bound of delta1 is mathematically trivial for any two probability laws. It is not evidence of useful privacy. Nor should an analyst silently lower delta below the exact requirement because a stronger-looking parameter is desired.

## A shared mask defeats a marginal-only argument

In XOR mode a fair bit R produces (R,x XOR R). For x=0 the joint outputs00 and11 each have probability1/2. For x=1 the outputs01 and10 each have probability1/2. Each coordinate alone is fair regardless of x, so each marginal has epsilon0. But the joint supports are disjoint and their XOR equals x.

After observing R=r, the second output is deterministically x XOR r. Its conditional law does depend on x and has no finite pure epsilon. Therefore the conditional hypothesis required for adding two zero budgets is false. The experiment displays no theorem budget for this mode. Its joint minimum delta is1 for every finite alpha.

The encoded horizontal output is2R+(x XOR R), not the number of ones. Encoding retains both bits. Replacing it by a count would be a different mechanism and could merge outputs; do not change the released object midway through the argument.

## Exact calculations and practical boundaries

The model uses integer p percentages. Fresh transcript-count weights have common denominator100^k, which can exceed JavaScript's safe integer range. Binomial coefficients, positive-excess numerators and claim cross-products therefore use BigInt. Returned fields use strings for exact integers/fractions and ordinary numbers for plots and readouts. No BigInt reaches JSON serialization.

At p=.5 fresh and copy laws are identical across inputs, with epsilon0; XOR mode still reveals x because it is a different mechanism and ignores p. At p=1 fresh/copy releases are deterministic and require delta1 at finite alpha. With k=1 fresh and copy profiles coincide for every allowed claim. Zero-probability interior outputs in copy mode are retained in the table, not smoothed artificially.

The graph evaluates integer alpha1 through100 exactly and uses segments only to guide the eye. The selected point is recomputed at the input's tenth-grid alpha, even between plotted grid points. This finite calculation does not certify unmodeled releases, data-dependent stopping rules or an arbitrary application's privacy accountant.

## Research extension: Moment accounting before choosing epsilon and delta

An entire privacy profile contains more information than one chosen budget. A useful alternative is a moment of the privacy loss. For a common-support pair and order $r>1$, define

$$
D_r(P\Vert Q)=\frac{1}{r-1}\log\sum_y P_y^rQ_y^{1-r},\qquad
\mathbb E_P e^{(r-1)L} = e^{(r-1)D_r(P\Vert Q)}.
$$

For independent fresh transcripts, the sum factors into a product, so these divergences add. For copying an already released answer, the transcript and original answer determine each other; the divergence stays unchanged. A zero $Q$ atom with positive $P$ instead makes this finite-moment argument unavailable. Do not apply it to the revealing XOR construction.

Markov's inequality bounds $P(L>\epsilon)$ by $\exp((r-1)(D_r-\epsilon))$. On the complementary outputs $P_y\le e^\epsilon Q_y$, while the remaining $P$ mass bounds their event excess. Thus a divergence bound $d_r$ valid in both ordered directions gives a sufficient approximate guarantee

$$
\epsilon=d_r+\frac{\log(1/\delta)}{r-1},\qquad 0<\delta<1.
$$

This is a sufficient conversion, not the exact hockey-stick profile. Orders may be compared using deterministic bounds; picking a flattering order from unvalidated estimates would be a different problem. The framework is developed in [Mironov's Rényi differential privacy paper](https://arxiv.org/abs/1702.07476).

For randomized response with $p=3/4$, order two has moment $7/3$. Fresh copies have $d_2=k\log(7/3)$; repeating one public answer still has $d_2=\log(7/3)$. At $\delta=1/10$:

| Fresh responses $k$ | Converted epsilon, approximately | Tight pure epsilon, approximately |
|---|---:|---:|
| 1 | 3.149883 | 1.098612 |
| 4 | 5.691777 | 4.394449 |
| 12 | 12.470159 | 13.183347 |

The conversion can be looser than the pure bound for short transcripts. Both valid bounds can be retained; this table does not claim universal improvement. The code computes count probabilities independently and confirms the product moment as an exact fraction before taking display logarithms.

```python
from fractions import Fraction as F
from math import comb,log
p=F(3,4); delta=F(1,10)
for k in [1,4,12]:
    P=[comb(k,t)*p**(k-t)*(1-p)**t for t in range(k+1)]
    Q=P[::-1]
    moment=sum(a*a/b for a,b in zip(P,Q))
    assert moment==F(7,3)**k
    copied_moment=p*p/(1-p)+(1-p)**2/p
    assert copied_moment==F(7,3)
    converted=log(moment.numerator)-log(moment.denominator)+log(1/float(delta))
    print(k,f'{converted:.6f}',f'{k*log(3):.6f}')
```

## Exercises and worked solutions

1. For p=3/4,k=2,fresh mode, find the smallest delta at alpha1. The difference vector is(8,0,-8)/16. Each direction has maximum1/2, so this is a(0,1/2) guarantee.
2. At the same settings with alpha3, does delta.374 pass? No:3/8=.375. Delta.375 passes exactly. The badge compares rational cross-products, not rounded display values.
3. Why does copying k=12 times retain epsilon ln3 at p3/4? Only one noisy bit is produced. The deterministic length12 constant transcript is recoverable from that bit and vice versa, preserving the whole profile.
4. Compute the example p=.9,k2,alpha0=3. The single delta is.6, the capped theorem delta bound is1 at epsilon ln9, and the exact delta there is.72. These are three different quantities with explicit roles.
5. Which assumption fails for the XOR counterexample? Conditional privacy of the second stage given the first output. Independence alone is not the theorem's general condition; the marginal statements omit the relevant conditioning.
6. Is postprocessing a reason to ignore a fresh rerun of a query? No. A rerun accesses the private input again. A function only of an already disclosed answer is postprocessing; these are different computations.

## Toward privacy-preserving learning

Iterative training can release a sequence of noisy updates. A valid analysis must specify adjacency, sensitivity, sampling, clipping, noise and what is released at each stage. This unit isolates the composition logic on an exactly enumerable mechanism. It does not claim that simply summing displayed parameters proves a training pipeline private. Sharper accountants and advanced composition are further tools whose assumptions and full transcript must also be checked.

Next, sensitivity and noise mechanisms explain how to obtain a valid single-stage guarantee before composing it.
