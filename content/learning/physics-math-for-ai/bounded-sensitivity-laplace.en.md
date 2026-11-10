## Start with the change that privacy must protect

Differential privacy compares output distributions on neighboring datasets. Before selecting a noise parameter, specify what counts as a neighboring change. This lesson uses a public fixed number n of records. Each record is clipped into a public fixed interval[0,B], and one person's value may be replaced by any other value in that interval. The roster and n are not hidden by this model.

This replace-one convention differs from adding or removing a row. A histogram can change two bins under replacement, even though only one person changes. An average with a changing denominator is a different query again. A sensitivity formula belongs to its adjacency relation and output definition, not merely to the word “average.”

The public bound is a mathematical assumption of the mechanism. Taking the maximum value observed in private data and declaring it public can itself disclose information. The experiment treats B as fixed independently of the protected data.

## Global sensitivity is a supremum, not one observed difference

For a real query f, define

$$
\Delta f=\sup_{D\sim D'}|f(D)-f(D')|.
$$

The bound ranges over the entire declared input domain. If a displayed pair differs only slightly, that does not imply a smaller global bound. Local sensitivity is also a different object: it fixes one dataset and maximizes over its neighbors. A single queried difference is neither the global supremum nor automatically local sensitivity.

The experiment displays the all-zero dataset and a neighbor obtained by replacing one record with a requested value, clamped to B. That inspection pair helps interpret the formulas. Separately, the density diagram uses an extremal pair attaining global sensitivity. Keep these two roles distinct.

## Derive four sensitivities and show they are attained

For the sum, all unchanged coordinates cancel. Its difference is the difference of the replaced values, at most B. Replacing a0 with B attains the bound, so Delta_sum=B.

For the fixed-size mean,

$$
\left|\frac1n\sum_i x_i-\frac1n\sum_i x'_i\right|
=\frac{|x_j-x'_j|}{n}\le\frac Bn.
$$

The same endpoint replacement attains B/n. This proof uses fixed n. For the count of the fixed predicate x>B/2, one indicator can change by at most1, and replacing0 with B attains1 because B is positive. A constant query f(D)=7 never changes, so its sensitivity is0.

These are exact global results for continuous records in[0,B], established by an upper-bound argument and an attaining pair. Enumerating a finite grid alone would not prove the upper bound over all real records. With no finite public record bound, the sum and fixed-size mean have unbounded replacement sensitivity.

## Add Laplace noise with the correct units

Release M(D)=f(D)+Z, where Z has Laplace density

$$
p_Z(z)=\frac1{2b}e^{-|z|/b},\qquad b>0.
$$

Scale b has the same units as f. Sensitivity has those same units, so Delta/b is dimensionless, as epsilon must be. Confusing b with the standard deviation introduces a factor sqrt2: Laplace variance is2b², not b².

[Dwork and Roth, §3.3](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) give the sensitivity-calibrated Laplace mechanism. Here is a direct scalar proof under our explicitly stated replacement adjacency. For neighboring centers u and v,

$$
\frac{p_u(y)}{p_v(y)}
=\exp\!\left(\frac{|y-v|-|y-u|}{b}\right)
\le\exp\!\left(\frac{|u-v|}{b}\right)
\le e^{\Delta/b}.
$$

The triangle inequality supplies the first bound. Integrating this pointwise inequality over any measurable output event gives the event definition of pure DP, in both directions. Thus choosing b>=Delta/epsilon suffices for epsilon>0. For an attained nonzero gap Delta, the extreme tail attains the density ratio exp(Delta/b), making the pure parameter tight for this untruncated scalar mechanism.

## Find the worst event when epsilon is too small

A failed pure claim can still have an approximate guarantee. Translate neighboring centers to0 and d>0, and write alpha=exp(epsilon). Their log density ratio is d/b for y<=0, (d-2y)/b for0<y<d and -d/b for y>=d.

If epsilon>=d/b, all event excesses are nonpositive and the required delta is0. Otherwise the density excess is positive exactly on the half-line

$$
S=(-\infty,t),\qquad t=\frac{d-b\epsilon}{2}.
$$

For any event A, including points where the excess is negative cannot increase its integral. Thus integrating over this full positive half-line is a proof of the all-event maximum, not a search over selected attacks. Reflection gives the same answer in the reverse direction.

Using the Laplace CDF on each side of the two centers gives

$$
\delta_d(\epsilon)=
\begin{cases}
0,&\epsilon\ge d/b,\\
1-\exp((\epsilon-d/b)/2),&0\le\epsilon<d/b.
\end{cases}
$$

This expression increases with d at a fixed budget. Since a pair attains d=Delta, plugging in global sensitivity gives the exact global profile. At d=0 the two laws are identical and every nonnegative epsilon works with delta0.

## Work three examples by hand

In the first guide n4,B2 and the sum query have Delta2. Scale b2 gives tight pure epsilon1 and required delta0 at claimed epsilon1. Its noise MSE is8.

The second guide uses the mean of the same bounded data. Sensitivity is2/4=.5 and b=.5 again gives tight pure epsilon1. MSE is .5 in mean-output units. Multiply the noisy mean by4 to compare in sum units: its error variance becomes16 times .5=8, matching the calibrated sum mechanism. A smaller number alone is not evidence of an improved dimensionless privacy–utility tradeoff.

The third guide returns to the sum with b1 and epsilon1. Its global gap is2, so required delta is1-exp(-1/2), approximately .393469. The displayed inspection neighbor replaces0 with1 and has gap1; that pair alone needs delta0. Its passing test does not certify all other neighbors. Replacing the query record with2 exposes the extremal pair directly.

## Accuracy is a separate calculation

The noise has mean0, so the released value is an unbiased estimate of the clipped query. Its error measures are

$$
\mathbb E[Z^2]=2b^2,\qquad
\mathbb E|Z|=b,\qquad
\Pr(|Z|>r)=e^{-r/b}\quad(r\ge0).
$$

For example b2 has MSE8, mean absolute error2 and a95% absolute-error radius2 ln20, approximately5.9915. These are noise errors about the clipped query. If the scientific target is an unclipped mean, clipping can add bias relative to that target; the privacy calculation does not remove that modeling error.

The real-valued release is not clipped back into a plausible range. Such output clipping is postprocessing and preserves a valid DP upper bound, but changes bias, risk and potentially the exact privacy profile. Input clipping establishes sensitivity; output clipping changes an already noisy answer. They are different operations.

## Read the diagrams without dropping the tails

The first diagram shows densities for centers0 and Delta after translating f(D0) to0. Its vertical axis is density, which may exceed1; only an integral is a probability. The shaded visible portion belongs to the maximizing half-line. The displayed window extends six scales beyond the centers, while the mechanism has support on the entire real line.

The exact all-event argument integrates the whole half-line, including its off-screen part. A finite plot or a few sampled output values cannot prove a bound on all real outputs. The profile graph evaluates the analytic formula in floating arithmetic at epsilon steps .1; connecting segments guide the eye.

The badge uses an integer cross-product for the pure Laplace boundary associated with the integer controls. Positive-delta values are floating analytic evaluations and are labeled approximations. A displayed rounded value should not replace its underlying computation.

## Constant queries and epsilon zero

A constant query reveals no information about its input, so it may be released with no noise at all. The experiment retains the selected positive b to keep a single density family; both neighboring output laws then remain identical. That intentional extra noise is not required for privacy.

For a nonconstant query with positive sensitivity and finite b, epsilon0 requires positive delta. Sending b to infinity is a limit, not a finite useful zero-epsilon pure mechanism. Do not divide by epsilon0 in the calibration formula b=Delta/epsilon.

## Exercises and worked solutions

1. For n8 and B4, compute sensitivities of sum and mean. They are4 and.5. Endpoint replacement attains both.
2. For mean sensitivity.5 and desired pure epsilon.25, choose the smallest Laplace b. It is2, giving MSE8 in mean-output squared units.
3. For Delta2,b1,epsilon1, which event witnesses failure? Use translated centers0 and2 and the half-line y<.5. Its probability excess is1-exp(-.5).
4. Does a query replacement value0 eliminate global sensitivity? No. Only the displayed pair becomes identical; another endpoint replacement still attains the original global bound.
5. Why is an observed sample standard deviation insufficient for calibration? It neither bounds all legal replacements nor supplies a public data-independent sensitivity bound. It can also be a private statistic.
6. For a constant query with b1, is MSE2 necessary? No. It is the experiment's chosen added noise; the constant could be disclosed exactly.

## Connect calibration to composition and learning

A valid single-query mechanism is the input to the composition analysis from the previous lesson. Changing the clipping domain, releasing several coordinates or revealing additional intermediate statistics changes that mechanism and must be reflected in the analysis. For vectors, independent Laplace coordinates are calibrated to an L1 sensitivity bound of the full vector; a scalar mean formula cannot certify an arbitrary gradient release.

The next companion lesson derives the Gaussian profile. Gaussian standard deviation is not Laplace scale, and nonzero Gaussian sensitivity has a different pure-privacy boundary.
