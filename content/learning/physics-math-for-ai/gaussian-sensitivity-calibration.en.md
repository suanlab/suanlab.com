## The same sensitivity can require a different noise analysis

Keep the public fixed n, input clipping interval[0,B] and replace-one adjacency from the Laplace lesson. The sum, mean, fixed-predicate count and constant have global sensitivities B,B/n,1 and0 respectively. The global bounds still require both an upper-bound proof and an attaining pair. What changes here is the distribution added to the query.

Release M(D)=f(D)+Z with Z normal of mean0 and variance sigma². Sigma is the standard deviation in output units. It is not the variance and it is not the Laplace scale b. At the same numerical scale, Gaussian MSE is sigma² whereas Laplace MSE is2b². This is an accuracy comparison; it does not imply identical privacy.

The density plot uses the extremal centers0 and Delta after translation. A separately queried neighboring pair may have a smaller gap. Its easier privacy problem cannot replace the global one.

## Derive the log likelihood ratio

For two means0 and d>0 with the same standard deviation sigma,

$$
p(y)=\frac{e^{-y^2/(2\sigma^2)}}{\sigma\sqrt{2\pi}},\qquad
q(y)=\frac{e^{-(y-d)^2/(2\sigma^2)}}{\sigma\sqrt{2\pi}}.
$$

Subtracting the squared terms gives

$$
L(y)=\log\frac{p(y)}{q(y)}=\frac{d^2-2dy}{2\sigma^2}.
$$

This affine expression has no finite upper bound as y tends to negative infinity. Therefore no finite pure epsilon works when the query has an attained positive sensitivity and sigma is finite. A finite horizontal plot may hide that tail, but hiding part of a plot does not change the release distribution.

This reasoning concerns the full untruncated Gaussian mechanism. A bounded or rounded output mechanism needs its own distributional analysis. It is not obtained by pretending that the Gaussian density vanishes outside the displayed axes.

## The optimal event is a half-line

For a claimed epsilon>=0, p(y)-exp(epsilon)q(y) is positive when L(y)>epsilon. Solving this inequality gives

$$
y<t,\qquad t=\frac d2-\frac{\epsilon\sigma^2}{d}.
$$

For any measurable event A, its integral of the density excess is bounded by the integral of the positive part over the whole real line. Thus the half-line above attains the all-event maximum. Integrate p and q over that half-line:

$$
\delta_d(\epsilon)=
\Phi\!\left(\frac d{2\sigma}-\frac{\epsilon\sigma}{d}\right)
-e^\epsilon\Phi\!\left(-\frac d{2\sigma}-\frac{\epsilon\sigma}{d}\right).
$$

Here Phi is the standard normal CDF. Reflecting the two laws gives the same reverse-direction maximum. This derivation distinguishes weighted event excess from simply the probability that privacy loss exceeds epsilon.

[Balle and Wang, Theorem8](https://arxiv.org/pdf/1805.06530) give the analytic Gaussian condition, using global L2 sensitivity. The scalar derivation above makes its two CDF terms explicit for this experiment. It does not use a sufficient classical tail bound outside that bound's parameter regime.

## Why the worst gap is global sensitivity

Let h=d/sigma. The two arguments are a=h/2-epsilon/h and c=-h/2-epsilon/h. Their normal densities obey phi(a)=exp(epsilon)phi(c). Differentiate the profile with respect to h; the terms proportional to epsilon/h² cancel, leaving

$$
\frac{d\delta_h(\epsilon)}{dh}=\phi(h/2-\epsilon/h)>0.
$$

Therefore the required delta increases with the neighboring gap. Our bounded queries have pairs attaining d=Delta, so using Delta gives the exact global profile. If Delta0, use delta0 directly instead of dividing by zero: the output distributions are identical.

For d>0, finite sigma and finite epsilon, the maximizing half-line has strictly positive density excess on a set of positive length. Its integral is positive, however tiny it may be. This is why an underflowed floating number cannot be accepted as a pure guarantee.

## Work the first guide by hand

Set n4,B1 and the sum query, so Delta1. With sigma1 and epsilon1, the optimal threshold is-.5. Thus

$$
\delta_*=\Phi(-.5)-e\,\Phi(-1.5)\approx .1269367375.
$$

Claimed delta.13 is sufficient, while delta.126 is not. The mean-square noise error is1. There is still no finite pure epsilon: the approximate guarantee permits nonzero additive event slack.

This number is not the probability that a particular person will be revealed, nor the proportion of records that can be ignored. It is the maximum weighted event excess for the specified neighboring output laws.

## A mean can reduce privacy loss without making it zero

The second guide keeps n4,B1,sigma1 and epsilon1 but changes the query to the mean. Global sensitivity becomes.25. The two CDF arguments are-3.875 and-4.125, giving required delta approximately2.9242721 times10^-6.

That value is much smaller than for the sum with the same numerical noise, but it remains positive. Claim delta0 fails. The noise is now measured in mean-output units; multiplying the released mean by4 produces standard deviation4 in sum units. A fair comparison must account for that transformation rather than compare bare MSE numbers.

Increasing n reduces mean sensitivity under this fixed-size replacement model. It does not justify changing to add/remove adjacency without reanalyzing the query and its denominator.

## Underflow is a numerical event, not a privacy theorem

The third guide uses mean sensitivity1/32, sigma10 and epsilon10. Its Gaussian tail probabilities are too small for an ordinary double-precision probability representation. The plotted required delta becomes numeric0, but log10(delta) is about-2223595.4904.

The result retains three distinct facts: the mathematical delta is positive; the ordinary probability has underflowed; its logarithm is finite. A delta0 claim is rejected for positive sensitivity regardless of the displayed decimal. A positive-delta claim is compared in log space.

The distinction also matters before total underflow. Subtracting two nearly equal tail probabilities can lose significant digits. Rounding a positive value to three decimals is not evidence of zero. The density window may put the optimal threshold far off-screen; the certificate explicitly says when this occurs.

## Evaluate the analytic formula without subtracting tiny tails

For negative large a and c, the expression has two difficult terms. Write R(z)=Phi(-z)/phi(z), the Gaussian Mills ratio. Because phi(a)=exp(epsilon)phi(c), when a<=-1 the profile can be evaluated as

$$
\delta=\phi(a)\bigl(R(-a)-R(-c)\bigr).
$$

The implementation evaluates this difference through paired continued-fraction recurrences and stores the logarithm. A power series handles the central CDF interval. Continued-fraction identities are documented in [NIST DLMF7.9](https://dlmf.nist.gov/7.9); the classroom implementation is independently checked against180 reference values computed with100 decimal digits, including probabilities that underflow in double precision.

These are floating evaluations of an analytic profile. They are not exact rational certificates or a claim that an ordinary machine Gaussian sampler implements ideal real arithmetic. The mathematical all-event guarantee comes from the distributional proof, while numerical tests assess the bounded implementation used for instruction.

## Error, calibration and units

The noise is unbiased around the clipped query and has

$$
\mathbb E[Z^2]=\sigma^2,\qquad
\mathbb E|Z|=\sigma\sqrt{2/\pi}.
$$

At sigma1 the mean absolute noise is approximately.79788456. About95% of draws fall within1.96 standard deviations; this accuracy statement is different from a DP delta. Neither statistic replaces the event-excess calculation.

For a desired positive approximate budget, one may solve the monotone analytic profile for sigma. This lesson exposes sigma directly so readers can compare the whole tradeoff. It does not silently claim to implement an optimal root solver. Increasing sigma reduces the gap-to-noise ratio; the derivative argument above therefore shows a nonincreasing privacy profile at fixed sensitivity and epsilon.

If the query is constant, no noise is necessary. The lab still retains its selected positive sigma and shows two identical Gaussian laws, sensitivity0 and epsilon0. Their positive estimation error is an intentional choice of the experiment, not a necessary privacy cost.

## Read the three panels together

The density panel uses the global extremal gap. The certificate separately shows the inspection neighbor, global sensitivity, approximate delta, claim and accuracy. The last plot shows epsilon against required global delta, with a horizontal claimed-delta line. All three depend on the same declared query and noise family.

The graph uses epsilon itself on the horizontal axis, unlike the alpha=exp(epsilon) axis in earlier finite-channel lessons. Its density height has inverse-output units and need not be bounded by1. Its finite display window omits tails visually while the analytic formula covers the full real support.

Input clipping and output clipping remain distinct. Public input clipping establishes the domain bound; output clipping would be postprocessing of the noisy answer and would change its exact profile and risk. This experiment leaves the released real value unclipped.

## Research extension: Whiten a correlated Gaussian release

For a vector release, covariance changes which neighboring directions are easy to distinguish. Take a positive-definite covariance matrix $\Sigma$ fixed publicly across databases. Compare $P=N(0,\Sigma)$ and $Q=N(d,\Sigma)$. Completing the square in their densities gives

$$
L(Y)=\frac12d^\top\Sigma^{-1}d-d^\top\Sigma^{-1}Y,\qquad
 h^2=d^\top\Sigma^{-1}d.
$$

Under $P$, this scalar privacy loss is normal with mean $h^2/2$ and variance $h^2$. An invertible whitening map preserves all event comparisons. Therefore the same analytic Gaussian profile applies with gap-to-noise ratio $h$, even though the output is multidimensional. Taking its exponential moment also gives

$$
D_r(P\Vert Q)=\frac{r h^2}{2},\qquad r>1.
$$

For a query, maximize this squared Mahalanobis gap over every legal neighboring difference. Choosing the covariance after seeing the private input would change the mechanism and invalidate this fixed-covariance argument.

With unit marginal variances and correlation $\rho$, $\Sigma=(1,\rho;\rho,1)$ is positive definite only for $|\rho|<1$. The common and opposing directions have different squared gaps:

| Correlation | $d=(1,1)$: $h^2$ | $d=(1,-1)$: $h^2$ |
|---|---:|---:|
| -1/2 | 4 | 4/3 |
| 0 | 2 | 2 |
| 1/2 | 4/3 | 4 |

Positive correlation helps the common direction here while worsening the opposing one. It is not a general privacy improvement. At $|\rho|=1$ the matrix is singular; a density with this inverse is unavailable, and support containment must be analyzed separately. The scalar native Gaussian lab remains unchanged.

The code independently inverts each rational matrix, checks the privacy-loss variance and calculates the order-two divergence, which equals $h^2$. Whitening geometry and moment accounting follow from the stated Gaussian laws; this is not a claim about a machine sampler or an unspecified training pipeline.

```python
from fractions import Fraction as F
for rho in [F(-1,2),F(0),F(1,2)]:
    S=[[F(1),rho],[rho,F(1)]]; det=1-rho*rho
    inverse=[[F(1)/det,-rho/det],[-rho/det,F(1)/det]]
    gaps=[]
    for d in [[F(1),F(1)],[F(1),F(-1)]]:
        v=[sum(inverse[i][j]*d[j] for j in range(2)) for i in range(2)]
        h2=sum(d[i]*v[i] for i in range(2))
        variance=sum(v[i]*S[i][j]*v[j] for i in range(2) for j in range(2))
        assert variance==h2 and h2>0
        gaps.append(h2)
    assert gaps==[2/(1+rho),2/(1-rho)]
    print(rho,*gaps)
```

## Exercises and worked solutions

1. At Delta1,sigma1,epsilon0, find the required delta. It is2Phi(.5)-1, approximately.3829249225.
2. Why does a finite sigma fail to give pure DP for a nonconstant query? The affine log likelihood ratio is unbounded on a real tail, producing a positive violating event for every finite epsilon.
3. If the plot reports delta0 but its positivity flag is1, may you claim pure privacy? No. Read the underflow flag and log probability; numerical zero is not analytic zero.
4. What happens to the mean sensitivity when fixed n doubles? It halves. Keeping sigma fixed then reduces the gap-to-noise ratio, but changes no output units automatically.
5. At the first guide's settings, compare delta.126 and.13. The exact analytic requirement is about.12693674; the former fails and the latter passes.
6. Which norm matters for a vector isotropic Gaussian release? The global L2 sensitivity of the entire released vector. A scalar or coordinatewise bound alone is insufficient without combining all coordinates correctly.

## Research connection

Private iterative optimization must specify clipping, sampling, noise, released updates and composition together. This lesson supplies a transparent single-stage Gaussian analysis, not a certificate for any training loop using Gaussian noise. The next scope topic studies what an observer can infer from a release and how likelihood ratios constrain posterior evidence.
