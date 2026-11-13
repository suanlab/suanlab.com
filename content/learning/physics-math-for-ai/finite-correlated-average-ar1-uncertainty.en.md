## Repeated observations are not automatically repeated information

Suppose a sensor reports a noisy value every second. If the noise stays high for several consecutive readings, averaging ten readings does not cancel it as effectively as averaging ten independent readings. Conversely, if positive and negative deviations tend to alternate, averaging a carefully specified pair can cancel some noise exceptionally well. The number of stored values alone does not determine the uncertainty of their average.

The preceding chapters described covariance between two spin components and between spatial sites of an Ising ring. Here the index labels consecutive observations in a stationary stochastic preparation. We will calculate the exact finite variance of their mean, derive a separate large-sample approximation, and examine negative dependence and degenerate endpoints. A spatial distance, an observation index and an animation frame remain different quantities.

You should know covariance, finite geometric sums and the variance of a sum of independent variables. [Geyer's notes on autocovariances](https://www.stat.umn.edu/geyer/8054/notes/initseq.html) provide background on dependence and uncertainty in averages. The stationary unit-variance AR(1) preparation, finite bounds and examples below are specified and derived separately here.

## Fix the stationary preparation before calculating uncertainty

Write each observation as an unknown common mean plus a centered fluctuation. Define

$$
X_i=\mu+Y_i,\qquad Y_1\sim\mathcal N(0,1),\qquad Y_{i+1}=\rho Y_i+\sqrt{1-\rho^2}\,\varepsilon_i,\qquad \varepsilon_i\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

The initial fluctuation and all innovations are independent. For |ρ|<1, this preparation is stationary: each Y has mean zero and variance one. Indeed, its next variance is ρ squared plus 1−ρ squared, equal to one. Starting instead from Y1=0 would produce a transient preparation and different early covariances. Fixing innovation variance to one would also define a different marginal variance, 1/(1−ρ squared).

The model accepts an integer sample count n=1 through 512 and a finite binary64 coefficient ρ in [−1,1]. The endpoints are explicitly included as degenerate stationary preparations with zero innovation variance; they are not labeled as mixing processes. The coefficient is specified, not estimated from a short path. The numerical mean expectation is the centered error expectation zero; uncertainty does not depend on the unknown shift μ.

## Derive the autocovariance rather than guessing it

Iterating the recursion for k steps gives the old fluctuation multiplied by ρ to the power k plus later innovations. Those innovations are independent of the old fluctuation. Hence

$$
\gamma_k=\operatorname{Cov}(Y_i,Y_{i+k})=\rho^k,\qquad \gamma_0=1,\qquad \Sigma_{ij}=\rho^{|i-j|}.
$$

For positive ρ, successive covariances have the same positive sign. For negative ρ, their signs alternate. This does not make the covariance matrix negative: any quadratic form is the variance of a linear combination of observations and must be nonnegative. In the interior |ρ|<1 the independent innovations give a nonsingular finite Gaussian preparation. At the endpoints the covariance becomes singular because every observation is a multiple of the same initial random value.

Stationarity is doing real work. It makes covariance depend on the lag k rather than the absolute starting index i. A drifting mean, changing noise scale or changing coefficient need not satisfy this law. A plot that happens to alternate is not enough to establish the model assumptions.

## Count every covariance pair in the finite mean

Define the sample mean and its centered error. Their variance is

$$
\overline X_n=\frac1n\sum_{i=1}^nX_i,\qquad \operatorname{Var}(\overline X_n)=V_n=\frac1{n^2}\sum_{i,j=1}^n\rho^{|i-j|}=\frac{n+2\sum_{k=1}^{n-1}(n-k)\rho^k}{n^2}.
$$

There are n diagonal pairs, each contributing one. For lag k there are n−k pairs above the diagonal and another n−k below it. These counts explain both the factor two and the boundary weight n−k. Replacing every weight by n is a large-sample operation, not the exact finite calculation.

At ρ=0, only the diagonal remains and V is 1/n. At nonzero ρ, simply dividing the marginal variance by n discards cross terms. Because the sample mean is unbiased for the common μ in this preparation, V is also its mean-squared error. With a biased or drifting preparation, mean-squared error requires a separate squared-bias term.

## Reexpress the answer as positive independent contributions

An alternating sum is a useful derivation but can be a poor numerical algorithm. Collect the independent random inputs instead. Let a_m be a finite geometric sum:

$$
a_m=\sum_{k=0}^{m-1}\rho^k,\qquad \sum_{i=1}^nY_i=a_nY_1+\sqrt{1-\rho^2}\sum_{j=1}^{n-1}a_{n-j}\varepsilon_j.
$$

The first term accounts for the initial preparation, and the remaining terms account for innovations injected at different observation times. Since all these inputs are independent with variance one,

$$
V_n=\frac{a_n^2+(1-\rho^2)\sum_{m=1}^{n-1}a_m^2}{n^2}.
$$

Every contribution is nonnegative. This identity preserves a tiny positive variance near ρ=−1 without subtracting two large rounded numbers. It also makes the endpoints transparent: if innovations vanish, only the weighted initial fluctuation remains. The recurrence a_(m+1)=1+ρ a_m gives all prefix values in a single pass. For ρ in [−1,1], each a_m lies in [0,m], a fact used to tighten directed numerical bounds.

This is the same variance as the covariance double sum, not a new approximation. The two expressions offer independent routes for verification. A numerical implementation should not use the same positive-contribution formula as its only reference calculation and then call the agreement independent evidence.

## Work through small positive and negative examples

For n=2 the exact result is particularly simple:

$$
V_2=\frac{1+\rho}{2}.
$$

A positive half correlation gives variance 3/4; a negative half correlation gives variance 1/4. The independent value is 1/2. The two signs therefore move uncertainty in opposite directions even though the magnitude of the lag-one correlation is identical.

For four observations, retaining all lag weights gives the following ideal rational examples:

| ρ | Exact V4 | Variance inflation 4V4 | Variance-equivalent count 1/V4 | Asymptotic τ/4 |
| --- | --- | --- | --- | --- |
| 0 | 1/4 | 1 | 4 | 1/4 |
| 1/2 | 33/64 | 33/16 | 64/33 | 3/4 |
| −1/2 | 7/64 | 7/16 | 64/7 | 1/12 |

For the negative case, the numerator is 4+2[3(−1/2)+2(1/4)−1/8]=7/4. Dividing by sixteen gives 7/64. Dropping the lag-two or lag-three terms changes the result. The mean uses all four values, and its uncertainty reflects all six distinct off-diagonal pairs.

## Distinguish variance from standard error and physical scale

The standard error is a square root, not the variance itself:

$$
\operatorname{SE}(\overline X_n)=\sqrt{V_n},\qquad \operatorname{Var}\!\left(\sum_iX_i\right)=n^2V_n,\qquad \operatorname{Var}(\mu+\sigma\overline Y_n)=\sigma^2V_n.
$$

For the negative half-correlation example with n=4, the variance is 7/64 and the standard error is √7/8. If observations are measured in meters, variance has squared-meter units and standard error has meter units. Scaling the marginal standard deviation by σ scales standard error by its magnitude and variance by σ squared.

Under this Gaussian preparation, the finite mean error is exactly Gaussian with variance V; normality does not require an asymptotic central limit theorem here. In a non-Gaussian process with the same covariance, the same finite variance formula still holds, but the distribution of the mean is not thereby determined. Uncertainty of a nonlinear estimator is another question again.

## Take the large-sample limit with its conditions intact

For a fixed |ρ|<1, the absolutely summable normalized autocovariances define the convention

$$
\tau=1+2\sum_{k=1}^{\infty}\rho^k=\frac{1+\rho}{1-\rho},\qquad nV_n\longrightarrow\tau,\qquad V_n\sim\tau/n.
$$

Some conventions put a factor one-half in an integrated correlation time. This chapter uses the explicit sum above, so independent observations have τ=1. For negative ρ, τ can be less than one because alternating terms cancel. It is a variance inflation factor for this observable, not the number of steps over which every form of dependence disappears.

The exact interior correction is

$$
V_n=\frac{\tau}{n}-\frac{2\rho(1-\rho^n)}{n^2(1-\rho)^2}\qquad(|\rho|<1).
$$

For positive ρ, the finite variance is below τ/n; for negative ρ it is above τ/n. At n=2 and ρ=1/2, τ/n=3/2 even exceeds the marginal variance one, while the exact finite variance is 3/4. The approximation is plainly inadequate for that short preparation.

The error is order n to the power −2 at fixed ρ, with a coefficient that is not uniformly bounded as ρ approaches one. Near either nonmixing endpoint, a finite n can be much shorter than the relevant decay scale. The statement does not justify replacing a finite error by τ/n while simultaneously moving ρ toward an endpoint without controlling that joint limit.

## Give a variance-equivalent count its precise meaning

For a marginal variance of one, define

$$
n_{\mathrm{eq}}=\frac{1}{V_n},\qquad R_n=nV_n,\qquad n_{\mathrm{eq}}=n/R_n.
$$

This count says how many independent unit-variance observations would give the same variance for their linear mean. It need not be an integer and can exceed the actual n. At ρ=−1/2 and n=4, it is 64/7, approximately 9.14. That does not create nine independent stored values or prove that every estimator receives nine observations' information.

The count is specific to the observable, the preparation and the reference marginal variance. If a downstream model uses squares, thresholds or another nonlinear function, its autocovariance can differ. Estimating an unknown ρ from data also adds a diagnostic problem not included in this specified-parameter calculation. A label such as effective sample size should always state the variance comparison it represents.

## Treat constant and alternating endpoints separately

At the two endpoints the innovation variance is zero. The process remains stationary under the specified symmetric initial Gaussian law, but is not mixing. Direct substitution into the path gives

$$
\rho=1:\quad Y_i=Y_1,\quad V_n=1;\qquad \rho=-1:\quad Y_i=(-1)^{i-1}Y_1,\quad V_n=\begin{cases}0,&n\text{ even},\\1/n^2,&n\text{ odd}.\end{cases}
$$

A constant path never reduces its initial fluctuation by averaging. In an alternating path an even-length centered average cancels exactly; an odd-length average retains Y1/n. Thus the variance-equivalent count is infinite for even n and n squared for odd n. The model labels these as endpoint properties, not ordinary mixing-time estimates.

At ρ=−1, substituting into the rational continuation of τ would give zero, but the ordinary infinite autocovariance series alternates without converging. The model therefore reports no ordinary summed IAT at either endpoint. A finite exact cancellation should not be presented as a valid nondegenerate mixing-CLT diagnostic.

The limitation is visible without advanced theory. Along the alternating path, every squared centered observation equals Y1 squared. Their average remains random even when the linear mean cancels exactly. Also, adding one observation can increase finite mean variance: n=2 gives zero, n=3 gives 1/9, and n=4 gives zero again. “More observations always lower this variance” is false for this dependence pattern.

## Connect the calculation to AI averages carefully

A mini-batch gradient is an average of vectors, so its covariance includes cross-observation covariance matrices. The scalar AR(1) example can represent one specified gradient component or one observable of a Monte Carlo chain. Positive serial dependence can make a nominal batch much less effective for that average; negative dependence can reduce its variance through cancellation.

This does not make an arbitrary training stream stationary. Parameter updates, changing data distributions, shuffled batches and adaptive sampling can change the relevant mean and covariance. The AR(1) coefficient in this chapter is a model input, not a correlation estimate extracted from such a stream. In a real analysis, identify the observable and preparation, examine whether a stationary model is appropriate, and keep parameter estimation distinct from the exact calculation under an assumed parameter.

Spatial covariance among coordinates inside one sample also differs from autocovariance across samples. The previous ring's distance and correlation length cannot be inserted as n or ρ without defining a new observation process. Conversely, a long sampler history does not tell you the lattice's spatial correlation length by itself.

## Keep numerical evidence separate from a plotted path

The model treats the supplied binary64 ρ as an exact input and computes 1−ρ squared with exact binary arithmetic. Directed arithmetic propagates bounds through the positive innovation decomposition. Prefix mean variances are calculated for every count up to n. Autocovariance powers retain exact dyadic coefficients and exponents, so a tiny nonzero distant lag is not silently converted into a structural zero.

Standard-error bounds use integer square roots rounded outward; they are not certified by calling an ordinary floating square root on a displayed variance. Resolved nonzero observations have a sign and relative enclosure width at most 2 to the power −48. Working precision can increase from 128 through 8192 bits. If a ceiling leaves a value unresolved, the model reports it unavailable. General magnitude-log projections are approximate and are not a replacement for bounds.

The independent reference uses Python Fraction on the actual binary64 parameter and the finite covariance lag sum. Small preparations are additionally checked by the direct covariance double sum. Decimal projections and square roots are independently recomputed at higher precision. The innovation decomposition is not reused as the reference. These checks concern the numerical model and manuscript. The registered reader has separate teaching controls; worker behavior, saved-state recovery and actual reading interactions need integration evidence.

A short randomly generated path would be an illustration, not proof of the theoretical variance. Independent repeated preparations would be needed to estimate that variance empirically, with their own sampling error. This chapter's calculations do not infer exact uncertainty from one visually smooth or alternating trace.

## Reproduce a four-observation calculation exactly

This small Python example computes the covariance quadratic form using exact rational arithmetic:

```python
from fractions import Fraction

n, rho = 4, Fraction(-1, 2)
variance = sum(rho ** abs(i - j)
               for i in range(n) for j in range(n)) / n**2
print(variance)       # 7/64
print(1 / variance)   # 64/7
```

Change the sign of ρ and recover 33/64. Then use ρ=−1 and compare even and odd n. The code evaluates the specified covariance model; it does not estimate an autocovariance from simulated data. To verify the implementation independently, this double sum is especially useful at small n, while the lag-counted form avoids constructing a large matrix.

The visualization distinguishes a covariance window, the finite mean-variance curve and the optional τ/n reference. Controls for n and signed ρ retain endpoint labels. A line chart of V against prefix length may oscillate for negative dependence; smoothing it into a monotone curve would change the result. The table retains all prefixes even when the covariance picture shows only its first window.

## Connect finite uncertainty to the experiment

The covariance window shows at most the first8×8 entries; the calculation uses all n observations. The blue curve reports exact finite-prefix mean variances, and the orange dashed curve is the optional asymptotic tau/n reference. These are theoretical uncertainties under a specified stationary preparation, not errors estimated from a generated path.

| Preparation to compare | Predict before changing it | Read in the result |
| --- | --- | --- |
| Four independent → positive half → negative half correlation | Does equal correlation magnitude imply equal uncertainty? | Compare variances1/4,33/64,7/64 and distinguish variance from its square-root standard error. |
| Positive half, n=2 → n=64 | Is tau/n already accurate for a short average? | At n=2 the exact variance3/4 differs from the asymptotic3/2; compare the finite/asymptotic ratio as n grows. |
| Alternating endpoint, even4 → odd5 | Must adding an observation reduce the variance? | The exact zero becomes1/25; ordinary summed IAT stays unavailable at the endpoint. |

Commit integer n and rho with Enter or blur. Use Log variance axis to inspect scale differences; exact zero dots sit below that axis and are not connected as positive values. Show asymptotic reference is disabled at nonmixing endpoints because no ordinary summed IAT is provided. The two paginated tables retain all lags and every prefix variance. A variance-equivalent count above n describes this linear mean only. It does not create independent observations or establish stationarity in an arbitrary training stream.

## Research extension: use known correlation to choose the mean's weights

The ordinary sample mean gives every observation the same weight. That is convenient and unbiased under the stationary preparation, but need not minimize variance when the covariance is known. Correlated neighbors partly repeat one another; an endpoint has fewer neighbors within the observed window. Generalized least squares turns that boundary difference into explicit weights.

### Optimize a specified estimator under a specified covariance

Retain X=µ1+Y, unit marginal variance, the stationary initial law and a known |ρ|<1. For n≥2, let Σᵢⱼ=ρ^|i−j| and Q=Σ⁻¹. Minimize the variance of a linear estimator wᵀX subject to weights summing to one. A Lagrange multiplier gives

$$
\widehat\mu_w=w^{\mathsf T}X,\quad \mathbf1^{\mathsf T}w=1,\qquad w_{\mathrm{GLS}}=\frac{Q\mathbf1}{\mathbf1^{\mathsf T}Q\mathbf1},\qquad V_{\mathrm{GLS}}=\frac1{\mathbf1^{\mathsf T}Q\mathbf1}.
$$

This is a different estimator from the ordinary mean already calculated above. It does not revise that estimator's exact variance V_n. With n=1 the only unbiased weight is one and the variance remains one; treat that case separately.

To find Q, write the Gaussian joint density as the stationary initial density times the innovation conditionals. Up to a constant independent of µ its negative twice-log density is

$$
(X-\mu\mathbf1)^{\mathsf T}Q(X-\mu\mathbf1)=Y_1^2+\frac1{1-\rho^2}\sum_{i=1}^{n-1}(Y_{i+1}-\rho Y_i)^2.
$$

Expanding the squares reveals a tridiagonal precision matrix. The endpoints occur in fewer innovation terms and therefore differ from the interior:

$$
Q_{ij}=\frac1{1-\rho^2}\begin{cases}1,&i=j\in\{1,n\},\\1+\rho^2,&1<i=j<n,\\-\rho,&|i-j|=1,\\0,&\text{otherwise}.\end{cases}
$$

Multiplying Q by the all-ones vector now needs only row sums. The endpoint weights and each interior weight are

$$
D_n=n(1-\rho)+2\rho,\qquad w_1=w_n=\frac1{D_n},\qquad w_i=\frac{1-\rho}{D_n}\quad(1<i<n).
$$

For positive ρ, endpoints get more weight than interior observations because their within-window redundancy is smaller. For negative ρ, the interior can exploit two-sided cancellation and gets more weight. All these weights remain positive and sum to one for |ρ|<1. For n=2 both weights are one-half, whatever ρ is.

### Prove the gain and distinguish exact from asymptotic claims

Summing the precision row sums gives an exact finite variance for this known-covariance estimator:

$$
\mathbf1^{\mathsf T}Q\mathbf1=\frac{D_n}{1+\rho},\qquad V_{\mathrm{GLS}}=\frac{1+\rho}{D_n}.
$$

The variance gain does not rely on a numerical optimizer. For any other unbiased weights, write w=w_GLS+δ with 1ᵀδ=0. Since Σw_GLS=V_GLS 1, the cross term cancels:

$$
\operatorname{Var}(w^{\mathsf T}X)-V_{\mathrm{GLS}}=(w-w_{\mathrm{GLS}})^{\mathsf T}\Sigma(w-w_{\mathrm{GLS}})\geq0.
$$

Positive definite Σ makes equality possible only for the same weights. The ordinary equal weights therefore attain equality at ρ=0 and for n=2, but generally lose some efficiency for n>2. At fixed |ρ|<1 both estimators still have the same leading large-n variance τ/n; their boundary corrections differ. A finite gain is compatible with the same leading asymptotic rate.

Under the Gaussian model with known covariance, GLS is also the exact maximum-likelihood estimator of µ. Its score, information and distribution give a stronger finite-sample statement than a generic large-sample argument:

$$
s_\mu=\mathbf1^{\mathsf T}Q(X-\mu\mathbf1),\qquad \mathcal I_\mu=\mathbf1^{\mathsf T}Q\mathbf1,\qquad \widehat\mu_{\mathrm{GLS}}\sim\mathcal N(\mu,\mathcal I_\mu^{-1}).
$$

The estimator is unbiased and attains the scalar information bound here. With non-Gaussian errors of the same mean and covariance it remains the best linear unbiased estimator, but this Gaussian distribution and likelihood interpretation need not hold. If ρ is estimated from the same data and plugged into the weights, these exact known-ρ statements no longer follow automatically.

### Keep singular endpoints outside the inverse-covariance proof

For fixed n≥2, the interior formulas have the following continuous limits:

$$
\lim_{\rho\to1^-}V_{\mathrm{GLS}}=1,\qquad \lim_{\rho\to-1^+}V_{\mathrm{GLS}}=0,\qquad \lim_{\rho\to-1^+}w=\frac{(1,2,\ldots,2,1)^{\mathsf T}}{2(n-1)}.
$$

At the exact endpoints Q does not exist. Check the paths directly instead: at ρ=1 all Xᵢ equal µ+Y₁, so no unbiased linear weighting removes that common error. At ρ=−1, (X₁+X₂)/2=µ exactly for n≥2; the displayed limiting weights also cancel the alternating fluctuation, including for odd n. This contrasts with the ordinary odd-length mean's variance 1/n². It says nothing about the precision of averages of squares or an arbitrary downstream estimator.

![Boundary-sensitive GLS weights and the finite variance ratio between ordinary and efficient means under a stationary AR(1) preparation.](/assets/learning/correlations-research-ar1-efficient-mean.en.svg)

The upper panel compares the weights at n=9. The lower panel shows the ordinary/GLS variance ratio for two signed coefficients; the odd-even behavior comes from the ordinary mean. The covariance and mean model are unchanged. Only the estimator changes, and the existing reader continues to report the ordinary mean.

The table reports exact-model finite variances rounded to six decimals. The code uses rational arithmetic for both covariance quadratic forms. The final column is one endpoint's weight, not the sum of both endpoints. No observations or coefficient estimates are generated.

| n | ρ | V_mean | V_GLS | Variance ratio | w_end |
| --- | --- | --- | --- | --- | --- |
| 3 | -0.8 | 0.120000 | 0.052632 | 2.280000 | 0.263158 |
| 4 | -0.8 | 0.046000 | 0.035714 | 1.288000 | 0.178571 |
| 9 | 0.8 | 0.572453 | 0.529412 | 1.081300 | 0.294118 |
| 16 | 0.8 | 0.410648 | 0.375000 | 1.095061 | 0.208333 |

```python
from fractions import Fraction

for n, rho in ((3,Fraction(-4,5)), (4,Fraction(-4,5)),
               (9,Fraction(4,5)), (16,Fraction(4,5))):
    ordinary = sum(rho**abs(i-j) for i in range(n)
                   for j in range(n))/n**2
    den = n*(1-rho)+2*rho
    endpoint = 1/den
    weights = [endpoint]+[(1-rho)/den]*(n-2)+[endpoint]
    efficient = sum(weights[i]*weights[j]*rho**abs(i-j)
                    for i in range(n) for j in range(n))
    assert efficient == (1+rho)/den
    print(n, f'{float(rho):.1f}', f'{float(ordinary):.6f}',
          f'{float(efficient):.6f}', f'{float(ordinary/efficient):.6f}',
          f'{float(endpoint):.6f}')
```

### Research checks and boundaries

#### Why are endpoint weights different without nonstationary noise?

**Research explanation.** The process is stationary, but the observed window has boundaries. Each interior reading has neighbors on both sides inside that window; an endpoint does not. The estimator responds to the finite observation geometry, not a change in marginal variance.

#### Does GLS create more independent observations?

**Research explanation.** No. It changes which linear combination estimates the common mean. The same dependent vector X is observed. Variance comparisons remain estimator- and model-specific, and cannot be transferred to every statistic or training stream.

#### Can the interior precision formula be substituted at ρ=−1?

**Research explanation.** No. Its denominator vanishes and Σ is singular. The exact alternating path provides a separate cancellation proof. A continuous limit of variances does not authorize taking the inverse of a singular covariance.

[Shalizi's GLS notes, sections 6–6.1](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf) describe inverse-covariance weighting. The finite AR(1) precision, boundary weights, variance gap and endpoint comparison above are derived here for the stationary unit-variance preparation. They do not infer stationarity or known covariance from a visually smooth path.

## Check your understanding with explained problems

### 1. Count lag pairs

For n=4, how many covariance terms occur at lags one, two and three in the full double sum?

**Explanation.** Above the diagonal there are three, two and one pairs respectively. Reflection supplies the same counts below it, giving six, four and two terms. The four diagonal terms are separate. This produces 4+2(3ρ+2ρ squared+ρ cubed), divided by sixteen.

### 2. Separate exact finite variance from its approximation

At ρ=1/2 and n=2, compare the exact variance with τ/n. Which should be used to describe this specified finite preparation?

**Explanation.** The exact value is (1+1/2)/2=3/4. The interior τ is 3, so τ/n=3/2. Use 3/4 for the finite preparation. The larger approximation is not a contradiction; its large-n conditions have not been reached.

### 3. Keep square-root units

At ρ=−1/2 and n=4, give variance, standard error and variance-equivalent count.

**Explanation.** They are 7/64, √7/8 and 64/7 respectively. The first has squared observation units, the second has observation units and the third is dimensionless. Reporting variance as standard error would underestimate the uncertainty in the original units.

### 4. Interpret perfect antithetic cancellation

At ρ=−1 and n=2, the linear mean error is zero. Does this make the mean of squared centered observations equally certain?

**Explanation.** No. The centered pair is Y1,−Y1, so its linear mean vanishes while both squares equal Y1 squared. Their average is still random. The infinite variance-equivalent count describes only the specified linear mean; it is not a universal information count.

### 5. Identify a preparation mismatch

An implementation starts from Y1=0 with |ρ|<1 and claims the stationary covariance matrix from the first observation onward. What assumption changed?

**Explanation.** The initial variance is now zero instead of one. Later variance builds through innovations and depends on the starting index, so the finite early covariance is not the stationary matrix ρ to the power |i−j|. Either prepare the stationary initial law or derive and use the transient covariance.

### 6. Audit a negative-dependence sample-count rule

A dashboard forces uncertainty to decrease every time n grows. Is that valid for the alternating endpoint, and should it show a usual summed IAT of zero?

**Explanation.** Neither rule is valid. Finite variances at n=2,3,4 are 0,1/9,0, so a monotone smoothing rule changes the mathematics. The ordinary infinite autocovariance series does not converge at ρ=−1. Keep the exact endpoint cancellation and unavailable ordinary IAT as separate facts.

## Continue toward source-specific interactive experiments

The three families now have complementary roles: the spin pair explains source Hessians, the ring introduces spatial distance and boundary corrections, and the AR(1) preparation explains uncertainty across observation indices. Covariance is common to them, but each uses a different observable, source, graph or index. Those specifications must remain visible when combining the lessons.

For further research, derive transient-start covariance, study observable-dependent autocovariances and compare an empirically estimated asymptotic variance with this specified-model finite answer. Multivariate or nonstationary extensions need additional models and evidence. The three registered Interactive Book readers connect fixed concept steps, visualizations, guided experiments, explained quizzes and cancellable workers. Their integration requires actual reader checks; standalone calculations alone do not complete the remaining curriculum or the research extensions.
