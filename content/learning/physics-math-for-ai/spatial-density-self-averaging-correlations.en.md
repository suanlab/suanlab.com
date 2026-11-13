## More sites do not always mean less uncertainty

Imagine a camera observing a large field of tiny binary pixels. Each pixel is bright with probability p. If the pixels fluctuate independently, averaging many of them stabilizes the overall brightness. But suppose the whole field sometimes receives one shared signal: every pixel then copies the same binary value. Adding pixels cannot average away a signal that they all share.

The brightness probability of a single pixel does not distinguish these preparations. A histogram pooled over individual pixels can therefore conceal a large difference in the variability of the field average. To understand that difference, we need a joint probability law, a covariance calculation and a specified way of increasing the system size.

This chapter defines a small exact spatial model. Its purpose is to separate equal marginals, independence, self-averaging and rare-event behavior. The prerequisites are Bernoulli variables, expectations, variance and finite sums. No physical temperature, equilibrium interaction or time evolution is assumed.

## Specify one choice for the whole system

Let N be the number of sites. At each size, choose a mode once for the entire system. With probability 1−cN, generate N independent Bernoulli(p) variables. With probability cN, generate one Bernoulli(p) variable Y and copy it to every site. Here cN is between zero and one.

The following sketch shows possible outcomes, rather than equal-probability configurations. A filled square denotes one and an empty square denotes zero. The two collective outcomes have probabilities p and 1−p conditional on choosing that mode.

```text
                 ONE MODE CHOICE FOR THE WHOLE FIELD
                                  |
                    +-------------+-------------+
                    |                           |
             independent mode               collective mode
                weight 1-cN                    weight cN
                    |                           |
           [1] [0] [1] [0]             [Y] [Y] [Y] [Y]
           [0] [1] [0] [0]             [Y] [Y] [Y] [Y]
         separate Bernoulli draws          one draw copied
```

Choosing a separate mode at each site would define a different joint distribution. Also, the sketch does not assign geometric distances to the sites. Every distinct pair has the same covariance in this preparation; it is not a model of correlations decaying with distance.

## Name the observables and the controls

Write the total occupied count as K and the density as ρ. Both the mean and the fluctuation of density concern a randomly prepared spatial field at a fixed size. A size record compares different preparations; it is not a sequence of frames from one evolving field.

$$
K=\sum_{i=1}^N X_i,\qquad \rho=\frac KN,\qquad p=\frac{A}{A+B},\qquad v=p(1-p).
$$

| Symbol | Meaning | Convention |
| --- | --- | --- |
| N | Number of spatial sites | Integer from 1 to 512 in the calculator |
| A, B | Occupied and empty weights | Nonnegative integers with positive sum |
| p | Occupancy probability at one site | Fixed along a size path |
| c | Base collective-mode weight | Between zero and one |
| cN | Collective-mode weight at the current size | c, c/N or c/N² |
| ε | Density deviation threshold | Positive rational, including values above one |
| K, ρ | Total count and density | Random variables of the specified preparation |

The calculator uses integer weights up to one million and positive denominators up to one million. It evaluates finite laws exactly with rational arithmetic. These limits bound the interactive calculation, not the mathematical definition of a size family for all positive integers N.

## Equal marginals do not specify the joint law

In the independent mode, E[Xi]=p. In the collective mode, Xi=Y and the mean is again p. Mixing these modes leaves the single-site law unchanged:

$$
\mathbb E[X_i]=p,\qquad \operatorname{Var}(X_i)=v,\qquad \mathbb E[\rho]=p.
$$

For two different sites, the independent mode gives E[XiXj]=p², whereas the collective mode gives p. Subtracting the product of the means yields

$$
\mathbb E[X_iX_j]=(1-c_N)p^2+c_Np,\qquad
\operatorname{Cov}(X_i,X_j)=c_Nv\quad(i\ne j).
$$

When 0<p<1, the normalized pair correlation is cN. At p=0 or p=1, covariance is zero but the correlation coefficient is undefined because its denominator is zero. At N=1 there is no distinct pair at all. Neither missing quantity should be displayed as a measured zero correlation.

## Sum the diagonal and the ordered pairs

The variance of a sum contains N diagonal terms and N(N−1) ordered off-diagonal terms. Each unordered pair occurs twice. Consequently,

$$
\operatorname{Var}(K)=Nv+N(N-1)c_Nv,\qquad
\operatorname{Var}(\rho)=\frac vN[1+(N-1)c_N].
$$

The independent contribution scales as N in the variance of K. With a fixed nonzero collective weight, the off-diagonal sum scales as N². Dividing by N² then leaves a nonzero density variance. A small covariance per pair can matter when many pairs contribute.

Take N=4, p=1/2 and cN=1/2. There are four diagonal terms of size1/4 and twelve ordered off-diagonal terms of size1/8. Their sums are1 and3/2. Thus Var(K)=5/2 and Var(ρ)=5/32. The independent density variance at the same p and N is1/16. The mean density remains1/2 in both cases.

## Recover the whole finite count law

In the independent mode, K is binomial. In the collective mode, only K=0 and K=N can occur. The exact count probability is therefore

$$
 m_k=(1-c_N)\binom Nk p^k(1-p)^{N-k}
 +c_N[(1-p)\mathbf1_{k=0}+p\mathbf1_{k=N}],\qquad 0\le k\le N.
$$

At N=4 and p=cN=1/2, the distribution gives this ledger. The last column uses ε=1/2, including equality.

| k | IID probability | Collective probability | Mixed probability | In deviation event |
| --- | --- | --- | --- | --- |
| 0 | 1/16 | 1/2 | 9/32 | yes |
| 1 | 1/4 | 0 | 1/8 | no |
| 2 | 3/8 | 0 | 3/16 | no |
| 3 | 1/4 | 0 | 1/8 | no |
| 4 | 1/16 | 1/2 | 9/32 | yes |

Sum the mixed column to obtain one. Multiplying each row by k/4 gives mean1/2; multiplying by (k/4−1/2)² gives variance5/32. These sums independently check the moment formula. The probability of a count already includes all named configurations having that count. Do not multiply it by another binomial coefficient.

## Define the deviation event before computing it

A tail probability must identify its observable, center and boundary convention. We use a fixed positive ε and include equality:

$$
T_N=\mathbb P(|\rho-p|\ge\epsilon),\qquad
|k(A+B)-NA|V\ge EN(A+B),\quad\epsilon=E/V.
$$

The integer comparison on the right is the exact event test. At N=3, p=1/3 and ε=1/3, the event counts are0,2,3. Replacing ε by333334/1000000 leaves only3. Floating-point comparison near equality could silently change the question.

If ε>1, the event is empty, even though a variance bound may remain positive. A count can also belong to the event while having zero preparation probability. For example, deterministic occupancy at p=0 puts all mass at k=0; other lattice counts do not become possible just because they satisfy a deviation inequality.

## Keep probability, reference and bound separate

Let TI be the IID event probability and TS the collective-mode event probability. TS comes from the two endpoints and does not depend on N when p and ε are fixed:

$$
T_S=(1-p)\mathbf1_{p\ge\epsilon}+p\mathbf1_{1-p\ge\epsilon},\qquad
T_N=(1-c_N)T_I+c_NT_S.
$$

Chebyshev gives an IID reference bound and a bound for the actual mixture:

$$
B_I=\min\{1,v/(N\epsilon^2)\},\qquad
B_C=\min\{1,\operatorname{Var}(\rho)/\epsilon^2\}.
$$

Using the known collective endpoint probability gives a third quantity:

$$
B_M=(1-c_N)B_I+c_NT_S,\qquad T_N\le B_M\le B_C\le1.
$$

To see the second inequality, first bound TS by min(1,v/ε²), then use concavity of min(1,x). BI alone is a guarantee for the IID component, not for the mixture. For the four-site ledger, TI=1/8, TS=1, TN=9/16 and BM=BC=5/8. The IID reference BI=1/4 is below the actual probability. Reusing it as a mixture bound would be wrong.

## More variance need not increase this tail

Variance summarizes a weighted average of squared deviations. One event threshold records only whether a deviation crosses that threshold. These comparisons need not order two distributions in the same way.

For N=2, p=1/10, cN=1/2 and ε=3/20, the event is k=1 or2. In the IID mode its probability is19/100. In the collective mode only the all-one outcome qualifies, with probability1/10. Thus the mixture has

$$
T_N=\frac{29}{200}<\frac{19}{100}=T_I,\qquad
\operatorname{Var}(\rho)=\frac{27}{400}>\frac9{200}=\operatorname{Var}_{I}(\rho).
$$

The collective mode moves mass to both endpoints. The more likely zero endpoint lies only1/10 from p and fails this threshold. The less likely one endpoint is far away and contributes strongly to squared deviation. This is how the variance can increase while the chosen tail probability falls. It does not contradict Chebyshev, which is an upper bound rather than an equality.

## State the size path before taking a limit

Keep p, c and ε fixed and compare three prescriptions for cN. For 0<p<1, the limiting behavior is

| Path | Limit of Var(ρ) | Limit of N Var(ρ) | Limit of variance-equivalent count/N |
| --- | --- | --- | --- |
| cN=c with c>0 | cv | +∞ | 0 |
| cN=c/N | 0 | v(1+c) | 1/(1+c) |
| cN=c/N² | 0 | v | 1 |

At c=0 all three paths reduce to IID. At p=0 or1 the density is deterministic, so every variance is zero and the variance-equivalent count is left undefined.

For the vanishing-weight paths, the finite corrections follow directly from the covariance sum:

$$
N\operatorname{Var}(\rho)=v(1+c-c/N)\quad(c_N=c/N),\qquad
N\operatorname{Var}(\rho)=v(1+c/N-c/N^2)\quad(c_N=c/N^2).
$$

At N=4,p=c=1/2, these paths give density variances11/128 and35/512. They are finite-size values, not the limiting coefficients3/8 and1/4 in the table. A plot of variance alone tends toward zero for both paths; plotting N times the variance reveals their different leading coefficients.

## Explain what self-averaging means here

Here self-averaging means that the density approaches its fixed mean p in probability along the specified size family. Since the density stays in[0,1], vanishing variance is both sufficient and necessary for this form of convergence:

$$
\mathbb P(|\rho-p|\ge\epsilon)\le\operatorname{Var}(\rho)/\epsilon^2,\qquad
\mathbb E[(\rho-p)^2]\le\epsilon^2+\mathbb P(|\rho-p|>\epsilon).
$$

Use the first inequality when the variance vanishes. For the reverse direction, take a limit in the second inequality for each ε, then let ε decrease to zero. Boundedness is doing real work in that argument.

For fixed c>0 and 0<p<1, the weak limiting density law is

$$
\rho\ \Rightarrow\ c(1-p)\delta_0+(1-c)\delta_p+cp\delta_1.
$$

For either vanishing-weight path it is simply δp. Zero-mass atoms are removed and coincident atoms are merged. These atoms describe a limit law, not a finite-size histogram. We have not specified a coupling of fields at different sizes, so this calculation does not assert almost-sure convergence across N.

The exact mixture identity also gives TN→c∞TS. This event limit is derived separately from the IID tail tending to zero, not inferred just from weak convergence at a boundary atom. Equality at ε=p or ε=1−p remains included.

## Use a variance-equivalent count carefully

For 0<p<1, define the number of independent sites giving the same variance for this density observable:

$$
N_{\rm var}=\frac{v}{\operatorname{Var}(\rho)}
=\frac{N}{1+(N-1)c_N}.
$$

It equals8/5 in the four-site ledger. At fixed c>0 it tends to1/c, even though the actual number of sites diverges. With cN=c/N, its fraction of N tends to1/(1+c); with cN=c/N², that fraction tends to one.

This count answers a variance-matching question for one known preparation and one observable. It does not certify the number of independent training examples, the information content of a dataset, or the effective sample size of a time MCMC trajectory. At deterministic p, the defining variance ratio is0/0 and is not supplied. An algebraic expression that remains finite after cancellation does not restore that interpretation.

## Equal leading variance can conceal very different rare events

Take p=10^−6, c=10^−6, cN=c/N² and ε=1−p. Only the all-one field qualifies. Then

$$
T_I=p^N,\qquad T_N=(1-c/N^2)p^N+cp/N^2.
$$

At N=512 the IID probability is10^−3072. The collective contribution is about3.8×10^−18. The exact mixed-to-IID ratio is finite but of order10^3054. A floating-point display may be unable to represent the small IID probability or the large ratio. Their exact rational representations still exist. Display unavailability is neither a zero probability nor mathematical infinity.

Both preparations have limiting N Var(ρ)=v. Nevertheless, the indicated event has polynomial rather than exponential asymptotics in the mixed family. Extending this same formula to all sizes gives −log(TN)/N→0, whereas the IID value is−log(p). This is a calculation for this particular event and family, not a general large-deviation principle for arbitrary correlated systems.

## Do not attach an unverified response identity

The input p changes both preparation modes. If we write g=logit(p), the response of the count mean to that input is

$$
\frac{d\mathbb E[K]}{dg}=Nv,\qquad
\operatorname{Var}(K)=Nv[1+(N-1)c_N].
$$

These differ when distinct sites are correlated and v>0. This does not refute the response identity for a correctly specified exponential family. If we instead define a new tilt exp(tK) of a fixed baseline law, its mean derivative at t=0 is the baseline variance of K. That is a different operation from the current p control. A fluctuation-response statement requires identifying the measure and the field being differentiated.

A useful AI analogy is a shared environmental disturbance affecting many inputs. It motivates checking dependence rather than counting observations alone. It does not turn this toy spatial law into a calibrated dataset model or an AI performance guarantee.

## Connect collective modes and rare events to the experiment

In the default N=4,p=1/2,c=1/2 preparation, the whole system can receive one shared signal. Density variance5/32 exceeds the IID value1/16. Before counting four pixels as four independent observations, ask whether they shared the same environmental disturbance.

| Action | Observation to inspect | Question to explain |
| --- | --- | --- |
| Compare the default with ‘Same marginals, independent preparation’ | Mean density remains1/2; variances are5/32 and1/16 | Does an identical single-pixel histogram imply identical uncertainty of the field average? |
| Move the selected count from K=0 through4 | At ε=1/2, only0 and4 belong to the event; total mixture tail is9/16 | How do grid membership, probability at that grid point and total event probability differ? |
| Enable ‘Show N-scaled variance’ on c/N and c/N² paths | Both variances vanish, but their N-scaled limits are3/8 and1/4 | Do two small variance curves necessarily have the same size scaling? |

The count slider selects a point of the current law; it does not generate a random field. ‘Log probability display’ reveals rare mass without turning exact zero into a positive probability. Size controls increase N at the same p,c,epsilon and path. Variance-equivalent count concerns this density observable alone. An undisplayable tiny IID tail at large N is neither zero probability nor an infinite tail ratio.

## Research extension: a Gaussian weak limit can miss the variance

The previous example showed that a small collective weight can dominate a rare event. A second surprise appears in central-limit scaling: most fields can have an ordinary Gaussian central shape while rare collective fields carry a finite part of the variance. Distributional convergence does not automatically justify replacing second moments by those of the limit.

### Scale the centered density and compare its probability law

Fix 0<p<1 and v=p(1−p)>0, with collective weight c_N→0. Let W_N be the usual square-root-size fluctuation and W_N^I its independent-mode counterpart:

$$
W_N=\sqrt N(\rho-p),\qquad \mathcal L(W_N)=(1-c_N)\mathcal L(W_N^I)+c_N\big[(1-p)\delta_{-p\sqrt N}+p\delta_{(1-p)\sqrt N}\big].
$$

The two laws differ only on the collective-mode choice, so their TV distance is at most c_N. The IID Bernoulli central limit theorem then gives a weak limit for every vanishing collective weight, regardless of how quickly it vanishes:

$$
\|\mathcal L(W_N)-\mathcal L(W_N^I)\|_{\mathrm{TV}}\leq c_N\longrightarrow0,\qquad W_N\Rightarrow\mathcal N(0,v).
$$

This does not claim TV convergence to a continuous Gaussian: each finite W_N is discrete. The TV comparison is between the two discrete preparations at the same size. Weak convergence compares bounded continuous test functions and can lose a very small probability moving very far into the tails.

The second moment sees that distant probability. Using the original covariance formula,

$$
\mathbb E[W_N^2]=v[1+(N-1)c_N],\qquad \mathbb E[W_N^2\mathbf1_{\mathrm{collective}}]=Nc_Nv.
$$

If c_N=c/N with c>0, the weak Gaussian has variance v but the finite second moments tend to v(1+c). If c_N=c/N^α with 0<α<1, they diverge despite exactly the same Gaussian weak limit. Density itself still self-averages for each α>0 because its unscaled variance tends to zero.

### Identify the missing condition: uniform integrability

A standard way to control passage of second moments is uniform integrability of the squared fluctuations:

$$
\lim_{R\to\infty}\sup_N\mathbb E[W_N^2\mathbf1_{|W_N|>R}]=0.
$$

It requires the second-moment contribution outside every sufficiently large fixed window to become uniformly small. For N large enough that both collective endpoints lie outside that window,

$$
\mathbb E[W_N^2\mathbf1_{|W_N|>R}]\geq Nc_Nv\qquad\bigl(R<\sqrt N\min(p,1-p)\bigr).
$$

At c_N=c/N, that lower bound is cv for arbitrarily large windows. The condition fails. The rare event's probability goes to zero, but its squared magnitude grows like N and compensates. This is a specific mechanism, not a failure of the central limit theorem or an assertion that all weak limits lose moments.

### Standardizing by the actual variance need not repair the limit

Let T_N divide the centered density by its exact finite standard deviation. It has mean zero and variance one at every finite N. For c_N=c/N^α with fixed 0<c≤1,

$$
T_N=\frac{\rho-p}{\sqrt{\operatorname{Var}(\rho)}},\qquad T_N\Rightarrow\begin{cases}\mathcal N(0,1),&\alpha>1,\\\mathcal N(0,(1+c)^{-1}),&\alpha=1,\\\delta_0,&0<\alpha<1.\end{cases}
$$

To derive this, divide the IID central fluctuation by sqrt(N Var(ρ)), then use its deterministic scale limit; the collective part still has vanishing probability. For α<1 the bulk shrinks to zero in these coordinates, while outliers carry the unit variance. The α=1 limiting Gaussian has variance below one. Neither conclusion contradicts finite Var(T_N)=1 because squared fluctuations are not uniformly integrable in those regimes.

The fraction of the standardized second moment carried by collective fields makes this explicit:

$$
\mathbb E[T_N^2\mathbf1_{\mathrm{collective}}]=\frac{Nc_N}{1+(N-1)c_N}\longrightarrow\begin{cases}0,&\alpha>1,\\c/(1+c),&\alpha=1,\\1,&0<\alpha<1.\end{cases}
$$

The optional α=1/2 family is a new mathematical comparison, not an additional scaling control in the reader, which retains fixed, inverse-size and inverse-square choices. At c=0 the law is IID. At p=0 or1 the standardizing variance is zero and T_N is not defined; those cases are outside this extension's assumptions.

![Second moments across vanishing collective-weight paths and the central standardized density compared with its narrower Gaussian weak limit.](/assets/learning/finite-size-research-rare-mode-limit-moments.en.svg)

The upper panel fixes p=c=1/2 and compares N Var(ρ) for three α values. The lower panel shows only a central window for α=1,N=256: its rare collective atoms lie outside the plotted window. The omitted atoms are precisely why central visual agreement cannot certify the variance of the whole law.

For the critical α=1 path, a theoretical central band has the limit below. Φ is the standard normal CDF, not a fitted sampling model:

$$
\Pr(|T_N|\leq1.96)\longrightarrow2\Phi(1.96\sqrt{1+c})-1.
$$

The table gives N, the actual second moment of W_N, the weak Gaussian's variance v, the collective second-moment contribution, the finite band probability and its limit. The exact rational event test squares both sides of the threshold inequality. The N=1024 row is a separate Python calculation beyond the reader's N≤512 range. These are probabilities under known p,c; they do not validate a plug-in confidence interval with estimated variance.

| N | E[W²] | Weak variance | Rare E[W²] | Finite band | Limit band |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.367188 | 0.250000 | 0.125000 | 0.948144 | 0.983627 |
| 64 | 0.373047 | 0.250000 | 0.125000 | 0.975461 | 0.983627 |
| 256 | 0.374512 | 0.250000 | 0.125000 | 0.983447 | 0.983627 |
| 1024 | 0.374878 | 0.250000 | 0.125000 | 0.983444 | 0.983627 |

```python
from fractions import Fraction as F
from math import comb, erf, sqrt

p, c = F(1,2), F(1,2)
v = p*(1-p)
for n in (16,64,256,1024):
    common = c/n
    masses = [(1-common)*F(comb(n,j),2**n)
              + common*((1-p if j==0 else 0)+(p if j==n else 0))
              for j in range(n+1)]
    variance = sum(m*(F(j,n)-p)**2 for j,m in enumerate(masses))
    band = sum(m for j,m in enumerate(masses)
               if (F(j,n)-p)**2 <= F(49,25)**2*variance)
    rare_second = n*common*v
    limit_band = erf(1.96*sqrt(1+float(c))/sqrt(2))
    print(n, f'{float(n*variance):.6f}', f'{float(v):.6f}',
          f'{float(rare_second):.6f}', f'{float(band):.6f}',
          f'{limit_band:.6f}')
```

### Research checks and scope

#### Does a convincing central histogram certify the variance?

**Research explanation.** No. Vanishing mass at increasingly large scaled values is nearly invisible to bounded test functions but can contribute order one to squared error. Inspect tail contributions or a suitable integrability condition before replacing moments by their weak-limit values.

#### Does finite unit variance force a unit-variance Gaussian weak limit?

**Research explanation.** No. This specified family has Var(T_N)=1 but a narrower Gaussian limit at α=1 and a point-mass limit at α<1. Variance is an unbounded test function and does not follow from weak convergence alone.

#### Is this an almost-sure statement about one growing field?

**Research explanation.** No. The preparation defines a law at each size without a coupling across sizes. These are weak and moment statements about that family, not a claim about a single spatial realization or a general AI data stream.

[MIT's uniform-integrability notes](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/resources/mit6_436jf18_lec19/) provide the convergence-of-expectations background. The discrete-mixture TV bound, moment escape, normalization regimes and band probabilities are derived here. The existing collective-mode reader continues to compute its original finite law and controls.

## Six exercises with explanations

### 1. Reconstruct the four-site covariance ledger

At N=4,p=c=1/2 on the fixed path, compute the diagonal sum, ordered off-diagonal sum, density variance and variance-equivalent count.

**Explanation.** Each diagonal variance is1/4 and each distinct covariance is1/8. Four diagonals give1 and twelve ordered pairs give3/2. Divide their total5/2 by16 to obtain5/32. Then (1/4)/(5/32)=8/5. Using six unordered pairs without multiplying by two would omit half the off-diagonal contribution.

### 2. Check the event against the displayed distribution

Use the four-site table and ε=1/2. Compute TI, TN and the three bounds. Which proposed bound fails for the mixed law?

**Explanation.** Only counts0 and4 qualify. Their IID masses sum to1/8 and their mixed masses to9/16. The collective event probability is1. BI=(1/16)/(1/4)=1/4, while BC=(5/32)/(1/4)=5/8 and BM=(1/2)(1/4)+(1/2)(1)=5/8. BI fails as a mixed-law bound because9/16>1/4.

### 3. Explain a smaller tail with a larger variance

For N=2,p=1/10,c=1/2 and ε=3/20, identify qualifying counts and verify both comparisons.

**Explanation.** Count0 has deviation1/10 and does not qualify. Counts1 and2 have deviations2/5 and9/10. IID probability is18/100+1/100=19/100; collective probability is1/10, so the mixture gives29/200. The variance formula gives (9/100)(1+1/2)/2=27/400, above the IID9/200. Different weighting of squared deviations and threshold crossings explains the difference.

### 4. Compare finite size with the limiting coefficient

At N=4,p=c=1/2, find Var(ρ) on the c/N and c/N² paths, then give each limit of N Var(ρ).

**Explanation.** The current weights are1/8 and1/32. Substitution gives11/128 and35/512. The limiting coefficients are v(1+c)=3/8 and v=1/4. The current scaled variances11/32 and35/128 have not yet reached these limits.

### 5. Distinguish boundary equality, zero mass and missing correlation

Find the event counts at N=3,p=1/3,ε=1/3. Explain what changes for deterministic p=0, and why N=1 does not establish self-averaging of a fixed-c family.

**Explanation.** Deviations are1/3,0,1/3,2/3, so the event is0,2,3. At p=0 all probability is at count0 and every positive-threshold deviation event has zero probability, regardless of which other lattice points qualify. Pair covariance is zero if a pair exists, but normalized correlation is undefined. At N=1 no distinct pair exists and the single-site law is Bernoulli for every c. Larger systems on a fixed positive-c path can still retain variance cv.

### 6. Separate a variance limit from a rare-event conclusion

For the all-one event with cN=c/N² and fixed 0<p<1,c>0, compare the IID and mixed formulas. Does Nvar/N→1 imply an IID tail?

**Explanation.** For the specific threshold ε=1−p with p<1/2, only the all-one endpoint qualifies and the probabilities are p^N and (1−c/N²)p^N+cp/N². The latter has a polynomial contribution that dominates the exponential term at large N. Thus the variance-equivalent fraction tends to one while the tail ratio diverges. The conclusion concerns this chosen event and preparation; the variance limit alone cannot establish a distribution-wide tail approximation.

The next family studies finite rounding and complex zeros in an explicitly restricted two-state partition function. The current reader lets you change the collective mode, size and threshold while comparing exact distributions and events. Continue through Book → Interactive Book.
