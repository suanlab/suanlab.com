## Balanced traffic can still alternate forever

Imagine two rooms. At every bell, everyone moves to the other room. Half the population in each room is a stationary preparation: swapping the rooms leaves those proportions unchanged. But if everyone starts in the first room, the distribution alternates between the two rooms forever. Balanced stationary traffic and convergence from a chosen initialization are different properties.

The previous chapters constructed stationary flows and corrected asymmetric proposals. This chapter asks what happens after repeatedly applying a fixed transition. We will derive the complete distribution, inspect its signed error, and distinguish pointwise convergence from averaging distributions over time. A two-state model is small enough to make every boundary case explicit.

For general definitions and the finite-chain convergence theorem, see [Levin–Peres, sections 1.3, 1.5–1.6 and 4.3–4.4](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf). Its irreducible, aperiodic hypothesis is a useful sufficient condition. Below we directly solve our own two-state family, including reducible absorbing cases, rather than apply that hypothesis outside its scope.

## Specify two transition probabilities and a separate initialization

The named states are 0 and 1. Write a for the probability of moving 0→1 and b for moving 1→0. Use departure rows and destination columns:

$$
P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix},\qquad 0\leq a,b\leq1.
$$

The finite experiment supplies integer numerators A,B and a common positive denominator D, so a=A/D and b=B/D. Each numerator is between zero and D, and D is at most 1,000,000. A separate numerator U and denominator V define the initial probability of state 1:

$$
u=\frac UV,\qquad \mu(0)=(1-u,u),\qquad 0\leq U\leq V,\quad 1\leq V\leq10^6.
$$

The initial probability u is independent of a and b. Changing the initialization changes the preparation, not the transition matrix. We record t=0 through a chosen N≤512. A bell represents one discrete transition, not physical seconds or a continuous-time transition rate.

| Quantity | Meaning | What must stay separate |
| --- | --- | --- |
| a,b | Conditional probabilities of changing rooms | Current room probabilities |
| u | Initial probability of state 1 | A stationary probability derived from P |
| μ(t) | Distribution after t transitions | A random trajectory's occupied state |
| N+1 | Number of recorded distributions from 0 through N | Number N of applied transitions |

## Solve stationarity and retain the identity family

A stationary distribution obeys one independent balance equation:

$$
\pi_0a=\pi_1b,\qquad \pi_0+\pi_1=1.
$$

When a+b>0, solving it gives a unique stationary distribution:

$$
\pi_0=\frac b{a+b},\qquad \pi_1=\frac a{a+b}.
$$

Uniqueness does not require both a and b to be positive. If a=0 and b>0, state 0 is absorbing and π=(1,0) is the unique stationary law. The reverse case has π=(0,1). Transient initial probability can flow into the absorbing state without creating another closed class.

If a=b=0, the formula would divide by zero. The matrix is the identity and every distribution is stationary. The model therefore provides a stationary family, not a selected equilibrium. Each initialization remains its own limit. Without choosing a reference from that family, a distance to “the” stationary distribution is not provided; this missing reference is not a zero distance.

## State detailed balance and reverse only on stationary support

For two states, the stationary balance equation is already the only off-diagonal detailed-balance equation. Every stationary law in this family therefore satisfies detailed balance. That includes the alternating chain and stationary laws of the identity, without implying that arbitrary initial distributions mix.

A stationary statistical time reversal requires division only on states with positive stationary probability:

$$
P^{\mathrm{rev}}_{ij}=\frac{\pi_jP_{ji}}{\pi_i}\qquad\text{only where }\pi_i>0.
$$

For a,b>0, the stationary distribution has full support and reversal gives P itself. In an absorbing example π=(1,0), the reverse row at state 0 is (1,0). The row conditioned on stationary state 1 is not provided: that conditioning event has zero probability. Filling it with the original transient row would invent a reverse process outside the observed stationary support.

The statement concerns stationary path probabilities. It does not include physical momentum reversal or arbitrary initial preparations at transient states.

## Reduce the distribution update to an affine recurrence

Let xt be the probability of state 1 at time t. The inflow from state 0 and the portion staying at state 1 give

$$
x_{t+1}=a(1-x_t)+(1-b)x_t=a+(1-a-b)x_t.
$$

The forward one-step flux is (1−xt)a and the backward flux is xtb. Their difference is xt+1−xt. A negative difference means net flow toward state 0, not a negative transition probability.

For a=1/4,b=1/2 and u=0, the first recorded probabilities of state 1 are:

| t | xt | Signed deviation from 1/3 | Total variation |
| --- | --- | --- | --- |
| 0 | 0 | −1/3 | 1/3 |
| 1 | 1/4 | −1/12 | 1/12 |
| 2 | 5/16 | −1/48 | 1/48 |
| 3 | 21/64 | −1/192 | 1/192 |

These are exact distribution calculations. A sampled path would contain a state 0 or 1 at each time, rather than all four probabilities. Exact distribution evolution does not supply the sampling variance of an estimate from one path.

## Derive the closed form and keep the eigenvalue sign

For a+b>0, subtract the stationary fixed point from the recurrence. With λ=1−a−b,

$$
x_t-\pi_1=(u-\pi_1)\lambda^t,\qquad x_t=\pi_1+(u-\pi_1)\lambda^t.
$$

The constant direction has eigenvalue one, while the zero-sum distribution difference has eigenvalue λ. Equivalently, writing Π for the matrix whose two rows are π,

$$
P^t=\Pi+\lambda^t(I-\Pi),\qquad \Pi=\begin{pmatrix}\pi_0&\pi_1\\\pi_0&\pi_1\end{pmatrix}.
$$

At t=0 the power is one, including when λ=0, and P^0 is the identity. When λ=0, the distribution resets to π after one transition, not before applying the first transition. When a=b=0, use xt=u directly; do not obtain a stationary law by dividing by a+b.

For positive λ less than one, a signed deviation shrinks without changing sign. For negative λ, it changes sign each step. For example, a=b=3/4 gives λ=−1/2, and u=0 produces 0,3/4,3/8,9/16. The preparation overshoots and undershoots 1/2. This sign change is not a negative probability or negative variance.

## Identify communicating classes and return periods

If a,b>0, both states communicate. The period is two only at a=b=1, where every step changes the state. Otherwise at least one state has a holding loop and the communicating class has period one.

If exactly one movement probability is zero, there are two communicating classes and only one closed class. The other state is transient. At a=0,b=1, the transient state 1 has no possible return after departure; its return period is not provided. The absorbing state 0 has period one. At a=b=0 both singleton classes are closed and have period one.

| Kernel | Unique stationary law? | Communication | Pointwise convergence |
| --- | --- | --- | --- |
| a,b>0, except a=b=1 | Yes | Irreducible, period one | From every initialization |
| a=b=1 | Yes | Irreducible, period two | Only from an already stationary distribution |
| Exactly one of a,b is zero | Yes | One closed absorbing class | From every initialization |
| a=b=0 | No | Two closed singleton classes | Each initialization stays at itself |

This table distinguishes the existence of a limit from convergence to a uniquely specified reference. In particular, reducibility does not prevent an absorbing chain from having a unique stationary law and attracting every initialization.

## Separate three kinds of limiting statement

For the unique-stationary cases with |λ|<1, the closed form directly proves that xt tends to π1. At a=b=1, λ=−1 and a nonstationary preparation never loses its alternating error. If u=1/2, it is stationary from the start even though the same kernel fails to mix other initial distributions.

The identity case also has a pointwise limit: the limit is u. Saying that it has no unique stationary reference is not saying that its distribution fails to converge. The model reports that initial-dependent limit while leaving reference-dependent distances unavailable.

A third statement concerns averages of distributions over time. Those averages may converge even when the distribution at the selected time does not. We will derive that result separately, keeping its observation count explicit.

## Derive the actual total-variation distance

For two states, the two component errors have equal magnitudes and opposite signs. Therefore

$$
\|\mu(t)-\pi\|_{\mathrm{TV}}=|x_t-\pi_1|=|u-\pi_1|\,|\lambda|^t\qquad(a+b>0).
$$

Unlike a general multi-state row-contraction upper bound, this is an equality for this complete two-state family. Total variation is nonincreasing even when the signed component error alternates. Smoothing away the alternating component would erase information; interpreting its sign changes as a growing distance would be another error.

An absorbing example with a=0,b=1/2,u=1 has xt=2^−t and distance 2^−t from π=(1,0). A deterministic alternating example from u=0 has constant distance 1/2. They both satisfy detailed balance at their stationary laws, but their transient behavior is very different.

## Use worst-initial distance for a mixing criterion

To assess the kernel rather than one favorable initialization, maximize the distance over possible initial distributions. A linear scalar error is maximized by one of the two point masses:

$$
d(t)=\max_{x\in\{0,1\}}\|P^t(x,\cdot)-\pi\|_{\mathrm{TV}}=\max(\pi_0,\pi_1)|\lambda|^t.
$$

For this experiment, define a quarter-distance crossing using an inclusive threshold:

$$
t_{1/4}=\min\{t\geq0:d(t)\leq1/4\}.
$$

When a=b=1/4, d(0)=1/2 and d(1)=1/4, so the crossing is one. For a=0,b=1/2, d(t)=2^−t, so the crossing is two. The latter is a direct distance calculation for an absorbing kernel, not an application of an irreducible-chain theorem.

An already stationary initialization can have distance zero at t=0 while d(0) is positive. It does not certify worst-initial mixing. If the selected recording range stops before the crossing, the result says “not reached within the recorded range.” It does not say that the crossing never occurs. Near-identity or near-alternating probabilities can require many more than 512 transitions. Exact alternation is different: d(t)=1/2 forever, so the quarter threshold is never reached. The identity has no selected unique reference for this criterion.

## Distinguish signed and absolute spectral gaps

The sign of λ matters when interpreting a spectral-gap number. In this family the two relevant quantities are

$$
g_{\mathrm{signed}}=1-\lambda=a+b,\qquad g_{\mathrm{absolute}}=1-|\lambda|.
$$

At a=b=1, the signed gap is two while the absolute gap is zero. A claim that a larger signed gap alone guarantees faster distribution mixing would fail at this reversible example. The absolute contraction factor remains one because λ=−1 preserves the alternating magnitude.

For a=b=999999/1000000, the signed gap is close to two but the absolute gap is only 2/1000000. The chain eventually converges, yet its alternating error decays slowly. Both small positive movement and nearly deterministic switching can produce slow contraction. Assumptions such as laziness are important when moving from a signed-gap statement to a general mixing interpretation.

## Average distributions without claiming a sampling-error result

Include the initial distribution and average the N+1 recorded distributions:

$$
\bar\mu(N)=\frac1{N+1}\sum_{t=0}^{N}\mu(t),\qquad \bar x_N=\frac1{N+1}\sum_{t=0}^{N}x_t.
$$

For a+b>0, λ≠1 and the finite geometric sum gives

$$
\bar x_N=\pi_1+(u-\pi_1)\frac{1-\lambda^{N+1}}{(N+1)(1-\lambda)}.
$$

At exact alternation, λ=−1. Starting at state 0, the distributions through N=2 have state-1 probabilities 0,1,0, giving average 1/3 and distance 1/6 from 1/2. Through N=3 the average is exactly 1/2, although the current distribution is concentrated at state 1. As N grows, the averaged distribution converges to π even when the pointwise sequence does not.

For the identity, the average remains the initial distribution. It is not forced toward uniform. The model averages probability laws; equivalently, it gives the expected empirical occupancy of a path with that initialization. It does not give the variance, confidence interval or effective sample size of a random time average. Those require the joint dependence of the sampled path, not just its single-time marginals.

## Keep an eigenvalue separate from a normalized observed correlation

Let X be the indicator of state 1 under a unique stationary law. The stationary variance and one-lag covariance are

$$
\operatorname{Var}_{\pi}(X)=\pi_0\pi_1,\qquad \operatorname{Cov}_{\pi}(X_0,X_1)=\pi_0\pi_1\lambda.
$$

When both stationary probabilities are positive, dividing the covariance by the variance gives the normalized one-lag correlation λ. At a=b=1 it is −1, the covariance is −1/4 and the variance is +1/4. Negative dependence does not make a variance negative.

For an absorbing stationary law, the indicator is constant. Its variance and covariance are exactly zero, but normalized correlation would divide by zero and is not provided. The matrix still has the algebraic eigenvalue λ. At the identity there is no selected stationary reference, so reference-dependent variance and correlation are not provided; choosing a member of the stationary family would be an additional preparation decision.

This is a binary indicator model, not a unit-variance Gaussian AR(1) process. The matching eigenvalue form explains a useful connection to the earlier correlated-average chapter, without making their marginal laws or uncertainty calculations identical.

## Preserve exact deviations below the linear display range

The input numerator and denominator are exact integers, so finite matrix powers and their signed differences can be evaluated as exact rational numbers. The independent closed form should agree with that evolution before either is projected to a decimal display. A rounded 1/3 or a rounded stationary probability is not the exact fraction itself.

Take A=500000,B=499999,D=1000000 and u=0. Then λ=10^−6 and

$$
\pi_1=\frac{500000}{999999},\qquad x_t-\pi_1=-\frac{\pi_1}{10^{6t}}.
$$

At t=64 the signed error and total variation remain nonzero, but their linear floating-point projection is outside the nonzero display range. Both probability bars can match the rounded stationary bars. The exact current distribution is still not stationary. Replacing its error by zero would change the mathematical statement.

The model retains the exact rational endpoints and sign. A magnitude logarithm can help display very small nonzero quantities, but it is approximate and is not a certified transcendental interval. Exact zero has no finite logarithmic magnitude. Neither a plotting epsilon nor a tolerance on two rounded probabilities determines exact stationarity, communication or a threshold crossing.

## Connect alternation and mixing to the experiment

Imagine transferring probability mass between two rooms. A,B,D specify the transition rule, while U,V specify the initial mass U/V in state 1. Transition denominator D and initialization denominator V are independent. The default A=1,B=2,D=4 gives a=1/4, b=1/2 and stationary state-one probability 1/3.

| Action | Observation to inspect | Question to explain |
| --- | --- | --- |
| Compare ‘Positive eigenvalue, monotone approach’ with ‘Negative eigenvalue, alternating approach’ | Signed state-one deviation and TV distance | Can distance shrink while the deviation changes sign each step? |
| From ‘Exact alternation, even final step’, press ‘One transition step’ | The current law moves to the opposite state; the pointwise limit remains unavailable | Why can the current law fail to converge despite a stationary, detailed-balanced reference? |
| Choose ‘Alternating kernel, stationary start’ | Current TV is zero, but worst-initial TV remains 1/2 | Does one favorable initialization prove mixing from every start? |

Changing N recomputes N transitions from the same initial law. One transition step increments that N; it does not draw one random sample. In logarithmic distance display, exact zeros appear as separate dots below the axis, while the signed-deviation plot retains its linear axis. The Cesaro bars average N+1 laws from t=0 through N. Turning a history-table page changes neither that averaging range nor N.

## Research extension: calculate a random time average's error

The Cesaro bars above average marginal distributions. They give the expected occupancy of a random path, but not its sampling variance. To complete that distinction, let X_t be the random indicator of state 1 and calculate the joint covariances. We can then compare initialization bias with random error, and see why improving distribution mixing can sacrifice useful negative correlation.

### Recover two-time covariance from conditional expectation

Keep a+b>0, π₁=a/(a+b), λ=1−a−b and initial probability u. Write x_t=E[X_t]. The Markov property and the original recurrence imply, for j≥i,

$$
\mathbb E[X_j\mid X_i]=\pi_1+(X_i-\pi_1)\lambda^{j-i},\qquad \operatorname{Cov}(X_i,X_j)=x_i(1-x_i)\lambda^{j-i}.
$$

Indeed the later innovations in the conditional mean contribute no covariance with X_i, leaving its own variance times the propagation factor. With a nonstationary start, that variance depends on i. Replacing it everywhere by π₀π₁ would silently change the initial preparation.

Record n states X₀,…,X_(n−1), so n=N+1 in the original reader. Their average estimates π₁ and has bias

$$
\widehat\pi_{1,n}=\frac1n\sum_{t=0}^{n-1}X_t,\qquad B_n=\mathbb E[\widehat\pi_{1,n}]-\pi_1=(u-\pi_1)\frac{1-\lambda^n}{n(1-\lambda)}.
$$

This is the expectation already represented by the averaged bars, now interpreted as estimator bias. The variance requires all pairs, not just those bars:

$$
V_n^{\mathrm{path}}=\frac1{n^2}\left[\sum_{i=0}^{n-1}v_i+2\sum_{0\leq i<j<n}v_i\lambda^{j-i}\right],\qquad v_i=x_i(1-x_i).
$$

The mean-squared error about the target adds the squared bias. An initially deterministic path can have zero variance at the first observation while still have a large error relative to π₁.

$$
\operatorname{MSE}(\widehat\pi_{1,n};\pi_1)=V_n^{\mathrm{path}}+B_n^2.
$$

At n=1, these formulas give V=u(1−u) and B=u−π₁. At exact alternation a=b=1, an even n average equals 1/2 exactly. For odd n its variance is u(1−u)/n² and its squared bias is (u−1/2)²/n², so MSE is 1/(4n²) for every u. A deterministic start's odd-length error is bias; a stationary start's is variance. Equal MSE need not mean equal decomposition.

### Relate stationary path variance to the earlier AR(1) formula

Now restrict to a,b>0 and |λ|<1 and initialize at π. The marginal variance is v=π₀π₁>0 and every normalized covariance is λ^k. Therefore the exact finite variance uses the same covariance sum as the earlier AR(1) calculation, multiplied by v. The random mean is generally discrete rather than exactly Gaussian. Its large-n variance coefficient is

$$
\sigma^2=v\frac{1+\lambda}{1-\lambda},\qquad \tau=\frac{\sigma^2}{v}=\frac{1+\lambda}{1-\lambda},\qquad nV_n^{\mathrm{path}}\longrightarrow\sigma^2.
$$

This connection uses the stationary binary observable, not an assertion that the two processes share a Gaussian marginal law. For an absorbing stationary law v=0, normalized correlation and τ are not defined by division; the earlier transient covariance formula still applies directly.

### Add holding and distinguish two optimization objectives

Define a new kernel by independently choosing to hold with probability δ, otherwise applying P. It preserves the same π, and its nonconstant eigenvalue becomes

$$
P_\delta=\delta I+(1-\delta)P,\qquad \lambda_\delta=\delta+(1-\delta)\lambda,\qquad 0\leq\delta<1.
$$

Its stationary variance coefficient increases with δ. Differentiate explicitly at fixed original P and fixed observable:

$$
\sigma_\delta^2=v\frac{1+\lambda_\delta}{1-\lambda_\delta},\qquad \frac{d\sigma_\delta^2}{d\delta}=\frac{2v(1-\lambda)}{(1-\lambda_\delta)^2}>0.
$$

For negative λ, however, distribution error contracts fastest when holding moves that eigenvalue exactly to zero. The resulting rows are π, so one transition resets any initial distribution:

$$
\delta_{\mathrm{reset}}=-\frac{\lambda}{1-\lambda}\quad(\lambda<0),\qquad \lambda_{\delta_{\mathrm{reset}}}=0,\qquad P_{\delta_{\mathrm{reset}}}=\Pi,\qquad \sigma_{\delta_{\mathrm{reset}}}^2=v.
$$

For a=b=0.9, λ=−0.8 and δ_reset=4/9. Resetting makes distribution mixing immediate but raises the stationary asymptotic variance from v/9 to v per recorded state. It removes the cancellation that helped the time average. This compares two specified attempt clocks; charging only actual movements or measuring computation time is another comparison.

![Finite mean-squared errors of a two-state occupancy average and the differing effects of added holding on contraction and variance inflation.](/assets/learning/detailed-balance-research-initialization-laziness-error.en.svg)

The upper panel compares exact finite MSE from the deterministic start u=0, including the initial observation. The lower panel compares |λ_δ| with τ_δ for the original a=b=0.9. Both are dimensionless but answer different questions; their minima need not coincide.

The table fixes n=9 and uses symmetric a=b. The first two rows use the original kernel, the last two the reset kernel a=b=0.5. Compare u=0 with the stationary u=0.5 in each case. Symmetry makes MSE agree between these starts while its bias/variance split differs. The code computes exact rational covariances, not the uncertainty of an empirically estimated transition matrix.

| a=b | u | n | Bias | Variance | MSE |
| --- | --- | --- | --- | --- | --- |
| 0.9 | 0.0 | 9 | -0.035007 | 0.003590 | 0.004815 |
| 0.9 | 0.5 | 9 | 0.000000 | 0.004815 | 0.004815 |
| 0.5 | 0.0 | 9 | -0.055556 | 0.024691 | 0.027778 |
| 0.5 | 0.5 | 9 | 0.000000 | 0.027778 | 0.027778 |

```python
from fractions import Fraction as F

n = 9  # Nine observations include the initial state: eight transitions.
for a, u in ((F(9,10),F(0)), (F(9,10),F(1,2)),
             (F(1,2),F(0)), (F(1,2),F(1,2))):
    b = a
    stationary = a/(a+b)
    lam = 1-a-b
    means = [u]
    for _ in range(n-1):
        means.append(a+lam*means[-1])
    variances = [x*(1-x) for x in means]
    variance = (sum(variances)+2*sum(variances[i]*lam**(j-i)
                 for i in range(n) for j in range(i+1,n)))/n**2
    bias = sum(means)/n-stationary
    mse = variance+bias*bias
    print(f'{float(a):.1f}', f'{float(u):.1f}', n,
          f'{float(bias):.6f}', f'{float(variance):.6f}',
          f'{float(mse):.6f}')
```

### Research checks and boundaries

#### Can averaged marginal bars determine the variance of a sampled mean?

**Research explanation.** They determine its expectation. The variance also needs the joint covariance of states at different times. Two processes can share single-time marginals while having different time-average uncertainty.

#### Does the reset kernel always give a smaller finite MSE?

**Research explanation.** No. It removes initialization error after one transition, but also removes negative serial covariance. The table gives a specified finite counterexample. For a general initialization, target, observable or recording length, compute bias and covariance rather than transfer a universal ranking.

#### Can the stationary variance coefficient be used at a nonmixing endpoint?

**Research explanation.** Not by an ordinary summable-covariance argument. Exact alternation has a separate finite cancellation calculation; the identity lacks a unique selected stationary target. Keep those boundary preparations distinct from the interior asymptotic formula.

[Levin–Peres, chapters 1, 12 and 17](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf) supplies the Markov/spectral and estimation context. The two-time covariance, initialization MSE and holding comparison here are derived explicitly for this binary family. They supplement the existing averaged-distribution display without turning it into a path-variance control or a Monte Carlo simulation.

## Predict, calculate and explain

### 1. Solve the asymmetric recurrence

Use a=1/4,b=1/2 and u=0. Find π1, λ, x3 and the selected preparation's total variation at t=3.

**Explanation.** π1=a/(a+b)=1/3 and λ=1/4. The closed form gives x3=(1/3)(1−1/64)=21/64. The distance is 1/192. The worst-initial distance is instead (2/3)(1/64)=1/96; its maximizing initialization is not u=0 in this asymmetric example.

### 2. Compare pointwise and averaged behavior

Set a=b=1,u=0 and record through N=3. What is the current state-1 probability, its distance, the averaged probability and the two spectral gaps?

**Explanation.** x3=1 and the pointwise distance is 1/2. The four recorded probabilities are 0,1,0,1, so the average is 1/2. The signed gap is two and the absolute gap is zero. The averaged equality does not establish pointwise convergence; nonstationary preparations continue to alternate.

### 3. Examine an absorbing, reducible chain

Use a=0,b=1/2,u=1. Is the stationary law unique? When is the quarter criterion reached, and can you provide a stationary normalized indicator correlation?

**Explanation.** The unique stationary law is (1,0), despite reducibility. xt=2^−t, and the worst-initial distance first equals 1/4 at t=2. At stationarity the indicator variance and covariance are zero, so normalized correlation is not provided. The stationary reverse row at the zero-probability state 1 is also not provided.

### 4. Interpret the identity boundary

Set a=b=0 and u=1/3. Give the pointwise limit, the averaged limit and the distance to the unique stationary law.

**Explanation.** The distribution stays (2/3,1/3), so both limits equal that initial law. There is no unique stationary law and no selected reference distance. Calling the unavailable distance zero would silently choose the initialization as the reference. Calling the preparation nonconvergent would also be wrong.

### 5. Distinguish a favorable initialization from a kernel criterion

Use a=b=1/3,u=1/2 and record only t=0. Does the zero current-preparation distance certify quarter mixing? Does not observing a crossing mean it never occurs?

**Explanation.** The chosen preparation is stationary, but the worst-initial distance at t=0 is 1/2. The recorded result has no crossing. Extending to t=1 gives worst distance 1/6 and reaches the threshold. “Not reached in this range” differs from permanent nonconvergence.

### 6. Diagnose coincident rounded bars

Use A=500000,B=499999,D=1000000,u=0 and t=64. The probability display matches the stationary display. Is the total variation exactly zero?

**Explanation.** No. It is π1/10^384 with π1=500000/999999, hence strictly positive. Its linear display is unavailable while its exact fraction and sign remain available. The independently computed recurrence and matrix evolution agree on a nonzero error; equal-looking bars do not prove stationarity.

## Connect finite mixing to AI and further research

A fixed two-state example can separate initialization bias, stationary invariance, negative dependence, worst-initial contraction and averaged marginals. These distinctions matter when interpreting a sampler that seems stable, a binary diagnostic trace or a reported spectral gap. A slowly contracting preparation and a periodic nonconvergent preparation require different explanations.

Using a learned or estimated transition matrix adds uncertainty about the kernel itself. Coarse-graining a larger process into two labels does not automatically give a Markov process. Time-varying kernels, general support graphs, high-dimensional spectral estimates and dependence-aware Monte Carlo error require further assumptions and proofs. A continuous-time generator also uses a different clock; probabilities in this chapter cannot simply be relabeled as rates.

The finite calculation completes the introductory derivation for this specified family. It does not complete the general theory, empirical sampler validation, physical thermodynamics or the broader research curriculum.
