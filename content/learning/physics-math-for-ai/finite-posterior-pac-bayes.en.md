## Learning a distribution requires explicit assumptions

Assign posterior probabilities q to six thresholds. A Gibbs classifier randomly selects a hypothesis from q when predicting, so its risk is E_q R(h). It differs from selecting the largest weight or taking a majority vote. Calling q a posterior does not automatically make it the true Bayesian posterior of a specified generative model.

Assume IID training examples, loss in [0,1], a positive prior pi fixed independently of the sample, and a preselected confidence level delta. If prior tilt is adjusted after examining data, the formula still produces numbers but its confidence interpretation is no longer automatically justified. Blue is the actual Gibbs posterior and orange the prior.

## Compute posterior and KL

After evaluating empirical risks, form

$$
q(h)=\frac{\pi(h)e^{-\beta\widehat R(h)}}{\sum_j\pi(j)e^{-\beta\widehat R(j)}},\qquad
\mathrm{KL}(q\Vert\pi)=\sum_hq(h)\log\frac{q(h)}{\pi(h)}.
$$

At beta zero, q equals pi and KL is zero. Larger beta concentrates on low sample risk, not directly on the known population-risk curve. KL measures departure from the declared prior in nats; it is not automatically distance from a true model.

## Derive the bound in three steps

For fixed h, Hoeffding's moment inequality gives E_S exp(lambda(R-Rhat))≤exp(lambda²/(8N)). Average over the prior and apply Markov's inequality. Union-bound the predeclared grid {1,2,4,8,16,32,64,128}. With probability at least 1-delta, simultaneously for its eight values,

$$
\log E_\pi e^{\lambda(R-\widehat R)}\le\frac{\lambda^2}{8N}+\log\frac8\delta.
$$

Apply the finite change-of-measure inequality E_q f≤KL(q||pi)+log E_pi exp(f). Simultaneously for every q,

$$
R(q)\le\widehat R(q)+\frac{\mathrm{KL}(q\Vert\pi)+\log(8/\delta)}\lambda+\frac\lambda{8N}.
$$

Thus data-dependent posterior selection is allowed, and selection among that fixed lambda grid is accounted for. This does not permit free post-data optimization over every real lambda or an arbitrary data-dependent prior.

## Gibbs optimization and the cost of choosing a prior family

For beta>0 and a positive fixed prior, substitute log(q_beta/pi)=−beta*r−log Z into KL(q||q_beta). Rearrangement proves

$$
E_q r+\frac{\mathrm{KL}(q\Vert\pi)}\beta=-\frac{\log Z}\beta+\frac{\mathrm{KL}(q\Vert q_\beta)}\beta.
$$

Thus the Gibbs distribution minimizes a regularized empirical objective over distributions. Nonnegativity of the final KL proves optimality; it does not identify a true Bayesian posterior or the population minimizer. The code checks the full identity for three candidate distributions, not just its equality at the optimum.

A predeclared finite family of priors can also be selected after data with an explicit cost. Give prior j a fixed positive mass w_j, summing to one, and allocate failure probability delta*w_j/8 to each prior/grid pair. The same moment and union argument then replaces log(8/delta) by log(8/(delta*w_j)). All posteriors and family/grid choices are covered on one event. This does not cover arbitrary priors invented from the observed labels.

| Fixed prior-family weight | Extra numerator cost |
|---|---:|
| 1/2 | log 2 |
| 1/4 | log 4 |
| 1/8 | log 8 |

```python
from math import exp,log
prior=[.5,.3,.2];risk=[0,.25,.75];beta=2
Z=sum(p*exp(-beta*r) for p,r in zip(prior,risk))
gibbs=[p*exp(-beta*r)/Z for p,r in zip(prior,risk)]
def kl(q,p):return sum(a*log(a/b) for a,b in zip(q,p))
for q in (prior,[1/3]*3,gibbs):
    objective=sum(a*r for a,r in zip(q,risk))+kl(q,prior)/beta
    certificate=-log(Z)/beta+kl(q,gibbs)/beta
    assert abs(objective-certificate)<1e-14
    assert objective>=-log(Z)/beta-1e-14
    print(objective,kl(q,gibbs)/beta)
```

The live control still uses one declared prior, not this family extension. Its small-tilt KL now uses normalized Gibbs log-ratios u and the convex generator exp(u)*u−expm1(u). Its series starts u²/2+u³/3, retaining the nonnegative second-order effect rather than clipping a negative direct sum to zero. The analytical Gibbs KL can resolve a tilt below the displayed probability precision. `expandedKL` retains the original q*log(q/pi) sum. Both are binary64 calculations, not interval-certified arbitrary precision.

## Worked example and exercise

For N=32, delta=.05, lambda=32 and beta=0, the penalty is log(160)/32+1/8≈0.2836. Add it to the actual Gibbs empirical risk. An upper bound exceeding one can be clipped using bounded risk, but the unclipped value remains visible so vacuity is not hidden.

Exercise: use a uniform prior, noise .1 and beta zero. Find population risk. Solution: average the six risks [.42,.26,.10,.26,.42,.58] to obtain .34. This need not equal the realized empirical average. Confidence concerns repeated training samples, not a posterior probability .95 of getting an individual label right.

## Generalization analysis in AI

Record prior independence, loss range, sample size, confidence and hyperparameter selection costs. A prior built from the whole dataset cannot simply inherit the same guarantee. Likewise a Gibbs-risk bound cannot be renamed a vote-risk bound without further reasoning. This finite experiment exposes every computed term and assumption rather than presenting a bound as an unconditional certificate.

Reference: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/). The grid adjustment and numerical examples are derived here.
