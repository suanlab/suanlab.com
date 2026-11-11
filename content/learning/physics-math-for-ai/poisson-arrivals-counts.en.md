## A counting path is different from its count distribution

A homogeneous Poisson process models arrivals with constant intensity lambda and independent increments on disjoint time intervals. The random count N(T) has mean and variance lambda T. One observed count need not equal either number, just as one coin toss need not equal its probability of heads.

The upper diagram shows a seeded arrival realization and the theoretical line lambda t. Each blue jump records an actual simulated event. The orange line is an expectation across realizations rather than a smoother version of the same sample. At zero intensity there are no arrivals; at zero observation time the count is zero.

## Build arrivals from waiting times

For positive intensity, independent exponential waiting times have the memoryless property. Using an open-interval uniform draw U, a waiting time is -log(U)/lambda. Add these times until the next event would fall outside the observation window. The pseudorandom seed makes the example reproducible; it is not a statistical proof of independent increments.

$$
\Pr(N(T)=k)=e^{-m}\frac{m^k}{k!},\qquad m=\lambda T,
\qquad p_{k+1}=p_k\frac m{k+1}.
$$

The recurrence computes the displayed count probabilities directly. It avoids interpreting the random number of visible arrivals as an estimate with no uncertainty. The simulation has a reported safety cap, and reaching it would be distinguishable from an unrestricted count path.

## A truncated plot must keep its missing mass

The main blue stems show p0 through pK. The orange stem at coordinate K+1 represents the entire event N(T)>K, not the single event N(T)=K+1. Its explicit bucket meaning prevents the plotting cutoff from becoming an accidental change of probability model.

The code sums the remaining positive recurrence terms rather than forcing the blue values to sum to one. Renormalizing them would instead condition the law on N(T)<=K. That is a valid different distribution only if such conditioning is intended and stated. Cutoff changes leave the seeded arrival path unchanged.

## Work through intensity two

With lambda=2 and T=1, the first three probabilities are exp(-2), 2exp(-2), and 2exp(-2). Their sum is 5exp(-2), approximately 0.676676. The remaining tail is approximately 0.323324. Theoretical count variance is two even if this particular sampled path shows zero, one or several arrivals.

## Conditioning on a total count changes the dependence structure

For 0<s<T, independent increments give N(s)∼Poisson(λs) and N(T)−N(s)∼Poisson(λ(T−s)). Divide their joint probability by the Poisson total probability. For λ>0 and an observed total n,

$$
\Pr(N(s)=k\mid N(T)=n)=\binom nk(s/T)^k(1-s/T)^{n-k}.
$$

Thus counts in the two pieces become negatively dependent after conditioning: one count determines the other. The unconditioned covariance is zero, while the conditional covariance is −n(s/T)(1−s/T). Equivalently, conditional on n arrivals, the ordered arrival times have density n!/Tⁿ on their ordered simplex; they are the order statistics of n independent uniform times. This is not a simulation obtained by renormalizing a truncated Poisson plot.

For n=4 and s/T=1/4, the conditional mean is 1 and variance 3/4.

| k | Conditional probability |
|---|---:|
| 0 | 81/256 |
| 1 | 108/256 |
| 2 | 54/256 |

```python
from fractions import Fraction as F
from math import comb
n, p = 4, F(1, 4)
law = [comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)]
mean = sum(k*w for k,w in enumerate(law))
variance = sum((k-mean)**2*w for k,w in enumerate(law))
covariance = sum((k-mean)*((n-k)-(n-mean))*w for k,w in enumerate(law))
assert sum(law)==1 and mean==1 and variance==F(3,4)
assert covariance == -variance
print(law, mean, variance, covariance)
```

The rational calculation checks the entire binomial law and its negative covariance. For an unconditioned observed count, log likelihood is n logλ−λT plus constants. Its interior maximizer is n/T and Fisher information is T/λ for λ>0. At n=0 the nonnegative-rate MLE lies at the boundary; regular interior approximations require separate justification. A count likelihood, a conditional bridge, and the unconditional seeded path answer different statistical questions.

## Exercise: two windows and two cutoffs

For a Poisson process, counts on adjacent unit intervals are independent Poisson(lambda) variables, and their sum is Poisson(2lambda). Compare this distributional statement with simply counting events in one simulated two-unit path. Then increase K from two to ten at lambda T=2. Explain why the tail bucket shrinks while the underlying arrival times and theoretical mean remain fixed. Finally describe what a renormalized three-stem plot would mean instead.

Further reading: [Lawler's author-hosted stochastic-calculus notes](https://www.math.uchicago.edu/~lawler/finbook.pdf). The finite display and arrival experiment are constructed independently.
