## Samples and the law of their mean

Assume independent identically distributed Bernoulli(p) observations. Their sum S is binomial, and their mean is S/n. The lab evaluates every possible k from zero through n rather than drawing a random sample path. Blue values are floating-point evaluations of the binomial formula. Increasing sample count leaves individual-trial variance unchanged while reducing the variance of the mean.

$$
P(S=k)={n\choose k}p^k(1-p)^{n-k},\quad E[\bar X]=p,\quad\operatorname{Var}(\bar X)=\frac{p(1-p)}n.
$$

## The LLN concerns fixed deviations

The weak law states that, for fixed positive ε, the probability that the mean differs from p by at least ε tends to zero. It does not make every finite sample mean equal to p. Chebyshev’s inequality bounds this probability by variance divided by ε². The displayed bound is capped at one. Compare the actual binomial tail sum with this upper bound rather than assuming the bound is tight.

$$
P(|\bar X-p|\ge\varepsilon)\le\frac{p(1-p)}{n\varepsilon^2}\longrightarrow0.
$$

## The CLT rescales fluctuations

The central limit theorem centers the mean and divides by its own standard deviation, giving convergence in distribution to a standard normal. Here it applies when 0<p<1. It does not say that the unscaled mean approaches a normal law with fixed variance. Independence, identical distribution and finite nonzero variance matter; dependent observations require other results.

$$
\frac{S-np}{\sqrt{np(1-p)}}\;\Rightarrow\;N(0,1).
$$

## Compare masses rather than density heights

Orange values are normal probabilities of width-one bins around each integer k, rather than density heights. This continuity correction puts both series in probability units. Normal tails outside the displayed bins are not renormalized, so their sum can be below one. The normal CDF uses a numerical approximation with absolute error within roughly one millionth. At p=0 or one, the binomial law is a point mass; standardization and the normal approximation are unavailable.

$$
P(S=k)\approx\Phi\!\left(\frac{k+1/2-np}{\sqrt{np(1-p)}}\right)-\Phi\!\left(\frac{k-1/2-np}{\sqrt{np(1-p)}}\right).
$$

## See where the Gaussian limit comes from

Keep p fixed in (0,1) and set σ²=p(1−p). For a real random variable W, its characteristic function is E[exp(itW)]. Independence turns the characteristic function of a sum into a product. The centered Bernoulli variable is bounded; Taylor expansion of the exponential therefore has a bounded third-order remainder. For each fixed real t,

$$
E\exp\!\left(\frac{it(X-p)}{\sigma\sqrt n}\right)
=1-\frac{t^2}{2n}+O(n^{-3/2}).
$$

The first-order term vanishes because E[X−p]=0, and the second-order coefficient uses E[(X−p)²]=σ². Raising this expression to the nth power, or taking its logarithm near one, gives

$$
\varphi_{Z_n}(t)=\left[1-\frac{t^2}{2n}+O(n^{-3/2})\right]^n
\longrightarrow e^{-t^2/2},\qquad Z_n=\frac{S_n-np}{\sigma\sqrt n}.
$$

The limit is the standard normal characteristic function. Lévy's continuity theorem converts this pointwise characteristic-function limit into convergence in distribution. We use that theorem here; its measure-theoretic proof is a separate step, not established by plotting a few values. The remainder constant can depend on p and t. This argument fixes p and does not supply a uniform normal-approximation error bound as p changes with n.

## A rare-event sequence that does not become normal

In a different experiment, let each row contain n independent Bernoulli(1/n) trials. Each row is identically distributed, but the trial law changes between rows. Now the count S_n has mean one and variance 1−1/n. Define its standardized version by the following formula:

$$
p_n=1/n,\qquad Z_n=\frac{S_n-1}{\sqrt{1-1/n}}\quad(n\ge2).
$$

For every n≥2, the event Z_n≤−1/2 is exactly S_n=0. A count of one gives Z_n=0, and a count of zero is below −1. Therefore

$$
P(Z_n\le-1/2)=(1-1/n)^n\longrightarrow e^{-1}\ne\Phi(-1/2).
$$

| n | Actual CDF at −1/2 | Standard normal CDF there |
|---|---|---|
| 5 | 0.327680 | 0.308538 |
| 20 | 0.358486 | 0.308538 |
| 100 | 0.366032 | 0.308538 |

The limit e⁻¹ is about 0.367879. Since the normal CDF is continuous at −1/2, this mismatch rules out convergence to the standard normal law. Unit variance by itself does not imply Gaussian shape. Here the expected rare-event count stays one instead of growing. The fixed-law CLT proved above is not contradicted: its assumptions do not describe this row-dependent experiment.

## Concentration of the mean is still a different question

The row mean has expectation 1/n and variance (n−1)/n³. Chebyshev's bound shows that the difference between the row mean and its expectation converges to zero in probability; the mean also converges to zero. This can coexist with nonnormal standardized counts. An evaluation with a vanishing error rate may therefore have a concentrated average while normal count-error bars remain inaccurate. Inspect expected event counts and the sampling assumptions rather than relying on the sample size alone.

## Compare exact rational probabilities with characteristic functions

The standard-library code evaluates rare-event zero masses exactly as fractions, and compares fixed-p characteristic functions numerically with their normal limit. The former is an exact finite calculation; the latter is an illustration of the proof, not a replacement for it or a convergence-rate certificate. This page's controls still examine one specified binomial law at a time and have n≤30; larger n and the changing-p sequence below belong to this separate executable example.

```python
from fractions import Fraction as Q
from math import exp, erf, sqrt, floor
from cmath import exp as cexp

normal_cdf = (1 + erf(-.5/sqrt(2))) / 2
for n in [5, 20, 100, 1000]:
    p = Q(1, n)
    zero_mass = (1-p)**n
    cutoff = 1 - .5*sqrt(1-1/n)
    assert floor(cutoff) == 0  # S integer: Z <= -1/2 iff S == 0.
    variance_mean = p*(1-p)/n
    assert variance_mean == Q(n-1, n**3)
    print(n, "CDF at -1/2:", float(zero_mass),
          "normal:", normal_cdf, "mean variance:", float(variance_mean))
assert abs(float((1-Q(1, 1000))**1000)-exp(-1)) < .0002
assert abs(exp(-1)-normal_cdf) > .05

p, t = .25, .7
sigma = sqrt(p*(1-p))
errors = []
for n in [10, 100, 1000]:
    step = t/(sigma*sqrt(n))
    one = (1-p)*cexp(-1j*step*p) + p*cexp(1j*step*(1-p))
    characteristic = one**n
    error = abs(characteristic-exp(-t*t/2))
    errors.append(error)
    print("fixed-p characteristic:", n, characteristic, "error:", error)
assert errors[-1] < errors[0]
```

## Further questions and research use

1. Does an LLN imply a CLT with the same assumptions? **No: concentration and a standardized distributional limit are different assertions.**
2. Why compare CDF values instead of a point mass to a density height? **They are probabilities in the same units, and convergence in distribution is tested at continuity points of the limiting CDF.**
3. Does the Taylor remainder above prove a uniform CDF error rate? **No: an additional argument would be required.**

For the continuity theorem and characteristic-function method, see [Sheffield, MIT 18.175, Lecture 15](https://math.mit.edu/~sheffield/175/Lecture15.pdf). For the fixed i.i.d. CLT statement, see [Pishro-Nik, §7.1.2](https://www.probabilitycourse.com/chapter7/7_1_2_central_limit_theorem.php). The rare-event counterexample follows from its explicit zero-count probability.

## Exercises with explanations

1. What is mean variance for p=0.5,n=10? **0.025.**
2. What if n triples? **That variance is divided by three.**
3. Is a small-n normal approximation always accurate near p=0? **No: inspect its error.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
