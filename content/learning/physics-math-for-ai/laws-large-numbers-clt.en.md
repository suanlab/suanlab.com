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

## Exercises with explanations

1. What is mean variance for p=0.5,n=10? **0.025.**
2. What if n triples? **That variance is divided by three.**
3. Is a small-n normal approximation always accurate near p=0? **No: inspect its error.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
