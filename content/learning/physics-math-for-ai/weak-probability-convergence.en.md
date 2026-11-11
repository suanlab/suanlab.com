## Coupled-variable distance versus distance between laws

Convergence in probability compares X_n and X on one probability space through deviation probabilities. Weak convergence, or convergence in law, compares their distributions. On the real line it can be characterized by expectations of every bounded continuous test function. If only laws are given, subtracting variables on the same outcome may not yet be defined.

$$
X_n\to X\text{ in probability}\iff P(|X_n-X|>\varepsilon)\to0\ \forall\varepsilon>0,\quad X_n\Rightarrow X\iff E[h(X_n)]\to E[h(X)]\ \forall h\in C_b.
$$

## Rare events and L1 convergence

The indicator of (0,1/n) on Uniform[0,1] has deviation probability and L1 distance 1/n for ε<1, and converges almost surely to zero. Mode two multiplies the indicator by n: convergence in probability remains, but L1 distance stays one. L1 convergence implies convergence in probability, while the reverse fails generally. The readout flags describe analytically established properties of these full sequences, not inferences extracted from a finite plot.

$$
P(|X_n-X|>\varepsilon)\le\frac{E|X_n-X|}{\varepsilon}.
$$

## Equal laws need not mean nearby variables

Let X be an equiprobable sign and set X_n=(−1)^n X. Every term has the same law as X, so weak convergence is immediate. Under this particular coupling, odd terms differ from the same X by two and even terms by zero. At ε=0.5, deviation probabilities alternate between one and zero, preventing convergence in probability to X. For a nonconstant target, law comparison does not determine coupled-path proximity.

$$
\mathcal L(X_n)=\mathcal L(X),\qquad P(|X_{2k+1}-X|>1/2)=1.
$$

## Compare CDFs at continuity points of the limit

The deterministic sequence 1/n converges to zero, but F_n(0)=0 while the limiting point mass has F(0)=1. Weak convergence requires CDF convergence only at continuity points of the limiting CDF. The CDF supremum distance here stays one. Convergence in law to a constant does imply convergence in probability, a special case. General weak convergence alone does not guarantee convergence of expectations of unbounded losses.

$$
F_n(t)\to F(t)\quad\text{at every continuity point }t\text{ of }F.
$$

## Prove that convergence in probability implies weak convergence

Take a bounded continuous test function h, with |h|≤M. Fix a small η>0 and choose R so that P(|X|>R)<η. Continuity makes h uniformly continuous on the compact interval [−R−1,R+1], so choose 0<δ<1 such that values at most δ apart there differ by at most η. On {|X|≤R, |X_n−X|≤δ}, this controls the test-function difference. On its complement use the bound 2M:

$$
|E[h(X_n)]-E[h(X)]|
\le\eta+2M\bigl(P(|X|>R)+P(|X_n-X|>\delta)\bigr).
$$

For this fixed R and δ, convergence in probability sends the final probability to zero. Then let η tend to zero. This proves the bounded-continuous-test definition of weak convergence, without incorrectly assuming that every continuous function is uniformly continuous on the entire real line.

## Prove the special converse for a constant target

Suppose X_n converges in law to a constant c. For any ε>0, choose a=c−ε/2 and b=c+ε/2. Both are continuity points of the limiting CDF. Their limiting CDF values are zero and one, so

$$
P(|X_n-c|>\varepsilon)\le F_n(a)+1-F_n(b)\longrightarrow0.
$$

Thus convergence to a constant also holds in probability. This does not extend to arbitrary nonconstant targets under a specified coupling; the alternating-sign example above shows why. A sequence of laws alone does not select a coupling with a particular target variable.

## Compare weak, total-variation and transport distances

The height-n rare-event example has law μ_n=(1−1/n)δ0+(1/n)δn, with target δ0. For every bounded continuous h, the difference of integrals is (h(n)−h(0))/n, whose magnitude is at most 2||h||∞/n. This directly proves weak convergence.

Define total variation here as sup over measurable sets A of |μ_n(A)−δ0(A)|, equivalently half the sum of absolute mass differences on a discrete space. It equals 1/n: the set {n} attains that difference. The first Wasserstein distance is the infimum of E|U−V| over couplings of the two laws. Since V must equal zero almost surely for every coupling with δ0, the distance is forced to be E|U|=1.

| n | TV | $W_1$ | Mean |
|---|---|---|---|
| 2 | $1/2$ | 1 | 1 |
| 4 | $1/4$ | 1 | 1 |
| 16 | $1/16$ | 1 | 1 |

Both weak convergence and total variation tend to zero here, while the mean and W1 do not approach the target's zero value. These metrics answer different questions about distant rare outcomes. The transport identity was derived for a point-mass target; it is not a general optimal-transport solver.

## Use bounded tests and unbounded losses appropriately

For the bounded continuous test h(x)=x/(1+|x|), the expected value is 1/(n+1)→0. For the unbounded loss |x| it is one. Truncating that loss at a fixed K gives min(n,K)/n→0, but its tail contribution is one whenever n>K. Therefore the family is not uniformly integrable. Passing to a limit for each fixed clipping level and then silently removing clipping is invalid without a uniform tail argument. See [dominated convergence and uniform integrability](../convergence-dominated-integration/) for a sufficient L1 criterion.

## Verify finite laws exactly

The code checks total variation, CDF distance, point-target transport cost, the bounded test and clipped/tail losses using fractions. The formulas above prove the full-sequence statements. The interactive controls still compare the original four sequences; transport and clipping calculations are separate extensions.

```python
from fractions import Fraction as Q

for n in [2, 4, 16, 64]:
    law = {0: 1-Q(1, n), n: Q(1, n)}
    target = {0: Q(1)}
    support = sorted(set(law) | set(target))
    mean = sum(Q(x)*law.get(x, 0) for x in support)
    tv = sum(abs(law.get(x, 0)-target.get(x, 0)) for x in support)/2
    # Every coupling with delta_0 has second coordinate zero.
    w1 = sum(abs(Q(x))*law.get(x, 0) for x in support)
    cdf_gap = max(abs(sum(m for x, m in law.items() if x <= t)
                      -sum(m for x, m in target.items() if x <= t))
                  for t in support)
    assert tv == cdf_gap == Q(1, n) and mean == w1 == 1
    bounded_test = sum(Q(x, 1+abs(x))*m for x, m in law.items())
    assert bounded_test == Q(1, n+1)
    for K in [1, 4, 16]:
        truncated = sum(min(Q(x), Q(K))*m for x, m in law.items())
        tail = sum(Q(x)*m for x, m in law.items() if x > K)
        assert truncated == Q(min(n, K), n)
        assert tail == int(n > K)
    print("n, TV, W1, mean, bounded test:", n, tv, w1, mean, bounded_test)
```

## Further questions and research use

1. Does very small total variation guarantee small error for every unbounded loss? **No: the table's expected absolute loss stays one.**
2. Can CDF supremum distance disagree with W1? **Yes: it is 1/n here while W1 is one.**
3. What should a distribution-shift claim specify? **The distance or test-function class, its relation to the task loss and the required tail moments.**

For weak convergence and its test functions, see [Durrett, §3.2](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf). The two implications and the finite rare-event distance calculations above are derived explicitly.

## Exercises with explanations

1. What is L1 distance at n=3 in alternating-sign mode? **Two.**
2. Why does the height-n example fail L1 convergence? **Its L1 distance stays one.**
3. Does CDF mismatch at zero refute weak convergence of δ1/n? **No: zero is a discontinuity of the limiting CDF.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
