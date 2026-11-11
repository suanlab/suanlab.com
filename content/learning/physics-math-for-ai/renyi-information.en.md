## A family indexed by order

Beyond Shannon entropy, information measures can use powers of masses. Rényi entropy changes the emphasis on rare and dominant outcomes through order α. This lab uses finite binary distributions, α in [0,8], and bits.

$$
H_\alpha(P)=\frac1{1-\alpha}\log_2\sum_i p_i^\alpha,\qquad \alpha\ne1.
$$

## Distinguish zero, one, two and infinite orders

Order zero is log positive-support size. Order one continuously extends to Shannon entropy. Order two is minus log collision probability for two independent draws. At infinite order, only the largest mass remains, giving min entropy. Its value is reported separately; the plotted order range ends at eight.

$$
H_0=\log_2|\operatorname{supp}P|,\quad H_1=H(P),\quad
H_2=-\log_2\sum_i p_i^2,\quad H_\infty=-\log_2\max_i p_i.
$$

At p=.8, values are H₀=1, H₁≈.72193, H₂≈.55639 and H∞≈.32193 bits. Entropy is nonincreasing with order. A uniform distribution has equal values at every order; a deterministic one gives zero. A tiny positive mass still counts toward order-zero support.

## Compare Rényi divergence

Differences from Q use the following divergence. Both Q masses are restricted to be positive, separating support singularities from order limits. Order one is KL; order zero is minus log Q mass on P support.

$$
D_\alpha(P\Vert Q)=\frac1{\alpha-1}\log_2\sum_i p_i^\alpha q_i^{1-\alpha},
\qquad D_0=-\log_2Q(\operatorname{supp}P).
$$

At p=.8,q=.5,α=2, D₂≈.44361 bits. Blue plots entropy against order and orange the Shannon reference. Compare divergences in the table and readout. Entropy and divergence are different quantities.

## Limits and computation

Substitution of α=1 into the original expression gives zero divided by zero. The implementation uses Shannon/KL at one and stable expm1/log1p expressions nearby. At order zero, it counts positive support rather than assigning every zero mass the value 0⁰=1. This introduction covers Rényi entropy and divergence; it does not identify all distinct definitions of Rényi mutual information with one another.

## Order-two divergence is a weight second moment

On a finite alphabet, let normalized masses p_i,q_i both be positive. Sampling iid under Q with weights w(i)=p_i/q_i gives E_Q[w]=1. Since order-two Rényi divergence is measured in bits,

$$
E_Q[w^2]=\sum_i\frac{p_i^2}{q_i}=2^{D_2(P\Vert Q)},\qquad
\operatorname{Var}_Q(w)=2^{D_2(P\Vert Q)}-1.
$$

D₂ is therefore the logarithm of a weight second moment. The ordinary normalizer estimate N^{-1}Σw is unbiased, with variance (2^D₂−1)/N. This is different from the variance of an estimate of a function f's mean; that error involves f, as in the earlier importance-sampling quantities V and A.

## Separate sample ESS from a population coefficient

For realized weights, ESS=(Σw)²/Σw² lies between one and N. Holding P,Q fixed, laws of large numbers send sample means of w and w² to one and E_Q[w²], giving

$$
\frac{\operatorname{ESS}_N}{N}
=\frac{(N^{-1}\sum_s w_s)^2}{N^{-1}\sum_s w_s^2}
\longrightarrow 2^{-D_2(P\Vert Q)}\quad\text{almost surely}.
$$

Finite alphabets and positive Q give finite moments here. General continuous-state problems require those conditions to be checked again. N·2^(−D₂) is not the expected ESS for finite N. A weight diagnostic alone also does not certify visited coverage or accuracy for a particular f.

## Repeated coordinate mismatch accumulates

Use P=(4/5,1/5), Q=(1/2,1/2) in each of d coordinates, independent under both distributions. Joint weights multiply across coordinates, so their second moment under Q factorizes.

$$
E_{Q^{\otimes d}}[w^2]=(34/25)^d,\qquad
D_2(P^{\otimes d}\Vert Q^{\otimes d})=d\log_2(34/25).
$$

The last column below holds d fixed and sends N to infinity. It is not an expected finite-sample ESS fraction.

| d | D₂ bits | Limit ratio |
|---|---|---|
| 1 | 0.44361 | 0.73529 |
| 10 | 4.43607 | 0.04620 |
| 20 | 8.87213 | 0.00213 |

Exponential moment growth here follows from repeating the same mismatch independently. Dimension alone does not force this particular degradation: if P=Q, divergence is zero at every d. Proposals whose coordinate fit changes with d require separate analysis.

## Compute expected ESS exactly for two draws

At d=1,N=2, enumerate four ordered pairs. Equal weights give ESS=2; unequal weights give ESS=25/17. Each category has probability one half, so expected ESS is 59/34≈1.73529. Multiplying the population limit coefficient by N instead gives 25/17≈1.47059, a different value. For N=2, ESS/N is always at least one half, so the d=10 population ratio approximately 0.0462 cannot be substituted as that sample fraction.

The code enumerates full states and two-draw pairs at d=1,2,3 with fractions, then computes higher-dimensional moments using exact powers. The existing controls concern a single binary coordinate and order α, a separate experiment.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod, log2

p, q = [Q(4, 5), Q(1, 5)], [Q(1, 2), Q(1, 2)]
base = sum(p[i]**2/q[i] for i in range(2))
assert base == Q(34, 25)
for d in [1, 2, 3]:
    states = list(product(range(2), repeat=d))
    masses = [prod(q[i] for i in state) for state in states]
    weights = [prod(p[i]/q[i] for i in state) for state in states]
    assert sum(a*w for a, w in zip(masses, weights)) == 1
    second = sum(a*w*w for a, w in zip(masses, weights))
    assert second == base**d
    expected_ess = sum(masses[i]*masses[j]*(weights[i]+weights[j])**2
                       /(weights[i]**2+weights[j]**2)
                       for i, j in product(range(len(states)), repeat=2))
    if d == 1:
        assert expected_ess == Q(59, 34)
        assert expected_ess != 2/second
    print("d, exact expected N=2 ESS, population coefficient times two:",
          d, expected_ess, 2/second)
for d in [1, 10, 20]:
    print("d, D2 in bits, limiting ESS/N:", d, d*log2(float(base)), float(1/base**d))
```

## Questions for research use

1. Does D₂ determine error for every target function? **No: distinguish weight moments from function-specific error.**
2. Is N·2^(−D₂) an expected finite-sample ESS? **No: it is based on a large-N coefficient for fixed distributions.**
3. What should a high-dimensional experiment record? **Coordinate dependence, proposal mismatch, support, moment conditions and sample size.**

For the Rényi and χ² connection, see [van Erven–Harremoës, §I-B](https://arxiv.org/html/1206.2459v2#S1.SS2). The weight identities, product law and finite ESS enumeration are derived directly for the stated distributions.

## Exercises with explanations

1. What is H₀ with one positive state? **log₂1=0.**
2. What is collision probability? **The probability that two independent draws match.**
3. Is order eight exactly infinite order? **No: distinguish a finite order from its limit.**

Further reading: [van Erven & Harremoës, Rényi Divergence and Kullback–Leibler Divergence](https://arxiv.org/abs/1206.2459).
