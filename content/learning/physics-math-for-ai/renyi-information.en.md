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

## Exercises with explanations

1. What is H₀ with one positive state? **log₂1=0.**
2. What is collision probability? **The probability that two independent draws match.**
3. Is order eight exactly infinite order? **No: distinguish a finite order from its limit.**

Further reading: [van Erven & Harremoës, Rényi Divergence and Kullback–Leibler Divergence](https://arxiv.org/abs/1206.2459).
