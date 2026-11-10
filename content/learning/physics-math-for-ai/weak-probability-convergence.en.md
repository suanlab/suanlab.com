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

## Exercises with explanations

1. What is L1 distance at n=3 in alternating-sign mode? **Two.**
2. Why does the height-n example fail L1 convergence? **Its L1 distance stays one.**
3. Does CDF mismatch at zero refute weak convergence of δ1/n? **No: zero is a discontinuity of the limiting CDF.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
