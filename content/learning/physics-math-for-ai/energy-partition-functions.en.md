## Energy as a negative-log probability form

Take three spins sᵢ in {−1,1} connected in a triangle. Coupling J and common field h define energy; temperature T>0 turns it into exp(−E/T) weights. This dimensionless example uses the convention kB=1.

$$
E(s)=-J(s_As_B+s_Bs_C+s_Cs_A)-h(s_A+s_B+s_C),\quad
p(s)=e^{-E(s)/T}/Z.
$$

## Sum all states for the partition function

Enumerate eight states to compute Z=Σs exp(−E(s)/T). At J=h=0 every energy is zero, Z=8 and each probability is 1/8. Z is a weight sum rather than a probability and may exceed one.

Numerically subtract maximum log weight M, then restore it using logZ=M+logΣexp(logw−M). Summing displayed relative weights alone does not recover the original Z.

## Low temperature and symmetry

Positive J favors aligned neighbors. At h=0, flipping every spin preserves energy, so mean total spin is exactly zero. Low temperature concentrates on ground states such as +++ and ---. Zero mean spin does not imply a uniform joint distribution.

## Energy expectations and entropy

Average energy and total spin under normalized probabilities. Averaging log p=−E/T−logZ gives entropy E[E]/T+logZ in nats. Parameter derivatives must retain the temperature factor.

$$
H(p)=-\sum_s p(s)\log p(s)=E[E]/T+\log Z,\quad
\frac{\partial\log Z}{\partial h}=E[\sum_i s_i]/T.
$$

## Connections to larger models

Computing Z can require exponentially many states. Energy-based learning must consider normalization changes as well as observed-state scores. Exact enumeration available for eight states is not automatically affordable for large neural energy models. Sampling and variational approximations bring separate accuracy and mixing conditions.

## Exercises with explanations

1. How is the removed maximum restored in logZ? **Add it back.**
2. What is mean total spin at h=0? **Zero by flip symmetry.**
3. Does low temperature always select one unique state? **No: ground states can be multiple.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
