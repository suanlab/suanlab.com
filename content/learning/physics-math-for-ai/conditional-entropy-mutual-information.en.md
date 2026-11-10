## Uncertainty removed by an observation

Entropy describes average uncertainty about X before observation. After observing Y, calculate the distribution of X for each y, then average its entropy using probabilities of Y. The reduction from the original entropy is mutual information.

$$
H(X\mid Y)=\sum_y p(y)H(X\mid Y=y),\qquad I(X;Y)=H(X)-H(X\mid Y).
$$

## A noisy binary observation

Let X~Bernoulli(p) and add independent Bernoulli(ε) noise by XOR: Y=X⊕N. The table contains four joint masses, each input mass times a conditional channel probability. Summing a column gives p(y); normalizing that column gives p(x|y).

At p=.5 and ε=.1, diagonal masses are 0.45 and off-diagonal masses 0.05. Observing Y=1 gives probability 0.9 that X=1. H(X)=H(Y)=1 and H(X|Y)=h₂(.1)≈0.469, so I≈0.531 bits.

## Check through joint entropy

For general p, output success probability is p(1−ε)+(1−p)ε. Knowing X leaves only noise uncertainty, giving H(Y|X)=h₂(ε). Conditional entropies can differ by direction, while mutual information is symmetric.

$$
H(X,Y)=H(X)+H(Y\mid X),\qquad I(X;Y)=H(X)+H(Y)-H(X,Y)=H(Y)-h_2(\varepsilon).
$$

## Perfect inversion preserves information

At ε=.5, output and input are independent and I=0. At ε=1, output always flips, but X=1−Y recovers the input exactly, giving I=H(X). Raw-label agreement is different from information. Blue plots mutual information against noise; orange marks input entropy as an upper bound.

A deterministic input p=0 or one contains no new source information. Noise can make H(Y)>0 while I(X;Y)=0. Output variability alone does not establish a useful representation.

## Null conditioning and research interpretation

A column with p(y)=0 has no ratio-defined conditional probability and is shown as a dash. Its weight in conditional entropy is zero. High mutual information also does not establish causation or task usefulness. State the variables, distribution and observation mechanism being analyzed.

## Exercises with explanations

1. What is mutual information under independence? **Zero.**
2. Must H(Y|X) equal H(X|Y)? **No: conditioning directions differ.**
3. How much source information can be gained about a deterministic input? **Zero.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
