## Specify a conditional transition law

A Markov model states that, given the current state, its next transition does not require the earlier history. This is a statement about conditional probabilities, not a claim that neighboring states are independent. The experiment uses states zero and one, with a the probability of moving from zero to one and b the probability of moving back.

$$
P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix},\qquad
p_{n+1}=(1-p_n)a+p_n(1-b).
$$

Here p_n means the marginal probability of state one. Rows of P are conditional laws, while a vector of state probabilities describes uncertainty about the current state. Mixing these objects can produce a matrix calculation with the correct shape but the wrong interpretation.

## Iterate the law rather than drawing convergence

The discrete plot uses actual probability updates. With a=0.2, b=0.4 and initial state one, probabilities evolve 1, 0.6, 0.44. A stationary probability, when a+b is positive, is a/(a+b), but the presence of that fixed law alone does not guarantee convergence in every parameter case.

The upper diagram labels arrows with the selected-horizon matrix P^n and nodes with the current law. At n=0 the horizon matrix is the identity. These arrows therefore represent the displayed interval rather than always representing one elementary step. The later ergodicity chapter examines periodic and reducible boundaries in detail.

## Continuous time needs a generator

In continuous mode the same a,b controls are jump rates with inverse-time units, rather than per-step probabilities:

$$
Q=\begin{pmatrix}-a&a\\b&-b\end{pmatrix},\qquad P(t)=e^{tQ}.
$$

For positive a+b, the state-one probability is pi1+(p0-pi1)exp(-(a+b)t). The code evaluates the transition matrix using expm1 near time zero. If both rates vanish, every state is absorbing and P(t) remains the identity. No unique stationary law is invented for this case.

## Reverse probabilities require support

For a selected horizon kernel K and initial law p, Bayes gives R_ji=p_i K_ij/pnext_j. This reverses the joint law of the two endpoint states. Where pnext_j is zero, that conditional row is not determined by the joint law. The table leaves it absent instead of dividing by zero or filling an arbitrary row that appears certified.

## Exercise: three different matrices

Use a=b=1. In discrete time the states alternate, so P² is the identity. In continuous time these numbers are rates: at time one the probability of remaining in state one is (1+exp(-2))/2, not zero. Explain why equating a generator with a finite-time transition fails. Finally set a=b=0 and initial state one; identify the reverse row that cannot be conditioned on and the family of stationary distributions.

Further reading: [MIT stochastic-process notes](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/). The two-state calculations are independently derived.
