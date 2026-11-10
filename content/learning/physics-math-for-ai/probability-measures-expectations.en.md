## Probability is a normalized measure

A probability measure is a measure of total mass one. Expectation is Lebesgue integration with respect to that measure. On finite spaces this is a weighted sum; continuous and other spaces use the same definition through appropriate measures. The default law has weights (0.1,0.2,0.3,0.4), while X sends outcomes to (0,0,1,2). Probabilities and numerical outputs play different roles.

$$
E_P[X]=\int_\Omega X(\omega)\,P(d\omega).
$$

## A law is a pushforward through preimages

Define the law of X by the probability of each output set’s preimage. Two default outcomes map to zero, so their masses add to 0.3. Outputs one and two have masses 0.3 and 0.4. Masses mapping to the same output are added, not duplicated or averaged. Measurability matters because the preimage must be an event to which the original probability measure applies.

$$
P_X(B)=P(X^{-1}(B)),\qquad P_X(\{0\})=0.1+0.2=0.3.
$$

## Compute expectation in either space

On the original space, sum 0·0.1+0·0.2+1·0.3+2·0.4=1.1. On the output space, 0·0.3+1·0.3+2·0.4 gives the same answer. For general nonnegative measurable h, prove the identity first for indicators, then simple functions and monotone limits. Integrable signed functions follow by splitting positive and negative parts. No inverse function for X is required.

$$
E_P[h(X)]=\int_{\mathbb R}h(x)\,P_X(dx).
$$

## Read variance, signs and collapsed laws

The default second moment is 1.9 and variance is 1.9−1.1²=0.69. Signed mode groups outputs minus one and one, giving expectation 0.2. Scale zero collapses the law to a unit atom at zero. Check absolute integrability when asserting a finite real expectation. These bounded finite examples are integrable, but a few computed sample averages would not prove existence of expectation for an arbitrary distribution.

$$
\operatorname{Var}(X)=E[X^2]-(E[X])^2,\qquad E|X|<\infty\text{ ensures a finite expectation}.
$$

## Exercises with explanations

1. Are masses sharing an output added? **Yes: their preimage events are disjoint.**
2. Does every pushforward require a Jacobian formula? **No: the preimage definition is more general.**
3. Is a zero-mass listed output automatically in the finite support? **No: distinguish it from positive-mass outputs.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
