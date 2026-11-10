## Same update, different graph

Repeating tree sum-product on a triangle gives loopy belief propagation. It can be useful, but tree exactness arguments do not transfer automatically. Convergence, speed, marginal accuracy and joint accuracy are separate properties.

## Build a tiny exact oracle

Three binary variables allow enumeration of eight states. Blue plots maximum marginal error against that exact reference. Larger problems often use approximation precisely because the oracle is unavailable, so this error display is not generally obtainable.

## Message change is not error against truth

Orange residual is maximum change between consecutive messages. A small residual suggests proximity to a fixed point, not correctness of that fixed point. Default positive coupling and field at A can produce beliefs more confident than the exact marginals even when updates become small.

Damping mixes old and proposed messages: mnew=(1−η)mold+η mproposal. It preserves normalization of positive messages but does not universally guarantee convergence or accuracy. At iteration zero there is no previous update, so residual is undefined and displayed as a dash.

$$
r_t=\max_{i\to j,x}|m^{(t)}_{i\to j}(x)-m^{(t-1)}_{i\to j}(x)|,\qquad e_t=\max_i|b_i^{(t)}(1)-p(X_i=1)|.
$$

## Matching marginals need not recover the joint

At h=0, spin-flip symmetry gives node marginals .5. BP can match these marginals, while exact neighboring spin correlation remains positive. Multiplying zero node-spin means gives zero and discards dependence. That product is a separate independence illustration, not a claim about BP pair beliefs.

## Audit assumptions by algorithm

Enumeration is exact but exponential. VE and junction trees can be efficient when intermediate scopes are small. Tree BP is exact after appropriate completion. Loopy BP needs model-specific stability or contraction analysis. Monte Carlo convergence and errors depend on additional sampling and mixing conditions. Algorithm names alone do not establish guarantees.

## Exercises with explanations

1. Does a small residual prove exact inference? **No: it diagnoses iterative change.**
2. Does damping guarantee correctness? **No: stability and accuracy differ.**
3. Do correct node marginals imply an independent joint? **No: correlations can remain.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
