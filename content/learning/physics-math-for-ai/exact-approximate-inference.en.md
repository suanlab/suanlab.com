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

## Check pair beliefs with an exact loop formula

Take the screen's zero-field triangle, but compare BP pair beliefs as well as node marginals. Let J be finite and t=tanh J. Expand each edge as $e^{Js_is_j}=\cosh J(1+ts_is_j)$. Summing spins cancels every term whose selected edges give some node odd degree. On a triangle only the empty subset and the complete cycle have even degree everywhere. Therefore

$$
Z=8\cosh^3(J)(1+t^3),\qquad
E[s_As_B]=\frac13\partial_J\log Z
=\frac{t+t^2}{1+t^3}.
$$

The factor 1/3 uses symmetry and the fact that all three edges share J. It is not a rule for arbitrary unequal couplings.

Uniform initial messages are a BP fixed point at h=0 because Σsᵢ exp(Jsᵢsⱼ)=2cosh J for either recipient spin. The corresponding pair belief is exp(Jsᵢsⱼ)/(4cosh J), with correlation t. For J>0 its gap from the exact correlation is t²(1−t²)/(1+t³)>0. Node marginals can all be .5 and message residual exactly zero while the pair belief remains wrong.

![Three bars distinguish exact edge correlation, BP pair correlation and the product of node means in a zero-field triangle, despite identical node marginals of one half.](/assets/learning/graphical-cycle-correlations-en.svg)

| J | Exact edge correlation | BP pair correlation | Product of node means |
|---|---|---|---|
| .2 | .234529 | .197375 | 0 |
| .7 | .794288 | .604368 | 0 |
| 1.3 | .978293 | .861723 | 0 |

The screen uses the final column as an independence illustration and does not label it BP pair correlation. Here the pair belief is derived separately to distinguish all three quantities. The code checks Z and correlation by enumeration, then compares the pair belief constructed from uniform messages.

## Which accuracy claims follow?

Deleting a tree edge separates two components, enabling induction over subtree sums. In the triangle the other two edges reconnect the endpoints, so that induction fails. The factor 1+t³ is the surviving closed-cycle contribution; pair correlation t alone omits its effect. This example analyzes one fixed point. It does not establish convergence for arbitrary fields, initializations or graphs with multiple cycles.

A convergence claim needs a specified update, norm, initialization and model assumptions. An approximation-accuracy claim needs evidence beyond the fixed-point residual. Exact oracles on small problems provide a research starting point for separating these questions.

**Exercise:** What happens to all three correlations at J=0? **Explanation:** They all vanish because t=0. Their positive-J differences come from distinct formulas, rather than floating-point noise.

```python
from itertools import product
from math import exp, tanh, cosh

for J in [.2,.7,1.3]:
    t = tanh(J)
    states = list(product([-1,1],repeat=3))
    weights = [exp(J*sum(s[i]*s[(i+1)%3] for i in range(3))) for s in states]
    Z = sum(weights)
    corr = sum(w*s[0]*s[1] for w,s in zip(weights,states))/Z
    assert abs(Z/(8*cosh(J)**3*(1+t**3))-1) < 1e-12
    assert abs(corr-(t+t*t)/(1+t**3)) < 1e-12
    # Uniform messages are a fixed point at zero field.
    pair_weights = [exp(J*a*b) for a,b in product([-1,1],repeat=2)]
    pair_corr = sum(w*a*b for w,(a,b) in zip(pair_weights,product([-1,1],repeat=2)))/sum(pair_weights)
    assert abs(pair_corr-t) < 1e-12
    for i in range(3):
        assert abs(sum(w*s[i] for w,s in zip(weights,states))/Z) < 1e-12
    assert corr > pair_corr > 0
    print('J, exact correlation, BP pair correlation, product of means:',
          J,round(corr,6),round(pair_corr,6),0)
```


## Exercises with explanations

1. Does a small residual prove exact inference? **No: it diagnoses iterative change.**
2. Does damping guarantee correctness? **No: stability and accuracy differ.**
3. Do correct node marginals imply an independent joint? **No: correlations can remain.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
