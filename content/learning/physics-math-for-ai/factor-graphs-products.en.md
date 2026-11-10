## Why introduce square nodes?

A factor graph is bipartite: circles represent variables, squares functions. Connect a variable to a factor precisely when it belongs to that factor's arguments. Different groupings of factors can represent the same joint law.

This lab sets sᵢ=2xᵢ−1 and multiplies exp(hsA), exp(JsAsB) and exp(JsBsC), with scopes {A},{A,B},{B,C}. Edge lengths and node positions do not encode numerical strength.

## From local values to probability

Fix an assignment, evaluate each function and multiply. At h=.4,J=.7, assignment 111 has weight exp(.4) × exp(.7) × exp(.7) = exp(1.8) ≈ 6.04965. A raw weight above one is valid. Summing weights over eight assignments gives Z≈13.6254 and normalized probability about .444.

$$
\begin{aligned}w(a,b,c)&=e^{hs_A}e^{Js_As_B}e^{Js_Bs_C},\\p(a,b,c)&=w(a,b,c)/Z.\end{aligned}
$$

## Product scopes and summation scopes

Multiplying factors takes the union of their scopes. The product of ψAB and ψBC depends on A,B,C. Summing out B produces a new A,C factor, generally an unnormalized function rather than a probability distribution.

$$
\tau(a,c)=\sum_b\psi_{AB}(a,b)\psi_{BC}(b,c).
$$

Variable elimination and message passing repeatedly apply these operations to avoid constructing the entire joint table at once.

## Coupling and independence

Positive J favors matching neighboring spins; negative J favors opposite spins. At J=0, pair factors equal one. If h also vanishes, every assignment is equally weighted and Z=8. A drawn interaction can become ineffective at special parameter values.

## Experiment and exercises

Read mask bits from lowest to highest as A,B,C. Compare the orange point with the table and multiply the three factors manually. The selected mask changes the highlighted assignment, not the model.

1. What is the product scope of ψAB and ψBC? **{A,B,C}.**
2. Does summing B always give a constant? **No: the result can depend on A,C.**
3. Do edges here claim causal direction? **No: they describe undirected computational factorization.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/ve/).
