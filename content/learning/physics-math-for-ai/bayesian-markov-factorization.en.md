## Represent a distribution as local products

A joint table for three binary variables has eight assignments. A graphical model replaces that large table with local functions. A Bayesian network A→B→C factorizes as p(a)p(b|a)p(c|b). Each conditional row sums to one; summing from the last variable proves normalization of the full joint.

$$
p(a,b,c)=p(a)p(b\mid a)p(c\mid b).
$$

## Multiply a concrete example

Let P(A=1)=.3 and each child agree with its parent with probability r=.8. Assignment 111 has probability .3×.8×.8=.192; 000 has .7×.8×.8=.448. Summing rows with C=1 gives .428, including paths where two flips cancel.

## An undirected factorization of the same law

An undirected representation multiplies unary φA with pair factors ψAB and ψBC, then divides by their full sum Z. Here φA=p(a), and each edge factor is a conditional agreement table multiplied by s. Both raw weights and Z scale by s², leaving probabilities unchanged.

$$
p(a,b,c)=\frac{\phi_A(a)\psi_{AB}(a,b)\psi_{BC}(b,c)}Z,
\qquad Z=\sum_{a,b,c}\phi_A\psi_{AB}\psi_{BC}.
$$

At s=2, Z=4; at s=4, Z=16. Factor values need not be probabilities. This is a deliberately constructed equivalence for one chain, not a claim that arbitrary directed and undirected representations convert without changing structure.

## Independence, positivity and causality

Here A and C are conditionally independent given B. General equivalence results between undirected Markov properties and clique factorization require positivity assumptions. Boundary controls allow zero factors, so do not invoke a positive-distribution theorem unconditionally.

A probabilistic DAG arrow alone establishes neither a causal effect nor an intervention distribution. This chapter models observational factorization. Causal interpretation needs additional assumptions about data collection, direction and confounding.

## Why a general DAG normalizes

Assume finite variables and nonnegative conditional tables whose rows sum to one for every parent assignment. Choose a sink v, a node with no children. The only factor involving its variable is its own conditional table. Summing x_v therefore removes that factor. Repeat on the remaining DAG. Every finite DAG has a sink, so induction proves that the whole product sums to one. Strictly positive probabilities are unnecessary for this argument.

$$
\sum_{x_v}\prod_i p(x_i\mid x_{\mathrm{pa}(i)})
=\prod_{i\ne v}p(x_i\mid x_{\mathrm{pa}(i)}).
$$

Normalized local tables alone do not justify the same construction on a directed cycle. For binary variables, multiplying p(a|b)=1{a=b} and p(b|a)=1{b=a} gives weight one to both 00 and 11, with total two. There is no sink-removal proof for this cyclic product.

## Removing arrowheads can impose a false independence

A collider A→B←C factorizes as p(a)p(c)p(b|a,c). Its last factor involves all three variables. Preserving that factor's scope in an undirected representation requires a clique containing A,B,C. Keeping only A−B−C would instead impose A⊥C|B, which a noisy-OR collider generally violates. This explains the added edge between parents during moralization.

In the next lesson's ε=.1,B=1 example, P(A=1,C=1|B)=9/28, while both marginals are 9/14. Their product is 81/196, whereas 9/28 is 63/196. All eight original joint masses are positive, so zeros do not explain the discrepancy. The current chain conversion is consequently not a structure-preserving recipe for every DAG.

**Exercise:** Does dividing the two-variable cyclic product by Z create a model? **Explanation:** Yes: Z=2 gives a valid undirected joint law. It still does not supply the acyclic generative order required by the BN normalization proof.

## Experiment and exercises

Switch representations and verify identical joint bars despite a different Z. At r=.5, downstream variables lose information about their parents. At r=1 only 000 and 111 remain.

1. Must an MRF factor sum to one? **No: normalize the whole product using Z.**
2. Does s=4 multiply probability of 111 by sixteen? **No: the denominator cancels the scale.**
3. Does edge length encode strength? **No: it only positions the diagram.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
