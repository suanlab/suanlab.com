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

## Experiment and exercises

Switch representations and verify identical joint bars despite a different Z. At r=.5, downstream variables lose information about their parents. At r=1 only 000 and 111 remain.

1. Must an MRF factor sum to one? **No: normalize the whole product using Z.**
2. Does s=4 multiply probability of 111 by sixteen? **No: the denominator cancels the scale.**
3. Does edge length encode strength? **No: it only positions the diagram.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
