## A path does not always mean the same dependence

Compare chain A→B→C, fork A←B→C and collider A→B←C. Conditioning on middle node B blocks chain and fork paths. An unobserved collider blocks its path; conditioning on it or a descendant can open the path. This lab has only three nodes and no added descendant.

In a general DAG, the conditioning set must block every undirected path. A path is blocked by an observed noncollider, or by a collider for which neither it nor any descendant is observed. Blocking just one of several paths does not establish separation.

## Verify with actual tables

Chains and forks use a fair root and independent flip noise ε. Collider roots A,C are independent and fair. B follows noisy OR: probability of B=1 is 1−ε if either parent is one, and ε if both are zero.

At ε=.1, unconditioned A,C masses are all .25. Observing B=1 gives weights (.025,.225,.225,.225), totaling .7. The posterior is (1/28,9/28,9/28,9/28). If one cause is absent, the other becomes more likely to explain the observation.

## Joint versus product of marginals

Blue is the conditional A,C joint and orange the product of conditional marginals. Different bars demonstrate dependence. The readout is mutual information under one selected event, not conditional mutual information averaged over every B value.

$$
p(a,c\mid B=b)\stackrel{?}=p(a\mid B=b)p(c\mid B=b).
$$

## D-separation is a sufficient condition

D-separation guarantees independence for distributions Markov to the graph. The reverse direction needs further conditions such as faithfulness. At ε=.5, interactions ignore their inputs, giving zero information despite an active path. At ε=0, observing B=0 in the collider fixes both parents to zero, another special conditional independence.

Do not identify graphical connectivity with dependence at every parameter value. Causal identification is a separate question too.

## Exercises with explanations

1. Does an unobserved collider open its path? **No: it blocks it.**
2. What happens when B is observed in a chain? **The endpoints are conditionally independent.**
3. Is MI after selecting B=1 averaged over B? **No: it conditions on that one event.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
