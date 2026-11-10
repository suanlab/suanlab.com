## A message summarizes the rest of a subtree

On a tree, message i→j summarizes the i-side subtree as a function of j. Multiply i's unary with messages from its other neighbors, then sum i out. Exclude the recipient's incoming message to avoid sending its information straight back.

$$
m_{i\to j}(x_j)\propto\sum_{x_i}\phi_i(x_i)\psi_{ij}(x_i,x_j)
\prod_{k\in N(i)\setminus j}m_{k\to i}(x_i).
$$

Positive message scaling cancels in final normalization. The implementation normalizes each two-state message for stability.

## Space and rounds in this example

The tree is A−B−C−D, with field h only at A. Initial messages are (.5,.5). A synchronous round computes every new message from the previous round. Information travels one edge per round, requiring three rounds for A's influence to reach D.

## Current beliefs versus exact marginals

Normalize a node's unary times all incoming messages. Blue shows current beliefs; orange enumerates the exact sixteen-state joint. At J=1,h=.7 after one round, C,D still have belief .5 while exact probabilities are about .6753 and .6335.

$$
b_i(x_i)\propto\phi_i(x_i)\prod_{k\in N(i)}m_{k\to i}(x_i).
$$

Three rounds communicate the entire tree, producing agreement within floating-point precision. A directed tree schedule can instead calculate each directional message once. Synchronous round count and scheduled message-operation count are different measures.

## Conditions and objectives

Exactness relies on tree structure and exact sum-product operations. The same repeated update on a cycle has different guarantees. Choosing each node's marginal mode also need not produce a joint MAP assignment; distinguish max-product and appropriate backtracking.

## Exercises with explanations

1. Is j's incoming message used when sending back to j? **No.**
2. Does a small local error after one round certify all information has propagated? **No: inspect paths from distant observations.**
3. What happens at J=0? **Variables are independent and initial local beliefs are already exact.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
