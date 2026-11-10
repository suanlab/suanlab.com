## Convert a cycle into a tree of clusters

Add diagonal AC to cycle A−B−C−D−A, forming clusters ABC and ACD with separator AC. The original graph has a cycle, but the cluster connection is a tree.

Every original factor must fit within a cluster. Clusters containing the same variable must form a connected subtree: the running-intersection property. An arbitrary tree of arbitrary clusters does not suffice.

## Assign every factor once

Use φ₁(A,B,C)=φAψABψBCe(C) and φ₂(A,C,D)=ψCDψDA. Shared variables do not justify duplicating unary or evidence factors. Doing so changes the model.

## Exchange two messages

Sum out nonshared variables: μ₁₂(A,C)=ΣBφ₁ and μ₂₁(A,C)=ΣDφ₂. Summing their product gives the model/evidence normalizer Z.

$$
b_{ABC}=\phi_1\mu_{21}/Z,\qquad b_{ACD}=\phi_2\mu_{12}/Z,
\quad b_{AC}=\mu_{12}\mu_{21}/Z.
$$

Marginalizing either cluster belief to AC must give the same separator belief. Blue and orange compare those routes. A tiny numerical residual reflects floating-point arithmetic, not two logically different required probabilities.

## Evidence and table interpretation

Observing C=1 gives zero separator mass to rows with C=0. Raw messages need not sum to one; the final two table columns are normalized marginals. With zero field and coupling, all sixteen assignments have weight one, Z=16 and each separator mass is .25.

## Exactness and computation

Valid cluster factorization, running intersection and complete tree message exchange permit exact inference. Large clusters still produce exponentially large tables. Here cluster tables have eight entries and separator tables four. Large treewidth can make exact junction-tree inference expensive.

## Exercises with explanations

1. Should shared variables make a factor appear twice? **No: assign each original factor once.**
2. Must messages be probability distributions? **No: they are separator functions.**
3. How should the separator marginals compare? **They agree within numerical precision.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
