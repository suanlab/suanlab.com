## Why an equality needs a license

An intervention expression is not turned into an observational expression by deleting the word do. A valid transformation needs assumptions about which paths remain after particular arrows are removed. The lab distinguishes three objects: the original structural model, the modified graph used to test a rule, and the distributions used on the rule's two sides. The criterion graph is not automatically the data-generating graph for either side.

Let X be held fixed, Y the outcome, Z the variable to remove or exchange, and W optional observed evidence. These node sets are disjoint. All conditional expressions below must be defined. The lab uses four binary nodes and six graph families; it illustrates the rules rather than implementing a general causal-identification search engine.

## The three graphical conditions

Write G with a bar over a node set for deletion of incoming arrows, and an underline for deletion of outgoing arrows. Apply each separation test in its specified graph:

| Rule | Transformation | Criterion graph |
|---|---|---|
| 1 | Delete observed Z | G with inputs to X removed |
| 2 | Exchange action on Z with observation of Z | Also remove outputs from Z |
| 3 | Delete action on Z | Also remove inputs to Z(W) |

Here Z(W) contains only Z nodes that are **not ancestors of W in G with inputs to X removed**. It is all of Z when W is empty. Each rule requires Y and Z to be d-separated given X,W in that criterion graph.

$$
\begin{aligned}
P(y\mid do(x),z,w)&=P(y\mid do(x),w),\\
P(y\mid do(x),do(z),w)&=P(y\mid do(x),z,w),\\
P(y\mid do(x),do(z),w)&=P(y\mid do(x),w).
\end{aligned}
$$

These are sufficient graph conditions for the respective equalities. Failure of a criterion does not by itself prove that two numerical values differ at every parameter setting.

## Reading paths instead of just correlations

For each simple undirected path between Y and Z, inspect its interior nodes. A non-collider blocks the path when conditioned on. A collider blocks it unless the collider or a descendant is conditioned on. The table labels paths active or blocked after the rule-specific surgery. If every path is blocked, the displayed graphical license is one. That conclusion comes from edges and conditioning, independent of the mechanism-strength slider.

The lab's structural assignments use independent uniform noises and conditional success probabilities

$$
q_v=0.2+s\,\operatorname{mean}(\text{parent bits}),\qquad 0\le s\le0.6.
$$

Root nodes have success probability 0.2. Interventions replace selected assignments by constants. All other mechanisms retain probabilities between 0.2 and 0.8. Sixteen endogenous states are summed exactly; no simulation noise or significance test determines a rule's license.

## Worked examples: exchange and ancestry

In the fork W→Z and W→Y, with W unobserved, rule 2 fails its criterion: deleting Z's outgoing arrows leaves the common-cause path open. At s=0.6 and query Z=1, the intervention mean is 0.32 while the observed mean is 0.50. Conditioning on W blocks the path and licenses exchange. Set W=1 to obtain mean 0.8 on both sides.

For rule 3, choose Y→Z→W and condition on W=1. Z is an ancestor of W, so Z(W) is empty. Deleting Y→Z anyway would give an incorrect criterion graph. The correct graph keeps the active Y→Z path. At s=0.6 and do(Z=1), the left mean is 0.2. Without that action, conditioning on W=1 gives 0.136/0.392, approximately 0.347. The action changes what the selected W group tells us about Y.

## Exercise and explained answer

Keep the fork without conditioning on W, choose rule 2, then set strength to zero. Does agreement of the means establish the graph condition?

**Answer.** Both means become 0.2 because every structural success probability is now 0.2. The common-cause path remains present in the declared graph, so its criterion still fails. Numerical cancellation or an inactive mechanism at one setting cannot establish a rule valid for the entire graph family. The blue and orange curves show equality at zero and disagreement at positive strengths.

## From a rule to an identification argument

A research derivation may apply several licensed transformations until the target expression contains only available distributions. The present tool checks local rules in declared finite models; it does not certify an entire study's graph, measurement quality, or support. Record the starting causal question, each graph surgery, the conditioning set, and the available data distribution for every step.

## Primary reference

Judea Pearl, [A Causal Calculus for Statistical Research](https://proceedings.mlr.press/r0/pearl95a.html), 1995, section 4. The finite graph families, numerical mechanisms and worked calculations here are original teaching examples.
