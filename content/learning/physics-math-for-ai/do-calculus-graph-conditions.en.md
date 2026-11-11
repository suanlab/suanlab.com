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

## Compose local graphical licenses into a front-door functional

Consider an auxiliary graph with hidden U causing X and Y, X→M→Y, and no direct X→Y edge. Assume M intercepts every directed X-to-Y path, no unblocked backdoor path connects X to M, and X blocks all backdoor paths from M to Y. Required observational conditionals must also have support.

First identify P(m|do(x))=P(m|x). Next adjust the M-to-Y intervention using X: P(y|do(m))=sum_x' P(y|m,x')P(x'). Because the X effect travels only through M and the relevant intervention blocks the confounding route, the licensed substitutions give

$$
P(y\mid do(x))=\sum_m P(m\mid x)\sum_{x'}P(y\mid m,x')P(x').
$$

The inner average is over the population distribution of X, not P(X|M=m). Reusing the latter would retain mediator-selection information and collapse the expression toward an observational quantity. Draw and check each surgery; a graph with a direct X→Y path or hidden X–M confounding need not support this functional.

In the binary example, U is fair, treatment propensities are (1/10,9/10), P(M=1|X)=1/5+3X/5, and P(Y=1|M,U)=1/10+2M/5+3U/10. The exact observed joint identifies both intervention means through the front-door expression, although the crude contrast is twice the true effect.

| Query | Observed mean or contrast | Front-door mean or contrast |
|---|---:|---:|
| X=0 | 21/100 | 33/100 |
| X=1 | 69/100 | 57/100 |
| Difference | 12/25 | 6/25 |

```python
from fractions import Fraction as F
from itertools import product
bit=lambda p,b:p if b else 1-p
joint={key:F(0) for key in product((0,1),repeat=3)}
for u,x,m,y in product((0,1),repeat=4):
    e=[F(1,10),F(9,10)][u];pm=F(1,5)+F(3,5)*x;py=F(1,10)+F(2,5)*m+F(3,10)*u
    joint[x,m,y]+=F(1,2)*bit(e,x)*bit(pm,m)*bit(py,y)
px=lambda x:sum(joint[x,m,y] for m,y in product((0,1),repeat=2))
pym=lambda x,m:joint[x,m,1]/sum(joint[x,m,y] for y in (0,1))
observed=[];front=[]
for x in (0,1):
    observed.append(sum(joint[x,m,1] for m in (0,1))/px(x))
    front.append(sum(sum(joint[x,m,y] for y in (0,1))/px(x)*sum(pym(t,m)*px(t) for t in (0,1)) for m in (0,1)))
    oracle=F(1,10)+F(2,5)*(F(1,5)+F(3,5)*x)+F(3,20)
    assert front[x]==oracle
    print(x,observed[x],front[x])
assert observed[1]-observed[0]==F(12,25) and front[1]-front[0]==F(6,25)
```

This is a composed derivation for a different three-observed-node graph, not a new graph preset or a general identification engine. The live four-node tool continues to test each of its three local rules, including the Z(W) ancestor restriction. Equality in one numerical realization cannot replace the graphical proof, and graphical identification does not certify a study's measurement or finite-sample estimation.

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
