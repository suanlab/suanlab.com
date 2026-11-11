## Expectation differs from an every-instance guarantee

A randomized algorithm must specify its probability space. An approximation algorithm must specify the optimum used in its ratio. This experiment uses six labeled vertices with triangle, star, cycle or empty edges. Random cuts and matching-based vertex covers solve different optimization problems. Their purpose here is to contrast two valid forms of analysis on an inspectable graph.

Blue is the exact PMF from all 64 independent uniform bit assignments. Orange highlights the PMF entry at one seeded sample score. The distribution is not estimated from that single sample, and the highlighted value is not assigned probability one. The upper drawing uses a separately selected cut mask, which need not equal the sample mask.

## Expected random-cut performance

Assign each vertex an independent fair bit. An edge crosses with probability one half. Linearity of expectation does not require all edge events to be jointly independent:

$$
C=\sum_{e=1}^m I_e,\qquad E[C]=m/2\ge\mathrm{OPT}_{cut}/2.
$$

This gives a one-half guarantee in expectation, not a guarantee that every sample attains half the optimum. A triangle can receive identical bits and have cut zero. High-probability guarantees and derandomization require additional arguments. The selected seed is a reproducible deterministic setting; the finite allowed seed set is not assumed to have the theoretical uniform-bit distribution.

## Build a vertex cover from a matching

Greedily add edges with disjoint endpoints until the matching M is maximal. Maximal means no edge can be added; it is different from maximum cardinality. Selecting every matched endpoint gives a cover: an uncovered edge would have two unmatched endpoints and could be added, contradicting maximality.

$$
|M|\le\mathrm{OPT}_{cover},\qquad
|cover|=2|M|\le2\mathrm{OPT}_{cover}.
$$

Every cover must hit each vertex-disjoint matching edge, establishing the lower bound. Read the actual endpoint pairs and cover bitmask to inspect the witness. The empty graph has optimum zero, so its multiplicative ratio is left undefined rather than dividing by zero.

## Remove randomness with conditional expectations

Give each nonnegative-weight edge e weight wₑ. Independent fair vertex bits make its expected crossing weight wₑ/2, so E[C]=Σwₑ/2≥OPT/2. Fix vertices sequentially. Before fixing the next bit, the current conditional expectation is the average of the two possible next conditional expectations. At least one choice cannot decrease it. After the last bit, the conditional expectation is the deterministic cut value, proving a deterministic one-half approximation.

An edge with both endpoints fixed contributes its known crossing weight. With at least one endpoint still independently fair, it contributes half its weight. These contributions make the conditional expectations computable without enumerating all future assignments; an implementation may update only incident edges. Nonnegative weights are needed for Σwₑ≥OPT in this proof.

| Procedure | Guaranteed quantity |
|---|---|
| One uniform random cut | At least OPT/2 in expectation |
| One selected random sample | Can still have cut zero |
| Conditional-expectation choices | At least OPT/2 on every input |

```python
from fractions import Fraction as F
edges=[(0,1,3),(1,2,2),(0,2,4),(2,3,1)]
def expected(partial):
    return sum(F(w)*(int(partial[i]!=partial[j]) if i in partial and j in partial else F(1,2)) for i,j,w in edges)
partial={};history=[expected(partial)]
for v in range(4):
    scores=[expected({**partial,v:b}) for b in (0,1)]
    bit=0 if scores[0]>=scores[1] else 1
    partial[v]=bit;history.append(expected(partial))
assert all(b>=a for a,b in zip(history,history[1:]))
cut=expected(partial)
opt=max(sum(w for i,j,w in edges if ((m>>i)&1)!=((m>>j)&1)) for m in range(16))
assert 2*cut>=sum(w for _,_,w in edges) and cut<=opt
print(history,partial,cut,opt)
```

The exact Fraction example uses a weighted four-vertex auxiliary graph; the live six-vertex lab remains unweighted. Its conditional expectations increase from 5 to 8, and exhaustive optima are used only to check the result, not to choose bits. A reproducible seed and a derandomization proof are different objects. The matching-based cover guarantee above solves a separate objective and remains its own certificate.

## Worked example and exercise

A triangle has cut zero or two. Among its eight active-vertex assignments, two assign all three bits equally. Thus P(C=0)=1/4 and P(C=2)=3/4. The three isolated vertices multiply both counts equally and do not change the probabilities. Expected cut is 1.5, half the three-edge count.

Exercise: compute the greedy-cover ratio for a five-leaf star. Solution: a matching can contain only one edge, so its endpoint cover has size two. The optimum selects only the center, size one. The ratio is two, meeting the guarantee's boundary. Expected random-cut size 2.5 and cover size two are different objectives and cannot be directly compared as competing scores.

## Evaluate AI heuristics carefully

For learned combinatorial solvers, use identical objectives, feasibility checks, baselines and optima or certified bounds. Average sampled performance, worst-case approximation and failure probability are separate reporting requirements. Small exhaustive optima provide valuable checks without establishing performance on arbitrary large inputs.

Reference: [MIT complexity and randomness course](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/).
