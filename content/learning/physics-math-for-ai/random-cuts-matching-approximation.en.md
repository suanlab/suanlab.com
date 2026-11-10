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

## Worked example and exercise

A triangle has cut zero or two. Among its eight active-vertex assignments, two assign all three bits equally. Thus P(C=0)=1/4 and P(C=2)=3/4. The three isolated vertices multiply both counts equally and do not change the probabilities. Expected cut is 1.5, half the three-edge count.

Exercise: compute the greedy-cover ratio for a five-leaf star. Solution: a matching can contain only one edge, so its endpoint cover has size two. The optimum selects only the center, size one. The ratio is two, meeting the guarantee's boundary. Expected random-cut size 2.5 and cover size two are different objectives and cannot be directly compared as competing scores.

## Evaluate AI heuristics carefully

For learned combinatorial solvers, use identical objectives, feasibility checks, baselines and optima or certified bounds. Average sampled performance, worst-case approximation and failure probability are separate reporting requirements. Small exhaustive optima provide valuable checks without establishing performance on arbitrary large inputs.

Reference: [MIT complexity and randomness course](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/).
