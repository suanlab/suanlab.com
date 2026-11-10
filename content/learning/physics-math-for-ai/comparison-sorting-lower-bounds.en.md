## From one algorithm to a model-wide limitation

Counting insertion sort's worst-case comparisons analyzes a particular algorithm. Proving that every comparison-based sorting method needs many comparisons establishes a lower bound for a computation model. These statements quantify over different objects. The experiment actually sorts every permutation of two through seven distinct keys using insertion sort or mergesort and records binary comparison outcomes.

Blue is the exact comparison-count PMF under uniform permutations. Orange highlights the selected permutation's entry. Query indices are reduced modulo n!, so retaining an index while changing n does not retain the same permutation.

## Decision-tree leaves and depth

Each key comparison has two outcomes. Those outcomes determine later comparisons and the returned order of input references. At least n! leaves are needed to distinguish all relative orders. What differs is the order in which input elements must be placed, not the set of sorted key values.

$$
2^h\ge n!,\qquad h\ge\lceil\log_2(n!)\rceil.
$$

This bounds worst-case depth for deterministic comparison sorting. Using n!≥(n/2)^{n/2} gives Omega(n log n). A fast individual input does not contradict this worst-case result. Also distinguish the average information lower bound log₂(n!) from the ceiling applied to worst-case depth.

## Count actual comparisons

Insertion sort counts each key comparison made while moving a key leftward. Index checks and loop branches are excluded. Mergesort counts comparisons between the leading keys of two sorted halves. Diagnostic trace storage and exhaustive permutation generation are separate costs from these sorting comparisons.

The checks verify ascending outputs, n! distinct outcome traces, and observed worst cases above the counting lower bound. Such enumeration supports the small implementation. The general lower bound comes from the leaf-count proof, not from extrapolating a finite chart.

## Worked example and exercise

For n=3 there are six permutations, and ceil(log₂6)=3. Insertion sort uses two comparisons on sorted input and three on reversed input. A worst-case lower bound of three does not require every input to take three. At n=5, mergesort's observed maximum is eight while the counting bound is seven. A lower bound is not automatically achievable for every size.

Exercise: compare the n=7 bound with this mergesort. Solution: 7!=5040 gives ceil(log₂5040)=13. Balanced mergesort's worst count is n ceil(log₂n)-2^{ceil(log₂n)}+1=21-8+1=14. The interface checks this finite value through all 5040 actual traces.

## Respect model boundaries

Counting or radix sorting of represented integers permits operations beyond key comparisons. Analyze key ranges, word lengths and memory assumptions separately. The comparison bound cannot be transferred unchanged to every computational model. Nor does this sorting argument decide the relationship between P and NP.

For AI-proposed algorithms, distinguish successful small examples, correctness for every input, and upper or lower bounds in a specified model. A lower bound is a quantified argument over allowed algorithms, not merely an observation that an implementation ran slowly.

Reference: [MIT algorithms materials](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
