## Attach inputs and units to the word fast

Search for a target in the sorted array A[i]=i. Linear search probes from the left; binary search probes the middle of the remaining interval. To compare them, specify N, target location, allowed operations and the cost unit. Here one ordinal key query is one probe. This abstract comparison model differs from counting individual machine instructions or two separate source-code branches.

Select a present relative rank, a missing key above all entries, or a missing key below all entries. Each plotted N uses the same query-generation rule. Blue and orange are actual linear and binary probe counts, not microseconds.

## Quantifiers and input families

$$
T(N)\in O(g(N))\iff \exists c>0,\exists N_0,\ \forall N\ge N_0:\ T(N)\le c g(N).
$$

Omega states an eventual lower bound, and Theta states both upper and lower bounds. Constants and small-size exceptions matter, but first specify what is maximized or averaged. Worst-case complexity maximizes over all allowed inputs of size N. Average-case complexity requires an input distribution. If the family always requests the first element, linear search takes one probe for every N. Calling that picture a general worst-case Theta(N) curve would be incorrect.

## Binary-search invariant

Sorted order and the remaining interval [lo,hi] are essential. Preserve the invariant that, if present, the target lies inside that interval. Compare with the midpoint and discard the impossible half. A missing query terminates when lo exceeds hi.

$$
L_{next}\le\lfloor L/2\rfloor,\qquad
T_{binary}(N)\le\lfloor\log_2N\rfloor+1.
$$

This is a worst-case comparison bound, not a claim that every selected target reaches it. The table records the actual probe index and its interval, allowing the invariant to be checked directly.

## Space and diagnostic traces

The core searches are iterative, with no recursive stack and a constant number of auxiliary scalar indices. The N-element input array is separate. However, this teaching interface also stores a probe trace and results for several N values. Including those diagnostics, the whole program is not constant-space. Distinguish input storage, algorithm auxiliary space, required output and debugging records when making memory claims.

## Include preprocessing and state which lower bound applies

In an ordered comparison search, an equality result identifies the probed key and terminates that successful branch. With at most h probes, a binary search tree can place at most 1+2+⋯+2^(h−1)=2ʰ−1 keys at its probe nodes. Thus worst-case successful search needs h≥ceil(log₂(N+1)). Iterative midpoint search reaches this bound, although a particular target can be much cheaper. This argument assumes ordinal key queries and sorted order; a hash lookup or an arithmetic operation on represented integer keys belongs to a different model.

For repeated membership queries on initially unsorted data, suppose sorting is charged its mergesort worst-case bound B=Nlog₂N−N+1 for power-of-two N. Charge every scan N probes and every binary query b=ceil(log₂(N+1)). For N>b in this declared model,

$$
B+Qb\le QN\quad\Longleftrightarrow\quad Q\ge B/(N-b).
$$

This is an algebraic crossover between charged costs, not a promise about every query stream or every machine. An always-first-element scan violates the N-probe charging assumption. Returning positions in the original unsorted array also requires retaining a position mapping.

| N | Charged sort bound B | First integer crossover Q |
|---|---:|---:|
| 8 | 17 | 5 |
| 64 | 321 | 6 |
| 1024 | 9217 | 10 |

```python
from math import ceil,log2
for N in (8,64,1024):
    B=N*(N.bit_length()-1)-N+1
    b=ceil(log2(N+1));Q=ceil(B/(N-b))
    assert B+Q*b<=Q*N
    assert B+(Q-1)*b>(Q-1)*N
    assert 2**(b-1)-1<N<=2**b-1
    print(N,B,b,Q)
```

The live lab already supplies a sorted array and does not execute this preprocessing pipeline. For retrieval systems, construction, updates, storage and query distributions must be reported alongside the core query bound. No finite timing of the provided array can establish an omitted preprocessing cost.

## Worked example and exercise

For N=32, linear search needs 32 probes to find key 31. Binary search probes 15,23,27,29,30,31, taking six. Missing key 32 also terminates after those six probes. A linear query for key zero instead takes one probe.

Exercise: derive the binary-search path for target six when N=8. Solution: index three contains three, leaving [4,7]. Index five contains five, leaving [6,7]. Index six matches, for three probes in total. Check the sorted-interval invariant and returned index against the table rather than just comparing a final count.

## Reporting complexity in AI systems

Retrieval indexes and sparse data structures can appear unfairly attractive if preprocessing, sorting, construction and updates are omitted. Report query probes, construction memory, observed latency and input distribution together. Finite curves help inspect constants and boundary behavior. General bounds require a quantified argument and a valid invariant, not merely a plausible plot.

Reference: [MIT algorithms course](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
