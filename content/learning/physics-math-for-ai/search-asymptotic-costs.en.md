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

## Worked example and exercise

For N=32, linear search needs 32 probes to find key 31. Binary search probes 15,23,27,29,30,31, taking six. Missing key 32 also terminates after those six probes. A linear query for key zero instead takes one probe.

Exercise: derive the binary-search path for target six when N=8. Solution: index three contains three, leaving [4,7]. Index five contains five, leaving [6,7]. Index six matches, for three probes in total. Check the sorted-interval invariant and returned index against the table rather than just comparing a final count.

## Reporting complexity in AI systems

Retrieval indexes and sparse data structures can appear unfairly attractive if preprocessing, sorting, construction and updates are omitted. Report query probes, construction memory, observed latency and input distribution together. Finite curves help inspect constants and boundary behavior. General bounds require a quantified argument and a valid invariant, not merely a plausible plot.

Reference: [MIT algorithms course](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
