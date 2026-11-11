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

## Average lower bounds and the order of quantifiers

The terminating comparison transcripts of a deterministic comparison-only sorter are prefix-free: once a transcript terminates, a longer transcript cannot share it as a prefix. If their lengths are ℓᵢ, Kraft gives Σ2^(−ℓᵢ)≤1. Jensen applied to the uniform average over n! input orders yields

$$
2^{-\overline\ell}\le\frac1{n!}\sum_i2^{-\ell_i}\le\frac1{n!},
\qquad\overline\ell\ge\log_2(n!).
$$

More generally, a nonuniform input law has entropy H(p), and the prefix-code argument gives Eₚℓ≥H(p). For an always-correct randomized sorter, fix a random tape on which it is correct on every input order; for this finite input set the exceptional probability is still zero. Apply the uniform average bound to each such deterministic tree and then average over tapes. Some input must have expected randomized cost at least log₂(n!). This is a statement about worst-input expected cost, not every input and not a proof for algorithms allowed to return wrong answers.

| n | Insertion mean comparisons under uniform permutations |
|---|---:|
| 3 | 8/3 |
| 4 | 59/12 |
| 5 | 463/60 |

```python
from itertools import permutations
from fractions import Fraction as F
from math import log2,factorial
def trace(p):
    a=list(p);out=''
    for i in range(1,len(a)):
        key=a[i];j=i-1
        while j>=0:
            less=key<a[j];out+='1' if less else '0'
            if not less:break
            a[j+1]=a[j];j-=1
        a[j+1]=key
    return out
for n in (3,4,5):
    codes=sorted(trace(p) for p in permutations(range(n)))
    assert all(not b.startswith(a) for a,b in zip(codes,codes[1:]))
    kraft=sum(F(1,2**len(t)) for t in codes)
    mean=F(sum(map(len,codes)),len(codes))
    assert kraft<=1 and float(mean)>=log2(factorial(n))
    print(n,mean,kraft)
```

The exact transcript calculation checks prefix-freeness and Kraft independently of a plotted mean. It counts comparisons of arbitrary distinct keys and reorders input references; exploiting known numeric labels to synthesize a fixed output would leave that model. Small enumeration supports the implementation, while the quantified argument supplies the general lower bound.

## Worked example and exercise

For n=3 there are six permutations, and ceil(log₂6)=3. Insertion sort uses two comparisons on sorted input and three on reversed input. A worst-case lower bound of three does not require every input to take three. At n=5, mergesort's observed maximum is eight while the counting bound is seven. A lower bound is not automatically achievable for every size.

Exercise: compare the n=7 bound with this mergesort. Solution: 7!=5040 gives ceil(log₂5040)=13. Balanced mergesort's worst count is n ceil(log₂n)-2^{ceil(log₂n)}+1=21-8+1=14. The interface checks this finite value through all 5040 actual traces.

## Respect model boundaries

Counting or radix sorting of represented integers permits operations beyond key comparisons. Analyze key ranges, word lengths and memory assumptions separately. The comparison bound cannot be transferred unchanged to every computational model. Nor does this sorting argument decide the relationship between P and NP.

For AI-proposed algorithms, distinguish successful small examples, correctness for every input, and upper or lower bounds in a specified model. A lower bound is a quantified argument over allowed algorithms, not merely an observation that an implementation ran slowly.

Reference: [MIT algorithms materials](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
