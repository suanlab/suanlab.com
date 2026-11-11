## Are two distributed representations equal?

Alice and Bob each hold an eight-bit representation x and y. Alice sends one message; Bob decides equality. No input distribution is assumed and the protocol is fixed beforehand. We count **communicated bits**, rather than local computation.

Sending all eight bits solves the problem exactly. An always-correct deterministic message shorter than eight bits cannot. Mapping 256 inputs into fewer messages gives distinct x and x′ the same message. With Bob's input fixed to x, the messages are indistinguishable but required answers differ.

$$
m\ge\log_2 256=8\quad\text{for exact deterministic one-way equality}.
$$

## Truncation versus a random fingerprint

Sending the lowest three bits makes x=0 and y=128 both appear as 000, a false positive. Equality is distinct from preserving semantic similarity or Euclidean distance.

Instead send a GF(2) inner product with a uniform public eight-bit row r. For nonzero difference δ=x XOR y, r·δ is a fair bit. Choose a nonzero coordinate of δ: toggling that coordinate of r pairs every zero-inner-product row with a one-inner-product row.

$$
P_r[r\cdot\delta=0]=1/2\quad(\delta\ne0).
$$

## Independence supplies the guarantee

Use k independent uniform public rows. For a fixed distinct input pair, all parities match with probability 2⁻ᵏ. Equal inputs always match, so there are no false negatives. The pair must be fixed independently of the public randomness; adversarially choosing inputs after seeing the rows is a different model.

$$
P[H(x)=H(y)\mid x\ne y]=2^{-k}.
$$

At k=3 the ideal collision probability is 1/8. At k=0, every pair matches and distinct-pair error is one. Even eight random rows can be linearly dependent, leaving collision probability 1/256, whereas sending the entire byte is exact.

## Interpret the fixed seed honestly

Blue shows ideal fixed-pair random collision probability. Orange shows deterministic truncation's worst-case error. These are different criteria, not two empirical error rates measured on the same sample. Selecting identical inputs makes the blue curve zero; the worst-case truncation curve still ranges over all possible pairs.

Table rows come from a small reproducible seeded generator to illustrate a transcript. They are **not asserted to be independent uniform samples**. A single seed does not inherit the ideal probability guarantee.

Public randomness is shared without charging mask transmission in this communication model. Actual systems must account for setup and synchronization when necessary. Embedding dimension alone determines neither transmitted bits nor an exact equality guarantee.

## The null space matters after fixing the matrix

Stack the public rows into H∈GF(2)ᵏˣⁿ with rank r. Fingerprints are Hx; collision differences satisfy Hδ=0. Rank-nullity gives dimension n−r and kernel size 2ⁿ⁻ʳ. Every nonempty fingerprint fiber has the same size. Consequently k<n always leaves a nonzero collision difference. An adversary selecting inputs after seeing H can choose that difference.

$$
\#\ker H=2^{n-r},\qquad
P_H[H\delta=0]=2^{-k}\text{ for each fixed }\delta\ne0.
$$

The second statement fixes δ independently before sampling H. The first fixes one H. Exchanging these quantifiers does not establish simultaneous exact distinction of every pair with small k.

## Enumerate all sixteen 2×2 matrices

For n=k=2, each public row is uniform on 00,01,10,11, giving sixteen equally likely matrices. For each fixed nonzero δ, four matrices collide. In contrast, ten matrices collide for some nonzero δ: precisely those of rank below two.

| Rank | Matrix count | Kernel size per matrix |
|---|---|---|
| 0 | 1 | 4 |
| 1 | 9 | 2 |
| 2 | 6 | 1 |

![Separate grids compare four matrices colliding for fixed difference δ=01 with ten singular matrices colliding for some nonzero difference, among sixteen equally likely binary 2×2 public matrices.](/assets/learning/coding-fingerprint-quantifiers-en.svg)

For example, rows 01,01 make x=00 and y=10 collide. The code performs GF(2) elimination and checks every nonzero difference. This exhaustive ideal-coin example is separate from the screen's seeded eight-bit transcript.

An n×n independent uniform binary matrix has full-rank probability ∏j=0ⁿ⁻¹(1−2ʲ⁻ⁿ): choose each successive row outside the previous span. At n=8 this is about .289919. Fixed-pair error 1/256 and simultaneous exactness probability .289919 are different claims. Sharing the identity matrix and sending eight bits distinguishes every pair exactly; random matrices are not always invertible like the identity.

**Exercise:** With k=3,n=8 and rank three, how many kernel differences remain? **Explanation:** 2⁵=32, including 31 nonzero differences. This is consistent with small average collision probability for a particular fixed difference.

Background: [Stanford CS369E Lecture 3, §1 on public-coin equality](https://theory.stanford.edu/~tim/w15/l/l3.pdf). The rank histogram and grids here are independently enumerated.

```python
from itertools import product
from collections import Counter
from math import prod

def dot(row,value): return (row&value).bit_count()%2

def rank(rows):
    basis={}
    for row in rows:
        while row:
            pivot=row.bit_length()-1
            if pivot in basis: row^=basis[pivot]
            else:
                basis[pivot]=row
                break
    return len(basis)

matrices=list(product(range(4),repeat=2))
counts=Counter(rank(rows) for rows in matrices)
assert counts=={0:1,1:9,2:6}
for delta in [1,2,3]:
    collisions=sum(all(dot(row,delta)==0 for row in rows) for rows in matrices)
    assert collisions==4
singular=sum(rank(rows)<2 for rows in matrices)
assert singular==10
print('2x2 rank histogram:',sorted(counts.items()))
print('fixed nonzero difference collision, some difference collision:',4/16,singular/16)
rows=[1,1]
kernel=[delta for delta in range(4) if all(dot(row,delta)==0 for row in rows)]
assert kernel==[0,2]
print('rank-one rows 01,01: kernel',kernel)
full_rank_8=prod(1-2**(j-8) for j in range(8))
print('8x8 ideal random full rank probability:',round(full_rank_8,6))
```


## Exercises with explanations

1. How many of all 256 possible rows collide for a nonzero difference? **128.** Coordinate toggling pairs the two parity classes.
2. Do matching fingerprints prove matching sources? **No.** Collisions remain possible; fingerprints can filter candidates.
3. Is public randomness physically free? **No.** It is a model assumption; deployment may incur synchronization cost.

Further reading: [Stanford CS369E, Randomized Communication Complexity](https://theory.stanford.edu/~tim/w15/l/l3.pdf).
