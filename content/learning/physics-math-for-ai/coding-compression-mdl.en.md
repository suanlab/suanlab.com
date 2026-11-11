## From probabilities to actual binary words

Minus log probability suggests ideal lengths, but actual codes use binary strings. Codebook zero assigns 0,10,110,111, with lengths 1,2,3,3. Codebook one assigns 00,01,10,11, all length two. Neither has a word that prefixes another, allowing sequential decoding.

$$
L=\sum_i p_i\ell_i,\qquad \sum_i2^{-\ell_i}\leq1\quad\text{for binary prefix codes}.
$$

## Count bits for eight observations

Counts (4,2,1,1) give data length 4×1+2×2+1×3+1×3=14 bits under codebook zero, versus sixteen under fixed-length code. Empirical entropy and the first code's mean length both equal 1.75 bits per symbol in this special matched example.

For uniform counts (2,2,2,2), the first code instead costs eighteen data bits and fixed code sixteen. A fixed codebook is not optimal for every distribution. Distinguish entropy bounds on optimal prefix lengths for a known finite source from the performance of these particular fixed codebooks.

## Tell the receiver which codebook was used

Codebook headers are 10 for the first candidate and 0 for the second. These headers are themselves prefix-free. Default two-part totals are therefore 2+14=16 bits and 1+16=17 bits. Observation count N is assumed known, so separate termination overhead is excluded from this comparison.

$$
L_{\rm total}=L(\text{codebook})+L(\text{data}\mid\text{codebook}).
$$

This is a small MDL example comparing two predefined valid codes, not a learned continuous model or full universal-coding implementation. Designing a codebook from the data would require specifying its description cost too.

## A short-looking invalid code

Codebook two assigns 0,01,10,11. Zero prefixes 01, and Kraft sum is 1.25. String 010 has two parses: 0|10 and 01|0. The screen reports raw mean length but withholds a valid total and excludes this candidate. Failure of the prefix property alone does not settle unique decodability of every general code; this particular list also has direct ambiguity.

Blue shows data bits for the two valid candidates; orange includes headers. The table shows the selected symbol code. Repeating the observation block scales data costs while leaving fixed header costs unchanged.

## Prove the mean-length lower bound using Kraft's sum

Separate from the fixed screen codebooks, consider a known finite source with positive masses p_i and a binary prefix code. There are at least two symbols and positive integer lengths ℓ_i. Its Kraft sum K=Σ2^(−ℓ_i) satisfies 0<K≤1. Normalize r_i=2^(−ℓ_i)/K to obtain a probability distribution. Expanding the bit-valued KL gives

$$
L-H(P)=D_{\rm KL}(P\Vert r)-\log_2K\geq0.
$$

Thus mean prefix-code length cannot be below entropy. Equality requires both zero KL and K=1, giving p_i=2^(−ℓ_i). A code near this mean lower bound need not give every message the same length.

## Round lengths and construct actual words

Choose Shannon lengths ℓ_i=ceil(−log₂p_i). Then 2^(−ℓ_i)≤p_i, so their Kraft sum is at most one. Assign binary intervals in order of increasing length to construct a prefix code. A word of length ℓ represents a dyadic interval of size 2^(−ℓ) inside [0,1). Previously assigned larger intervals align with the current grid; their cumulative size leaves room for the next interval because the total cannot exceed one. Disjoint intervals ensure that no word prefixes another.

Each length is strictly below −log₂p_i+1, giving the following bound for this construction, without assuming it is optimal.

$$
H(P)\leq L_{\rm Shannon}<H(P)+1.
$$

For p=(1/2,1/3,1/6), the lengths are (1,2,3), words (0,10,110), Kraft sum 7/8 and mean 5/3. Entropy is approximately 1.45915. The different valid code (0,10,11) has shorter mean 3/2. Shannon construction is not always optimal.

## Blocking reduces per-symbol integer overhead

For iid blocks of m symbols, probabilities multiply and block entropy is mH(P). Applying the same construction to blocks yields

$$
H(P)\leq\frac{L_m}{m}<H(P)+\frac1m.
$$

These are the actual construction results for the same p. Rate is bits per symbol and Gap is Rate−H(P). Equal rates at m=1 and 2 illustrate that actual gaps need not strictly decrease at every step.

| m | Rate | Gap |
|---|---|---|
| 1 | 1.66667 | 0.20752 |
| 2 | 1.66667 | 0.20752 |
| 4 | 1.58854 | 0.12939 |

The block alphabet has 3^m members, so a bit-efficiency bound does not remove computational or storage costs. Both parties are assumed to know the distribution and block length. Communicating a fitted distribution or codebook requires an additional cost, as in the earlier MDL model description.

## Verify rational lengths and decoding

The code chooses lengths using rational comparisons rather than a floating-point logarithm followed by ceil. It constructs canonical binary words in length order, checks their prefix property, and round-trips every two-block message. This construction is separate from the screen's two fixed candidates.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod, log2

p = [Q(1, 2), Q(1, 3), Q(1, 6)]
H = -sum(float(v)*log2(float(v)) for v in p)
for m in [1, 2, 4]:
    masses = [prod(block) for block in product(p, repeat=m)]
    lengths = []
    for mass in masses:
        length = 0
        while Q(1, 1 << length) > mass:
            length += 1
        lengths.append(length)
    words = [None]*len(masses)
    code, previous = 0, 0
    for i in sorted(range(len(masses)), key=lambda i: (lengths[i], i)):
        length = lengths[i]
        code <<= length-previous
        assert code < 1 << length
        words[i] = format(code, '0'+str(length)+'b')
        code += 1
        previous = length
    assert all(not v.startswith(w) for i, w in enumerate(words)
               for j, v in enumerate(words) if i != j)
    for i, j in product(range(len(words)), repeat=2):
        remaining, decoded = words[i]+words[j], []
        while remaining:
            k = next(k for k, w in enumerate(words) if remaining.startswith(w))
            decoded.append(k)
            remaining = remaining[len(words[k]):]
        assert decoded == [i, j]
    kraft = sum(Q(1, 1 << n) for n in lengths)
    rate = sum(v*n for v, n in zip(masses, lengths))/m
    assert kraft <= 1 and H <= float(rate) < H+1/m
    print("m, alphabet size, Kraft sum, rate, gap:", m, len(masses), kraft, rate, float(rate)-H)
```

## Questions for research use

1. Is −log₂p_i an exact file length? **No: distinguish integer words, headers and termination rules.**
2. Does the entropy bound control every realized message length? **Here it bounds an average under the known source.**
3. What should increasing m report? **Per-symbol cost, alphabet and codebook size, and model transmission cost.**

For background, see [MIT 6.441, Chapter 6, Theorems 6.5–6.6](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/67d8e68cd8fda55366e3f9f0a9465c17_MIT6_441S16_chapter_6.pdf). The three-mass block codewords and lengths are constructed directly.

## Exercises with explanations

1. Which two-part code is shorter for uniform counts? **Fixed code, totaling seventeen bits.**
2. May an invalid candidate be selected because it looks short? **No: valid decoding comes first.**
3. Do deterministic empirical counts make this fixed code cost zero per symbol? **No: this example retains a four-symbol codebook.**

Further reading: [Grünwald, A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/abs/math/0406077).
