## How much compression is possible with errors?

Lossless compression must recover the source exactly. Lossy compression permits discarding information within a specified distortion budget. This lab assumes an iid fair binary source and Hamming cost d(x,y)=1{x≠y}. Distortion is the average mismatched-bit fraction. Different sources or distortion measures require different results.

$$
R(D)=\inf_{P_{Y|X}:E[d(X,Y)]\leq D}I(X;Y).
$$

## The exact binary boundary

Here R(D)=1−h₂(D) up to D=.5 and zero thereafter. At D=.1, rate is about .531 bits per symbol. Zero distortion requires one bit per fair source bit. With budget at least .5, always reconstructing zero achieves distortion .5 without sending information.

$$
R(D)=\begin{cases}1-h_2(D),&0\leq D\leq1/2,\\0,&1/2\leq D\leq1.\end{cases}
$$

## Why this boundary appears

For actual error d≤D≤.5, the binary Fano relation gives H(X|Y)≤h₂(d)≤h₂(D), hence I(X;Y)≥1−h₂(D). A binary symmetric test channel with crossover D reaches the same distortion and information. The rate–distortion coding theorem connects this information optimization to operational long-block coding; the graph itself does not construct a block codec.

## Separate the budget from a selected channel

Blue is optimal R(D) over all allowed test channels. Orange is 1−h₂(ε) for a symmetric test channel. The curves match through ε=.5, but larger flips are inefficient. At ε=.75, channel information is about .189 bits while the optimum for that distortion budget is zero.

Inspect feasibility separately: actual distortion ε must not exceed D. A channel with D=.1 and ε=.25 uses less information but violates the constraint. Blue and orange dots represent different controls.

## Connections and limits

The example motivates studying representation information alongside reconstruction cost. Its closed form does not automatically describe semantic image or text loss, or arbitrary neural networks. Units are asymptotic bits per symbol, not exact single-file byte lengths or guarantees for every sample. Specify source, reproduction alphabet, distortion and average-versus-worst-case conditions first.

## A finite block also has a necessary bit budget

Encode n iid fair bits X^n into a message M and reconstruct a binary block Y^n. There are at most 2^b messages, and the decoder has no source access beyond M. Assume a deterministic encoder or independent private randomness arranged so that X^n→M→Y^n holds. Let average Hamming distortion be d̄=(1/n)ΣP(X_i≠Y_i), with d̄≤D≤1/2. This is different from a worst-block guarantee.

$$
I(X^n;Y^n)\leq I(X^n;M)\leq H(M)\leq b.
$$

Set E^n=X^n XOR Y^n. Given Y^n, the source and error vectors determine each other. Remove conditioning, use subadditivity, then apply concavity of h₂.

$$
H(X^n\mid Y^n)=H(E^n\mid Y^n)
\leq H(E^n)\leq\sum_{i=1}^nh_2(d_i)
\leq nh_2(\bar d)\leq nh_2(D).
$$

Since H(X^n)=n, combining the bounds yields b/n≥1−h₂(D). This converse excludes impossible rates even at finite block length. It does not establish a code attaining the boundary at that same n. Asymptotic achievability and the exact finite-block attainable region are separate claims.

## Find actual reconstruction blocks for a one-bit message

With b=1, select two reconstruction blocks and map each source block to a nearer one. For any fixed reconstruction pair, nearest-neighbor encoding minimizes each source block's Hamming cost. Averaging over all 2^n equally likely source blocks then gives exact distortion.

Enumerate every distinct reconstruction pair at n=2,3,4: respectively 6,28,120 candidates. D* is the best mean distortion within this specified two-reconstruction, nearest-neighbor family. Gap is the actual rate 1/n minus R(D*).

| n | D* | Gap |
|---|---|---|
| 2 | $1/4$ | 0.31128 |
| 3 | $1/4$ | 0.14461 |
| 4 | $5/16$ | 0.14604 |

Each remains above the information boundary. At n=3, for example, (000,111) has mean error count 3/4 and per-bit distortion 1/4. Its rate 1/3 exceeds R(1/4)≈0.18872. An optimal information-valued test channel does not by itself implement that exact rate when sending three bits through a one-bit message.

## Record mean, worst case and model scope

The table averages over a uniform source. Some blocks in the n=3 example incur one error, giving worst per-bit distortion 1/3. Do not report mean 1/4 as a guarantee for every block. Message length b is fixed; these are not variable-length mean bit costs.

The code exhaustively enumerates reconstruction pairs and source blocks. It obtains exact rational distortion for small finite problems, without replacing the long-block achievability theorem. The existing D and ε controls describe single-letter test channels and do not generate these block codes.

```python
from fractions import Fraction as Q
from itertools import product, combinations
from math import log2

def h(p):
    return 0 if p in [0, 1] else -p*log2(p)-(1-p)*log2(1-p)
def distance(a, b):
    return sum(x != y for x, y in zip(a, b))

for n, expected in [(2, Q(1, 4)), (3, Q(1, 4)), (4, Q(5, 16))]:
    source = list(product([0, 1], repeat=n))
    results = []
    for pair in combinations(source, 2):
        costs = [min(distance(x, c) for c in pair) for x in source]
        results.append((Q(sum(costs), n*len(source)), pair, max(costs)))
    best = min(row[0] for row in results)
    assert best == expected
    rate = Q(1, n)
    boundary = 1-h(float(best))
    assert float(rate) >= boundary
    selected = next(row for row in results if row[0] == best)
    if n == 3:
        antipodal = [min(sum(x), n-sum(x)) for x in source]
        assert Q(sum(antipodal), n*len(source)) == Q(1, 4)
        assert Q(max(antipodal), n) == Q(1, 3)
    print("n, pairs, best distortion, rate, R(D), gap:",
          n, len(results), best, rate, boundary, float(rate)-boundary)
    print("One optimal pair and its worst per-bit distortion:", selected[1], Q(selected[2], n))
```

## Questions for research use

1. Does satisfying a converse prove a code exists at that point? **No: a necessary lower bound alone does not prove existence.**
2. Is D* a worst-case error guarantee? **No: it averages under the specified uniform source.**
3. What should a practical comparison specify? **Block length, encoder, decoder, rate, distortion criterion and computation cost.**

For background, see [MIT 6.441, Chapter 23](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/880c39878ba7dc35a5fbb35758d42a69_MIT6_441S16_chapter_23.pdf). The error-vector converse and finite reconstruction-pair enumeration are worked out under the stated assumptions.

## Exercises with explanations

1. What is R(.5)? **Zero.**
2. Does the ε=1 channel have zero information? **No: perfect inversion is invertible and retains one bit.**
3. Is a low-information channel outside the distortion budget a valid solution? **No: it must satisfy the constraint.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
