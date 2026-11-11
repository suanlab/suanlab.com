## What is impossible if a finite table is computable?

All 278 commands from the previous lab halt. Enumerating them gives an exact finite-language minimum. Uncomputability instead says that no algorithm always halts and returns exact K_U(x) for every finite string under a fixed **universal** prefix machine U.

Knowing a short description for one object is different from computing every minimum. Individual upper bounds, and some exact values, can still be proved.

## Count short descriptions

Kraft's inequality bounds the number of prefix programs of length at most L by 2ᴸ. Distinct outputs cannot outnumber programs, so not every length-(L+1) string can have such a short description.

$$
\#\{x:K_U(x)\le L\}\le 2^L.
$$

The finite lab covers two outputs at L=3, four at L=5, sixteen at L=8 and all 256 at L=9. Multiple programs may have the same output: there are 22 programs within length eight but only sixteen distinct outputs.

## A diagonal contradiction

Suppose a terminating algorithm A computes K_U exactly. Given n, enumerate strings of length n+1 in lexical order. Use A to select the first xₙ with K_U(xₙ)>n. Counting guarantees existence, so this search halts.

Yet the search procedure and A are fixed. A program producing xₙ needs only a self-delimiting description of n, costing O(log n), plus a constant. For sufficiently large n:

$$
\begin{aligned}
K_U(x_n)&>n,\\
K_U(x_n)&\le O(\log n)+c<n.
\end{aligned}
$$

This contradiction rules out A. The finite coverage plot illustrates counting and duplicate outputs; it does not prove this universal theorem.

## Why more runtime does not settle the question

Running programs sequentially can become stuck on one that never halts. Dovetailing gives candidates alternating simulation steps, eventually finding each terminating program. Discovered descriptions provide improving upper bounds, but generally cannot certify that every shorter unfinished candidate will never halt later.

An AI-generated compressed description can be executed and checked against its target. A search failure or timeout does not establish intrinsic randomness or universal optimality.

## Construct the O(log n) prefix description explicitly

The diagonal argument's n is a positive integer. Let its binary representation have length m=floor(log₂n)+1. Write m−1 zero bits followed by those m binary digits. This is Elias gamma coding. Counting leading zeros tells the decoder how many bits remain, so no codeword is a proper prefix of another:

$$
\ell_\gamma(n)=2\lfloor\log_2n\rfloor+1.
$$

There are 2ᵐ⁻¹ integers with m digits, each costing 2m−1 bits; the total Kraft mass is Σm≥1 2⁻ᵐ=1. The fixed procedure “read n, use the alleged exact K algorithm and output the first incompressible string” can consequently be a prefix machine taking gamma(n). Universal simulation adds constant c, giving K_U(x_n)≤2floor(log₂n)+1+c<n for sufficiently large n. A finite table with an unspecified c does not compute numerical universal K values.

| n | Gamma length |
|---|---|
| 1 | 1 |
| 2 | 3 |
| 8 | 7 |
| 256 | 17 |
| 4096 | 25 |

The code checks integer decoding and a finite Kraft sum. It is not a universal K oracle and does not replace the diagonal proof by a finite simulation.

## A convergent upper bound differs from a known stopping point

A fixed U has a known program encoding a self-delimiting literal and its length, giving a computable upper bound B(x). Simulate every program of length≤B(x) for t steps each. Let b_t(x) be the smaller of B(x) and the shortest discovered program producing x. Finitely many finite simulations make b_t computable. Eventually a shortest program halts, so b_t(x) reaches exact K_U(x).

If a procedure could always decide whether a given stage t had already reached K_U(x), increasing t until that decision succeeded would compute exact K, contradicting uncomputability. An integer bound eventually stabilizing and an algorithm certifying that stage are different claims. The finite D avoids this universal problem because all commands and their termination are known.

The cited Vitányi paper primarily uses plain complexity C. Here literal and integer descriptions retain prefix K's self-delimiting overhead; the plain n+O(1) literal bound is not copied unchanged to prefix complexity.

**Exercise:** Does a long interval without improvement prove optimality? **Explanation:** No. A shorter program may halt much later. Distinguish this from completion evidence established for a particular finite family.

```python
from fractions import Fraction as F

def gamma(n):
    assert n>=1
    payload=bin(n)[2:]
    return '0'*(len(payload)-1)+payload

def decode(code):
    zeros=len(code)-len(code.lstrip('0'))
    assert len(code)==2*zeros+1 and code[zeros]=='1'
    return int(code[zeros:],2)

for n in range(1,4096):
    code=gamma(n)
    assert decode(code)==n
    assert len(code)==2*(n.bit_length()-1)+1
# Exact Kraft mass for all positive integers with at most 12 binary digits.
mass=sum(F(1,2**len(gamma(n))) for n in range(1,4096))
assert mass==1-F(1,2**12)
for n in [1,2,8,256,4096]:
    print('n, self-delimiting code length:',n,len(gamma(n)))
print('finite gamma Kraft mass:',mass)
# This tests an integer code, not a universal Kolmogorov-complexity oracle.
```


## Exercises with explanations

1. Is byte 173 covered at L=8? **No in this finite language.** Its nine-bit literal is shortest here, which is not a universal lower bound.
2. Does uncomputability rule out useful compressors? **No.** It rules out exact optimization for every input, not useful discoveries.
3. Why establish termination in the contradiction? **The alleged short description must be an actual program producing xₙ.** Totality of A and counting ensure that.

Further reading: [Vitányi, How Incomputable Is Kolmogorov Complexity?](https://homepages.cwi.nl/~paulv/papers/incomp.pdf).
