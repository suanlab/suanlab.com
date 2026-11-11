## Parity lives in a finite field

GF(2) contains zero and one. Addition is XOR and multiplication is AND, so 1+1=0. Matrix multiplication can then test whether selected bits sum to even parity. This is different from real-valued least squares: use finite-field arithmetic throughout.

Hamming(7,4) adds three parity bits to four information bits. Columns of H are binary positions 1 through 7, with the low-order row first:

$$
H=\begin{pmatrix}1&0&1&0&1&0&1\\0&1&1&0&0&1&1\\0&0&0&1&1&1&1\end{pmatrix}.
$$

## Encode and locate one error

Place low-order payload bits at positions 3,5,6,7 and solve parity bits at 1,2,4 so Hc=0. Valid codewords form the four-dimensional null space of H. A generator matrix G equivalently satisfies c=Gm and HG=0.

Payload integer 9 has low-first bits (1,0,0,1), producing codeword 0011001 when read from position one. Flipping position three yields syndrome (1,1,0), interpreted low-first as integer three. Flip that position again.

$$
\begin{aligned}
r&=c+e,\\s&=Hr=He\pmod 2.
\end{aligned}
$$

## Derive the distance guarantee

Every column is nonzero and distinct. Thus no weight-one or weight-two vector lies in the null space. Columns one and two XOR to column three, so the vector supported at positions 1,2,3 is a valid weight-three codeword. Minimum distance is exactly three.

$$
d_{\min}=3,\qquad t=\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor=1.
$$

## Passing parity does not prove recovery

For payload 9, flip positions one and two instead. Their combined syndrome is again three. The single-error decoder flips position three and reaches another valid codeword, decoding payload 8. Post-correction syndrome is zero despite incorrect information.

The original code can detect up to two errors if used only as a detector. This particular decoder instead interprets every nonzero syndrome as one error and can miscorrect two. SECDED designs add an overall parity bit or another mechanism; that extension is not implemented here.

Blue and orange points compare sent and received bits. The table also shows corrected bits, making miscorrection visible.

## A perfect code has exactly one center for every received word

Columns 1,2,4 of H form the GF(2) standard basis, so rank H=3 and its null space has dimension 7−3=4, giving sixteen codewords. Minimum distance three makes their radius-one balls disjoint. Each contains its center and seven single-error neighbors. Their total size is 16(1+7)=128=2⁷, covering the entire received space. This is the meaning of a perfect Hamming(7,4) code.

![Sixteen disjoint Hamming(7,4) balls each contain one codeword and seven single-error neighbors, covering all 128 received words. Center labels read positions one through seven.](/assets/learning/coding-hamming-perfect-cover-en.svg)

The syndrome maps received words to eight cosets. Two vectors with the same syndrome differ by a null-space vector, so each coset has sixteen elements. Syndrome zero has leader zero; nonzero syndrome j has the single-error leader at position j. Syndrome decoding consequently selects the unique radius-one center and equals nearest-neighbor decoding.

## Compare block error for the same four-bit payload

This decoder returns to the original codeword exactly for error weights zero and one. For weight at least two, e plus the decoder's correction of weight at most one cannot be zero. It reaches another valid codeword. Since information extraction bijects codewords with four-bit payloads, the payload is also wrong. Under independent BSC noise,

$$
P_{e,\mathrm{Ham}}=1-(1-\varepsilon)^7-7\varepsilon(1-\varepsilon)^6.
$$

Align the comparison at four information bits. Repeating each bit three times consumes twelve uses, and the chance that at least one of four independent blocks fails is 1−(1−Pₑ,rep3)⁴. Hamming uses seven. Directly comparing single-bit error .028 with four-bit block error .1496944 would mix criteria.

| ε | Hamming block error, 7 uses | Repetition block error, 12 uses |
|---|---|---|
| .01 | .002031042 | .001191467 |
| .05 | .044380542 | .028686147 |
| .1 | .149694400 | .107383193 |

The code enumerates H's null space, verifies disjoint coverage of all 128 received words and checks the success condition for every error weight. These are exact small-model results, not a guarantee of efficient optimal decoding for arbitrary large codes.

**Exercise:** Does zero syndrome after nearest-neighbor correction prove one actual error? **Explanation:** No. Every received word lies in some radius-one ball, so this decoder always produces a valid codeword. Recovering the transmitted center is a separate guarantee.

```python
from fractions import Fraction as F
from math import comb

def syndrome(word):
    value=0
    for j in range(1,8):
        if (word>>(j-1))&1: value^=j
    return value

code=[word for word in range(128) if syndrome(word)==0]
assert len(code)==16
owner={}
for center in code:
    for error in [0]+[1<<j for j in range(7)]:
        received=center^error
        assert received not in owner
        owner[received]=center
assert len(owner)==128
for received in range(128):
    s=syndrome(received)
    corrected=received^(1<<(s-1) if s else 0)
    assert corrected==owner[received]
for weight in range(8):
    for error in range(128):
        if error.bit_count()!=weight: continue
        s=syndrome(error)
        corrected=error^(1<<(s-1) if s else 0)
        assert (corrected==0)==(weight<=1)
for epsilon in [F(1,100),F(1,20),F(1,10)]:
    hamming=1-(1-epsilon)**7-7*epsilon*(1-epsilon)**6
    bit_repetition=3*epsilon**2*(1-epsilon)+epsilon**3
    four_repetitions=1-(1-bit_repetition)**4
    print('epsilon, Hamming four-bit block error (7 uses), repetition four-bit error (12 uses):',
          float(epsilon),round(float(hamming),9),round(float(four_repetitions),9))
```


## Exercises with explanations

1. Can a parity-position error be corrected? **Yes.** Every position has its own nonzero column.
2. Is the XOR of codewords valid? **Yes.** Linearity gives H(c₁+c₂)=0.
3. What is the rate? **4/7.** Count information bits over GF(2), not real-vector coordinates.

Further reading: [MIT MAS.160, Hamming Codes and Syndromes](https://ocw.mit.edu/courses/mas-160-signals-systems-and-information-for-media-technology-fall-2007/94144b6cc83df4bea4f9cef22c91baf4_1017_error.pdf).
