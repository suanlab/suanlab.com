## From distributions to individual descriptions

Shannon entropy is average uncertainty under a distribution. Algorithmic information asks for the shortest program producing one given string. Whatever its generation history, 01010101 admits the description “repeat 01 four times.” Generation and describability are distinct; a short word alone cannot justify strong randomness claims.

Fix a universal prefix machine U. A program must output its target and halt:

$$
K_U(x)=\min_{p:U(p)=x}|p|.
$$

## A small computable language

The lab uses a finite decoder D instead of a universal machine. Every output is eight bits and every command terminates immediately. Headers count toward description length.

| Grammar | Output | Total bits |
| --- | --- | --- |
| 0 + eight bits x | x literally | 9 |
| 10 + bit b | repeat b eight times | 3 |
| 110 + two bits ab | repeat ab four times | 5 |
| 1110 + four bits abcd | repeat abcd twice | 8 |

The programs are prefix-free: identifying the header fixes payload length. Their Kraft sum is 256×2⁻⁹ + 2×2⁻³ + 4×2⁻⁵ + 16×2⁻⁸ = 15/16.

## Compare programs for one output

Byte 85 is 01010101. Its literal program 001010101 costs nine bits; period-two program 11001 costs five; period-four program 11100101 costs eight. Minimum length is five. Byte zero has the three-bit program 100. With language level zero, only literals remain and both targets cost nine.

$$
K_D(x)=\min_{p:D(p)=x}|p|.
$$

Bars compare candidate lengths for the same output. This finite K_D is not universal K_U. A found description establishes an upper bound; optimality here follows from exhaustive enumeration of the declared family.

## Charge the interpreter

A universal machine can simulate this fixed decoder with a constant interpreter cost:

$$
K_U(x)\le K_D(x)+c_D.
$$

Different universal languages agree up to additive constants independent of x. Those constants can dominate short objects. The finite decoder here is not itself universal, so do not claim the full two-way invariance theorem for this experiment.

When AI models compress data, ask whether decoder, dictionary and architecture are shared or charged. Inventing a new target-specific instruction after seeing the data while excluding its implementation cost changes the comparison.

## Representation language changes description cost

Suppose universal prefix machine U can identify and simulate another fixed prefix machine V using c bits. Prepending that identifier to a shortest V program gives K_U(x)≤K_V(x)+c. If V is also universal, reverse simulation yields two-sided invariance |K_U(x)−K_V(x)|≤c′. The constants depend on the machine pair, not the target x. The current finite eight-bit-output D is not universal, so only the first direction follows automatically.

For a fixed computable bijection f with computable inverse, a fixed postprocessing program gives K_U(f(x))≤K_U(x)+c_f. Applying the inverse gives the reverse inequality. Computable changes of coordinates or serialization thus preserve universal complexity up to an additive constant. They need not preserve the finite grammar's K_D exactly, make the constants small or preserve runtime.

## Shortest programs and summed program mass

Give each finite prefix command weight 2⁻length and sum contributions producing the same output. The complete family has Kraft mass 15/16, making this a subprobability:

$$
m_D(x)=\sum_{p:D(p)=x}2^{-|p|},\qquad
\sum_xm_D(x)=\frac{15}{16},\qquad
m_D(x)\ge2^{-K_D(x)}.
$$

Thus −log₂m_D(x)≤K_D(x). The mass reflects multiple descriptions, rather than only the shortest one. Do not silently treat m_D as normalized; dividing by 15/16 changes the logarithmic cost too.

| Byte | K_D | m_D | −log₂m_D |
|---|---|---|---|
| 0 | 3 | 83/512 | 2.624961 |
| 85 | 5 | 19/512 | 4.752072 |
| 173 | 9 | 1/512 | 9 |

The code constructs the four command families and sums exact fractions. This finite calculation does not implement universal algorithmic probability or its coding theorem. The distinction between summing descriptions and choosing one also connects to the next lesson's MDL mixture.

**Exercise:** If a different decoder is invented for each x with a special “print x” instruction, is the invariance constant shared? **Explanation:** No. A target-dependent decoder changes its simulation cost. Hiding that cost does not compare the same fixed machine pair.

```python
from fractions import Fraction as F

def programs():
    out=[('0'+f'{x:08b}',x) for x in range(256)]
    out += [('10'+str(x),255*x) for x in range(2)]
    out += [('110'+f'{x:02b}',int(f'{x:02b}'*4,2)) for x in range(4)]
    out += [('1110'+f'{x:04b}',int(f'{x:04b}'*2,2)) for x in range(16)]
    return out

from math import log2

family=programs()
assert sum(F(1,2**len(code)) for code,x in family)==F(15,16)
for target in [0,85,173]:
    descriptions=[code for code,x in family if x==target]
    minimum=min(map(len,descriptions))
    mass=sum(F(1,2**len(code)) for code in descriptions)
    assert mass>=F(1,2**minimum)
    print('byte, finite minimum, program mass, negative log mass:',
          target,minimum,str(mass),round(-log2(float(mass)),6))
```


## Exercises with explanations

1. How short is 11111111? **Three bits, program 101, in this language.** Source length and program length differ.
2. Is the lab minimum optimal over all algorithms? **No.** It is optimal among 278 fixed commands.
3. Can a target-specific instruction make any object look simple? **Yes, if decoder cost is hidden.** State and charge the shared model consistently.

Further reading: [Grünwald and Vitányi, Algorithmic Information Theory](https://arxiv.org/abs/0809.2754).
