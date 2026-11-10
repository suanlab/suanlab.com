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

## Exercises with explanations

1. How short is 11111111? **Three bits, program 101, in this language.** Source length and program length differ.
2. Is the lab minimum optimal over all algorithms? **No.** It is optimal among 278 fixed commands.
3. Can a target-specific instruction make any object look simple? **Yes, if decoder cost is hidden.** State and charge the shared model consistently.

Further reading: [Grünwald and Vitányi, Algorithmic Information Theory](https://arxiv.org/abs/0809.2754).
