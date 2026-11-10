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

## Exercises with explanations

1. Which two-part code is shorter for uniform counts? **Fixed code, totaling seventeen bits.**
2. May an invalid candidate be selected because it looks short? **No: valid decoding comes first.**
3. Do deterministic empirical counts make this fixed code cost zero per symbol? **No: this example retains a four-symbol codebook.**

Further reading: [Grünwald, A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/abs/math/0406077).
