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

## Exercises with explanations

1. How many of all 256 possible rows collide for a nonzero difference? **128.** Coordinate toggling pairs the two parity classes.
2. Do matching fingerprints prove matching sources? **No.** Collisions remain possible; fingerprints can filter candidates.
3. Is public randomness physically free? **No.** It is a model assumption; deployment may incur synchronization cost.

Further reading: [Stanford CS369E, Randomized Communication Complexity](https://theory.stanford.edu/~tim/w15/l/l3.pdf).
