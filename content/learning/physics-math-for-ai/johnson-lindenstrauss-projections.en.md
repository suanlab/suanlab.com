## Preserve all distances in a finite point set

The Johnson–Lindenstrauss principle preserves pairwise distances of fixed finite points within small relative error. Required output dimension scales with log point count and inverse squared tolerance. This lab applies (1±ε) to squared distances, not unsquared norms.

The four points in R¹²⁸ are zero, e₁, e₂ and the vector with all coordinates 1/√128. They are fixed before the projection matrix. The table records all six original squared distances and projected ratios.

## Why scale by one over square root m?

For an ideal matrix with independent N(0,1/m) entries, ||Av||²/||v||² follows χ²ₘ/m for fixed nonzero v. Expected squared norm is preserved. Expectation alone does not make every pair simultaneously accurate, so tail control is required.

## From one pair to six pairs

For 0<ε<1, a squared-Gaussian-norm tail is bounded by 2exp[−m(ε²/4−ε³/6)]. A union bound over six pairs gives:

$$
P(\text{some pair fails})\leq12\exp[-m(\varepsilon^2/4-\varepsilon^3/6)].
$$

Cap displayed probabilities at one. A sufficient m for failure at most .05 is the ceiling of ln(12/.05) divided by the exponent coefficient. It is not the smallest necessary dimension. These four points span only a small subspace, making a general bound conservative.

## Realization versus probability guarantee

The screen uses a fixed pseudo-random seed and Box–Muller transform for reproducibility. That finite construction is separate from the ideal independent Gaussian distribution assumed by the theorem. Changing seed changes realized errors, not the bound at fixed m,ε. One successful or failed seed neither proves nor disproves the theorem.

Blue points show six ratios; orange horizontal lines mark one and 1±ε. Connecting lines guide reading by pair index and are not a density. Do not mix squared-distance tolerance with the identical numerical tolerance on unsquared distances.

## Scope

The guarantee concerns points fixed before the matrix. It does not preserve all vectors in a continuous ambient space under dimension reduction, nor arbitrary points chosen adversarially after seeing the matrix. Learning or retrieval quality requires further arguments beyond pairwise-distance preservation.

## Exercises with explanations

1. What is the scaling? **Divide standard Gaussian entries by √m.**
2. How many pairs are checked? **Six.**
3. Does a failure bound of one predict certain failure? **No: it is an uninformative bound.**

Further reading: [Stanford CS 368, Lecture 4 — Johnson–Lindenstrauss](https://cs368-stanford.github.io/spring2022/lectures/lecture4-notes.pdf).
