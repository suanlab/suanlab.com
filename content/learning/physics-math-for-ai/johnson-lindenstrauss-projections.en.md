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

## Derive the chi-square tail constants

Under an ideal independent Gaussian projection, the squared-norm ratio of a fixed nonzero vector is Z/m, Z~χ²_m. Completing the square in a Gaussian integral gives E exp(tG²)=(1−2t)^(−1/2) for t<1/2. Independence then gives

$$
E e^{tZ}=(1-2t)^{-m/2},\qquad t<1/2.
$$

For Z≥m(1+ε), apply Markov with t>0 and optimize at t=ε/[2(1+ε)]. For Z≤m(1−ε), use t<0 and optimize at −ε/[2(1−ε)]. For 0<ε<1 this produces two rates.

$$
a_+(\varepsilon)=\frac{\varepsilon-\ln(1+\varepsilon)}2,\qquad
 a_-(\varepsilon)=\frac{-\varepsilon-\ln(1-\varepsilon)}2,
$$

$$
P\left(\left|\frac Zm-1\right|\geq\varepsilon\right)
\leq e^{-m a_+}+e^{-m a_-}.
$$

Both rates are positive. The inequality ln(1+ε)≤ε−ε²/2+ε³/3 gives a_+≥ε²/4−ε³/6, and a_- is no smaller. This recovers the simpler two-sided bound above. Logarithms here are natural and tolerance concerns squared distances.

## Pairwise dependence does not prevent a union bound

For N points fixed before projection, the number of distinct nonzero pairs is L≤N(N−1)/2. Sharing one matrix makes their errors dependent, but independence between pairs is unnecessary. Adding the bounds gives total failure at most L(e^(−ma_+)+e^(−ma_-)).

For four points, six pairs and target failure .05, compare the following integer dimensions. Two-tail is the smallest integer making this two-exponential bound at most .05; Coarse is the earlier simpler bound's sufficient dimension. Neither is claimed to be the smallest dimension actually needed for an embedding.

| ε | Two-tail | Coarse |
|---|---|---|
| 0.25 | 367 | 421 |
| 0.50 | 102 | 132 |
| 0.80 | 46 | 74 |

The code checks the Gaussian MGF by direct numerical integration and searches integer dimensions. Integration is an approximate check, not a replacement for the probability proof. The screen retains its simpler bound and fixed-seed experiment; this table is a separate theoretical comparison.

## Vectors chosen after seeing the matrix require a different claim

A linear map with m<d has a nonzero nullspace vector. Choosing that vector after inspecting the matrix gives projected norm zero, violating relative preservation for ε<1. Distinguish a finite set fixed in advance from a claim about all of R^d. Learning a map or selecting evaluation points using it requires additional arguments.

```python
from math import exp, log1p, log, ceil, sqrt, pi, isclose

for eps in [.25, .5, .8]:
    upper_rate = (eps-log1p(eps))/2
    lower_rate = (-eps-log1p(-eps))/2
    coefficient = eps*eps/4-eps**3/6
    assert min(upper_rate, lower_rate) >= coefficient
    def union(m):
        return 6*(exp(-m*upper_rate)+exp(-m*lower_rate))
    m = 1
    while union(m) > .05:
        m += 1
    assert union(m-1) > .05
    coarse = ceil(log(12/.05)/coefficient)
    print("epsilon, two-tail sufficient m, coarse m:", eps, m, coarse)
    for t in [eps/(2*(1+eps)), -eps/(2*(1-eps))]:
        def f(x):
            return exp(-(.5-t)*x*x)/sqrt(2*pi)
        steps, lo, hi = 4000, -12, 12
        h = (hi-lo)/steps
        integral = f(lo)+f(hi)
        integral += sum((4 if j % 2 else 2)*f(lo+j*h) for j in range(1, steps))
        integral *= h/3
        assert isclose(integral, 1/sqrt(1-2*t), rel_tol=1e-10)
print("MGF integration checks passed; these do not measure seed-specific failure probabilities.")
```

## Questions for research use

1. Does dependence between pair errors invalidate a union bound? **No: their individual probability bounds can still be added.**
2. What is minimal in the table? **The integer satisfying the specified probability bound.**
3. What belongs in the reproduction record? **Fixed points, distribution, dimensions, squared-error tolerance, failure level, and the distinction between seed and theory.**

For background, see [Stanford CS 368, Lecture 4](https://cs368-stanford.github.io/spring2022/lectures/lecture4-notes.pdf). The optimized Chernoff tails and integer table here are computed for the stated squared-distance convention.

## Exercises with explanations

1. What is the scaling? **Divide standard Gaussian entries by √m.**
2. How many pairs are checked? **Six.**
3. Does a failure bound of one predict certain failure? **No: it is an uninformative bound.**

Further reading: [Stanford CS 368, Lecture 4 — Johnson–Lindenstrauss](https://cs368-stanford.github.io/spring2022/lectures/lecture4-notes.pdf).
