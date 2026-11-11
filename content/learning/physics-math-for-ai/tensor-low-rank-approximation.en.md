## Specify the approximation problem

A low-rank approximation asks how much of an array a small number of components can describe. State both the rank notion and the error norm before comparing answers. Here the components are orthogonal coordinate outer products and the error uses the Frobenius norm, the square root of the sum of squared entries.

$$
X=a\,e_0\otimes e_0\otimes e_0+b\,e_1\otimes e_1\otimes e_1.
$$

The defaults are a=3 and b=1. The plot and table show all eight original entries and the reconstruction. Discrete coordinates are displayed as discrete values, rather than interpolated into a continuous field.

## Compute the discarded energy

Orthogonality makes the squared residual equal the sum of the squared discarded weights. Retaining the first default component therefore gives error one.

$$
\|X\|_F^2=a^2+b^2,\qquad \|X-\widehat X\|_F^2=\sum_{s\text{ discarded}}\lambda_s^2.
$$

Components are selected by absolute magnitude, so a large negative weight is not treated as unimportant. Ties are resolved by coordinate order in the implementation. That convention does not establish a unique optimum: equal magnitudes allow different equally accurate selections.

## Retained energy and the zero case

$$
E_{\mathrm{retained}}=\frac{\sum_{s\text{ retained}}\lambda_s^2}{a^2+b^2}.
$$

Keeping one default component retains 9/10 of the energy. Keeping both retains all of it. Keeping neither gives an error equal to the original norm. If both original weights are zero, the fraction is undefined because its denominator is zero. The interface leaves it undefined rather than inventing a perfect score. The absolute reconstruction error remains zero.

## Where matrix intuition applies

For this special example, an unfolding has singular values |a| and |b|. The matrix rank-one approximation lower bound matches the error of retaining the larger coordinate component. That proves optimality here, rather than for arbitrary CP approximations.

General higher-order CP approximation differs from matrix SVD. A set of tensors with bounded CP rank need not be closed, and a best approximation can fail to exist. Tucker and TT truncations require their own error analyses. Do not transfer the matrix Eckart–Young conclusion to all tensor formats without checking assumptions.

For model compression, reconstruction error is only one measurement. Evaluate prediction quality and execution cost separately; a small tensor error does not automatically preserve a task metric.

## Construct a rank-two sequence whose limit has rank three

Use unconstrained real CP rank. Let e₀,e₁ be the coordinate basis of R² and use the Frobenius norm on 2×2×2 arrays. For real ε≠0, define

$$
W=e_1\otimes e_0\otimes e_0+e_0\otimes e_1\otimes e_0+e_0\otimes e_0\otimes e_1,
\qquad
T_\varepsilon=\frac{(e_0+\varepsilon e_1)^{\otimes3}-e_0^{\otimes3}}{\varepsilon},\quad\varepsilon\ne0.
$$

The second expression uses two CP terms, so its CP rank is at most two. Its k=0 slice is [[0,1],[1,ε]], with determinant −1, proving that its rank is exactly two. Expanding the product gives three desired terms, three terms of coefficient ε with two e₁ factors, and one term of coefficient ε². Orthogonality of these coordinate entries yields

$$
\|T_\varepsilon-W\|_F^2=3\varepsilon^2+\varepsilon^4\longrightarrow0.
$$

To establish the failure of best approximation, we must also prove that W cannot have CP rank two.

## Prove the limiting CP rank by two slices

Fix the third index to obtain W’s slices S₀=[[0,1],[1,0]] and S₁=[[1,0],[0,0]]. If W had a two-term CP representation, both slices could be written S_k=U D_k Vᵀ for diagonal D_k and 2×2 factor matrices U,V. Invertibility of S₀ forces U,V,D₀ to be invertible. Consequently

$$
S_1S_0^{-1}=U D_1D_0^{-1}U^{-1}
\quad\text{would be diagonalizable},\qquad
S_1S_0^{-1}=\begin{pmatrix}0&1\\0&0\end{pmatrix}.
$$

The actual matrix is nonzero but its square is zero. Every eigenvalue is zero, so a diagonalizable matrix with those eigenvalues would be zero, a contradiction. Thus W has rank greater than two; its displayed three-term representation proves rank exactly three.

The infimum of rank-at-most-two approximation error is zero, because T_ε approaches W. No feasible tensor attains that infimum. This proves nonexistence of a best approximation for this target, rather than failure of one optimizer. The bounded-CP-rank set is not closed. Its rank-two closure contains W. Its border rank is exactly two: the S₀ slice has determinant −1, whereas a CP-rank-at-most-one sequence would have slice determinant zero, preserved by limits.

## Observe growing terms despite a bounded answer

The two individual CP terms have norms (1+ε²)³ᐟ²/|ε| and 1/|ε|, although their difference tends to a tensor of norm √3. Rescaling individual factors cannot remove the norm growth of the complete rank-one terms.

| ε | Squared approximation error | Norm of the second CP term |
|---|---|---|
| 1/2 | 13/16 | 2 |
| 1/8 | 193/4096 | 8 |
| 1/32 | 3073/1048576 | 32 |

For a numerical cancellation example, apply H=[[1,1],[1,−1]] to every mode. This invertible transformation preserves CP rank. The limiting entries become 3−2(i+j+k). At ε=2⁻⁵⁶, binary64 rounds every factor 1±ε to one; directly subtracting the two outer products therefore produces an all-zero array. Exact arithmetic still gives entries near (3,1,1,−1,1,−1,−1,−3).

## Reproduce the sequence and cancellation in Python

The code checks exact errors, the nilpotent slice product and the transformed entry formulas. It then contrasts exact fractions with the explicitly evaluated binary64 factors. The interactive lab remains the earlier orthogonal two-corner example, whose rank-one optimum does exist.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod

indices = list(product(range(2), repeat=3))
w = [Q(sum(index) == 1) for index in indices]

def outer(v):
    return [prod(v[i] for i in index) for index in indices]

for epsilon, expected in [(Q(1,2), Q(13,16)),
                          (Q(1,8), Q(193,4096)),
                          (Q(1,32), Q(3073,1048576))]:
    approximation = [(a-b)/epsilon for a, b in zip(outer([1,epsilon]), outer([1,0]))]
    squared_error = sum((a-b)**2 for a, b in zip(approximation, w))
    assert squared_error == expected == 3*epsilon**2 + epsilon**4
    print(epsilon, squared_error, 1/epsilon)

s0 = [[0,1],[1,0]]
s1 = [[1,0],[0,0]]
def multiply(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
j = multiply(s1,s0)  # s0 is its own inverse
assert j == [[0,1],[0,0]]
assert multiply(j,j) == [[0,0],[0,0]]

epsilon = Q(1,2**56)
exact, raw_float, limit = [], [], []
for index in indices:
    signs = [1 if i == 0 else -1 for i in index]
    factors = [1+epsilon*s for s in signs]
    exact.append((prod(factors)-1)/epsilon)
    raw_float.append((prod(float(a) for a in factors)-1)/float(epsilon))
    limit.append(sum(signs))
assert raw_float == [0.0]*8
assert all(abs(float(a)-b) < 1e-12 for a,b in zip(exact,limit))
print('raw binary64', raw_float)
print('transformed limiting tensor', limit)
```

Source connection: [de Silva and Lim, arXiv v2](https://arxiv.org/abs/math/0607647v2), Proposition 4.6 and §4.3, studies this rank-three limiting construction and diverging coefficients. The slice-product proof and binary64 calculation above are worked derivations for this book. Research practice should record component norms and initialization alongside reconstruction error; this example does not assert that every CP problem lacks a solution.

## Exercises with answers

1. For a=−3 and b=1, which component survives? **The one with weight −3.**
2. What is the retained fraction for a zero input? **Undefined: it is 0/0.**
3. What happens when both components are retained? **The reconstruction error is zero.**

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
