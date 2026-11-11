## Free and summed indices

Contraction multiplies and sums designated indices. For shape (2,2,2) and a length-two covector,

$$
Y_{ij}=\sum_{k=0}^1X_{ijk}v_k.
$$

Free indices i,j remain in the output; summed k disappears, leaving shape (2,2). Elementwise multiplication retains that axis and is a different operation. With Einstein notation, check that free indices match on both sides even when summation symbols are omitted.

## Work the reference

Let Xᵢⱼₖ=1+4i+2j+k and v=(1,2):

$$
Y_{00}=1\cdot1+2\cdot2=5,\qquad
Y_{01}=3\cdot1+4\cdot2=11.
$$

The other entries are 17 and 23. Four outputs each require two multiplications and one addition: eight and four in a direct implementation. These operation counts are not measured hardware runtimes.

## Axis choice changes the result

Contracting the first axis instead leaves (j,k):

$$
Z_{jk}=\sum_{i=0}^1v_iX_{ijk}.
$$

Its values are (11,14,17,20). The shape remains (2,2), hiding a possible semantic bug. This lab preserves the original relative order of free axes. Use explicit output indices when a library’s ordering could differ.

## Dual spaces explain the pairing

Natural abstract contraction pairs a space with its dual. Their coordinates transform oppositely so the resulting scalar or remaining tensor transforms consistently:

$$
\langle v^*,x\rangle=\sum_i v_i^*x_i.
$$

A real Euclidean orthonormal basis identifies vectors and covectors through the inner product. Complex Hermitian products need conjugation; general coordinates may need a metric. Do not assume an arbitrary array expression is automatically coordinate invariant.

## Change a basis without changing the scalar

Take a real vector with old coordinates x=(2,1)ᵀ and a covector with row coordinates α=(3,4). Their contraction is αx=10. If the new basis columns are E′=ES with invertible S, then the same vector and functional have coordinates

$$
x'=S^{-1}x,\qquad \alpha'=\alpha S,\qquad
\alpha'x'=\alpha SS^{-1}x=\alpha x.
$$

The covector acts on vectors; its coordinates therefore transform in the opposite way. Choose a shear S=[[1,1],[0,1]]. Its new basis is not orthonormal.

| Quantity | Old coordinates | New coordinates |
|---|---|---|
| Vector x | (2,1)ᵀ | (1,1)ᵀ |
| Covector α | (3,4) | (3,7) |
| Euclidean gradient vector g | (3,4)ᵀ | (−1,4)ᵀ |
| Metric matrix | I | [[1,1],[1,2]] |
| Natural contraction | 10 | 10 |

Reusing the old row α with x′ would give seven. Using the new gradient vector with an ordinary coordinate dot product would instead give three. Neither is the invariant contraction.

## Distinguish a gradient vector from its covector

For the scalar function f(x)=3x₁+4x₂, α=df is its differential. The old Euclidean gradient is g=(3,4)ᵀ. In the new basis, the metric and gradient satisfy

$$
M'=S^TS,\qquad g'=S^{-1}g=(M')^{-1}(\alpha')^T,
\qquad (g')^TM'x'=\alpha'x'=10.
$$

Indeed, (g′)ᵀM′=gᵀS=αS. The physical squared length of x is still five, although the ordinary sum of squares of x′ is two. This is why identifying a vector and a covector requires the metric in nonorthonormal coordinates. The earlier [gradient unit](/book/online/physics-math-for-ai/gradients/) derives the same metric relationship from steepest descent.

## Check the coordinate transformations in Python

Exact fractions check the inverse, transformed covector, metric gradient and pairing through separate matrix products. This example changes coordinates in a two-dimensional space; the existing three-axis contraction lab above keeps its bases fixed.

```python
from fractions import Fraction as Q

def transpose(a):
    return [list(row) for row in zip(*a)]

def multiply(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def inverse(a):
    x, y = a[0]
    z, w = a[1]
    determinant = x*w-y*z
    assert determinant != 0
    return [[w/determinant, -y/determinant], [-z/determinant, x/determinant]]

s = [[Q(1), Q(1)], [Q(0), Q(1)]]
x = [[Q(2)], [Q(1)]]
alpha = [[Q(3), Q(4)]]
x_new = multiply(inverse(s), x)
alpha_new = multiply(alpha, s)
metric = multiply(transpose(s), s)
g_new = multiply(inverse(metric), transpose(alpha_new))
assert x_new == [[1], [1]] and alpha_new == [[3, 7]]
assert g_new == [[-1], [4]]
assert multiply(alpha, x) == multiply(alpha_new, x_new) == [[10]]
assert multiply(transpose(g_new), multiply(metric, x_new)) == [[10]]
assert multiply(transpose(g_new), x_new) == [[3]]
assert multiply(alpha, x_new) == [[7]]
assert multiply(transpose(x_new), multiply(metric, x_new)) == [[5]]
print(x_new, alpha_new, g_new, metric)
```

Research exercise: replace the shear by another invertible matrix and verify the invariant pairing. For a higher-order contraction, transform every free and summed slot according to its own space and dual-space convention. Matching array shapes alone cannot establish that transformation law.

## AI applications and exercises

Attention contracts the feature index in QᵢₐKⱼₐ to produce scores Sᵢⱼ. Batch and head axes need their own semantics: leave them free or explicitly state any intended summation. Shape agreement cannot catch every incorrect axis choice.

The plot shows indexed output values; the table retains their free coordinates. A zero covector gives a zero array without removing its shape.

1. Contracting the last axis of (3,4,5) with (5,) gives **(3,4)**.
2. The zero covector gives **a (2,2) zero array**.
3. Equal axis lengths **can conceal a wrong contraction**; check actual entries.

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
