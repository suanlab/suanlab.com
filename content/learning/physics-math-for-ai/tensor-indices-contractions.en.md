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

## AI applications and exercises

Attention contracts the feature index in QᵢₐKⱼₐ to produce scores Sᵢⱼ. Batch and head axes need their own semantics: leave them free or explicitly state any intended summation. Shape agreement cannot catch every incorrect axis choice.

The plot shows indexed output values; the table retains their free coordinates. A zero covector gives a zero array without removing its shape.

1. Contracting the last axis of (3,4,5) with (5,) gives **(3,4)**.
2. The zero covector gives **a (2,2) zero array**.
3. Equal axis lengths **can conceal a wrong contraction**; check actual entries.

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
