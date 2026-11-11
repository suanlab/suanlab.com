## Connect many axes through a chain

An order-d array with n coordinates per axis contains n to the power d entries. A Tensor Train represents it through small cores connected by internal indices. To evaluate an entry, fix every physical coordinate and sum over the internal indices.

$$
X_{i_1\cdots i_d}=\sum_{a_1,\ldots,a_{d-1}}G^{(1)}_{1i_1a_1}\cdots G^{(d)}_{a_{d-1}i_d1}.
$$

The boundary bond dimensions are one. Each internal size r_k is a bond dimension, distinct from a physical axis size. In the diagram, external legs carry physical indices; edges between cores carry summed bond indices.

## Contract three explicit cores

The target is X=e000+t e111. With bond dimension two, separate paths reconstruct the two corners. The middle core prevents the paths from mixing. The supplied bond-one construction retains only the first corner.

$$
X_{ijk}=\sum_{a,b}G^{(1)}_{1ia}G^{(2)}_{ajb}G^{(3)}_{bk1}.
$$

For t=1, the bond-two reconstruction has zero error. The bond-one reconstruction has Frobenius error one. These are properties of the displayed construction, rather than a universal error rule for truncating arbitrary tensors.

## Gauge freedom: different cores, same array

Multiply one path in the first core by g and divide the corresponding path in the next core by g. Every product cancels the change. Moving the slider from one to two changes the cores while leaving every reconstructed entry unchanged.

$$
(AG)(G^{-1}B)=AB,\qquad G\text{ invertible}.
$$

The control demonstrates a diagonal scaling instance of this identity. Its positive range excludes zero, where the inverse would not exist. General floating-point settings can introduce tiny rounding residuals. Core values are consequently not a unique description of the represented array.

## Allocated dimensions versus minimal ranks

The minimal TT rank at a cut equals the rank of the matrix unfolding whose rows contain the first k physical indices. Changing the axis order changes these cuts. At t=0 the minimal internal ranks are one, even if both allocated bonds remain two. The allocation still stores sixteen scalars.

The lab explicitly contracts constructed cores; it does not implement general TT-SVD. In applications, consider ordering, approximation error and contraction cost alongside parameter counts. TT formats can describe structured high-dimensional functions compactly, but an arbitrary dense array need not admit small accurate bonds.

## Prove the minimal rank at every cut

Fix the order of a nonzero finite array X. At cut k, contract all cores on the left into L_k and all cores on the right into R_k. The unfolding satisfies

$$
X^{\langle k\rangle}=L_kR_k,\qquad
\operatorname{rank}X^{\langle k\rangle}\le r_k.
$$

This proves a lower bound for every exact TT representation. The bound is achievable simultaneously. Factor the first unfolding as U₁V₁ using its exact matrix rank, with U₁ having full column rank, and reshape U₁ into the first core. Reshape V₁ so its row index combines the retained bond with the second physical index, then factor again.

The original second unfolding equals (U₁⊗I) times that reshaped remainder. Since U₁⊗I has full column rank, multiplication preserves the remainder’s rank. Therefore the next retained rank is exactly the original second unfolding rank. Repeating this argument produces all cores and proves equality at every cut. Zero-array representations need a separate positive-bond convention; the present target always has its first corner equal to one.

These ranks describe the array being represented. With t≠0 the lab’s exact target needs two at each internal cut, while its configured bond-one approximation represents a different, rank-one array.

## Prove that bounded TT ranks admit a best approximation

Fix positive integer rank bounds r_k. The represented arrays with each unfolding rank at most r_k form a closed set: every forbidden larger matrix minor is a continuous polynomial that must vanish. The intersection over finitely many cuts is still closed. The exact factorization argument identifies this set with the specified TT rank bounds.

It contains zero. A minimizing sequence for Frobenius error can be chosen with bounded error, hence bounded array norm by the triangle inequality. In finite dimensions a subsequence converges. Closedness keeps the limit feasible and continuity of the norm makes it attain the infimum. Thus a best bounded-TT-rank approximation exists. This array-level result does not guarantee convergence of a core-parameter optimizer or bounded gauge-scaled core values. Compare the [CP nonattainment example](/book/online/physics-math-for-ai/tensor-low-rank-approximation/).

## Change the ordering and calculate the ranks

Take four binary indices and X_ijkl=δ_ik δ_jl. In order (i,j,k,l), the middle unfolding is the 4×4 identity and has rank four. In order (i,k,j,l), it is the outer product (1,0,0,1)ᵀ(1,0,0,1), of rank one.

| Physical-axis order | Minimal internal ranks | Stored core scalars |
|---|---|---|
| (i,j,k,l) | (2,4,2) | 40 |
| (i,k,j,l) | (2,1,2) | 16 |

The count is Σₖ nₖrₖ₋₁rₖ with boundary bonds one. The dense array has sixteen entries, so the table shows an ordering advantage without promising compression below the dense size.

For the reordered case, explicit cores are G¹_{1ia}=δ_ia, G²_{ak1}=δ_ak, G³_{1jb}=δ_jb and G⁴_{bl1}=δ_bl. Summing a,b immediately gives δ_ikδ_jl. The order preserves the paired indices together instead of carrying both unresolved pairs across the middle cut.

## Reproduce all cuts in Python

Exact rational Gaussian elimination computes three unfolding ranks for each order. A separate contraction checks the four displayed cores at all sixteen entries. This is a separate four-axis example; the interactive lab still contracts its original three cores and does not run general TT-SVD.

```python
from fractions import Fraction as Q
from itertools import product

def matrix_rank(matrix):
    a = [[Q(x) for x in row] for row in matrix]
    pivot_row = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(pivot_row,len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[pivot_row], a[pivot] = a[pivot], a[pivot_row]
        scale = a[pivot_row][col]
        a[pivot_row] = [x/scale for x in a[pivot_row]]
        for i in range(len(a)):
            if i != pivot_row:
                scale = a[i][col]
                a[i] = [x-scale*y for x,y in zip(a[i],a[pivot_row])]
        pivot_row += 1
        if pivot_row == len(a):
            break
    return pivot_row

def target(index):
    i,j,k,l = index
    return int(i == k and j == l)

def unfolding(order, cut):
    matrix = []
    for row in product(range(2),repeat=cut):
        values = []
        for col in product(range(2),repeat=4-cut):
            index = [0]*4
            for axis,value in zip(order,row+col):
                index[axis] = value
            values.append(target(index))
        matrix.append(values)
    return matrix

for order, expected, count in [((0,1,2,3), [2,4,2], 40),
                                ((0,2,1,3), [2,1,2], 16)]:
    ranks = [matrix_rank(unfolding(order,k)) for k in [1,2,3]]
    assert ranks == expected
    bonds = [1]+ranks+[1]
    assert sum(2*a*b for a,b in zip(bonds,bonds[1:])) == count
    print(order, ranks, count)

# Explicit reordered cores: delta(i,a), delta(a,k), delta(j,b), delta(b,l).
for index in product(range(2),repeat=4):
    i,j,k,l = index
    reconstructed = sum(int(i==a)*int(a==k)*int(j==b)*int(b==l)
                        for a,b in product(range(2),repeat=2))
    assert reconstructed == target(index)
```

Source connection: [Oseledets, Tensor-Train Decomposition](https://doi.org/10.1137/090752286), Theorem 2.1 and Corollary 2.4, establishes unfolding-rank representations and best-approximation existence. The binary-index ordering example and exact checks here are independently constructed. TT-SVD truncation bounds and data-driven rank selection remain further study.

## Exercises with answers

1. Does changing g change the target? **No: adjacent factors cancel.**
2. How many scalars does the bond-one construction store? **Six: two in each core.**
3. Must an allocated bond of size two have minimal rank two? **No; set t=0.**

Further reading: [Oseledets, Tensor-Train Decomposition (2011)](https://doi.org/10.1137/090752286).The small arrays, reconstructions and cost examples here are independently constructed.
