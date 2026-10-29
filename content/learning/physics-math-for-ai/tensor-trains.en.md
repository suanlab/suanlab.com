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

## Exercises with answers

1. Does changing g change the target? **No: adjacent factors cancel.**
2. How many scalars does the bond-one construction store? **Six: two in each core.**
3. Must an allocated bond of size two have minimal rank two? **No; set t=0.**

Further reading: [Oseledets, Tensor-Train Decomposition (2011)](https://doi.org/10.1137/090752286).The small arrays, reconstructions and cost examples here are independently constructed.
