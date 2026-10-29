## Read a network as a sum

A tensor-network node represents an array. Edges identify indices to sum; open legs remain as output axes. Geometric distances in the drawing do not encode numerical magnitudes. Connectivity and index sizes determine the operation.

$$
Y_{il}=\sum_{j=1}^{m}\sum_{k=1}^{p}A_{ij}B_{jk}C_{kl}.
$$

The shapes are n×m, m×p and p×q. The result has shape n×q. The internal indices j and k disappear after summation. Start by identifying which axes each repeated index connects, rather than guessing from the placement of nodes.

## Two routes to the same result

Associativity in exact real arithmetic gives two contraction routes. They have the same result but different intermediate arrays and costs. The lab evaluates both routes separately and lists their actual output entries side by side.

$$
(AB)C=A(BC).
$$

The left route forms an n×p intermediate. The right route forms an m×q intermediate. Other implementations may fuse operations, but this lab counts ordinary dense matrix products with explicit intermediate arrays.

## Count scalar multiplications

Multiplying an n×m matrix by an m×p matrix uses m multiplications for each of np output entries. Applying that count to both stages yields

$$
C_L=nmp+npq,\qquad C_R=mpq+nmq.
$$

At n=2, m=3, p=4, q=2, the left route uses forty multiplications and the right uses thirty-six. Their intermediates contain eight and six entries. Changing the dimensions to 4,2,1,4 reverses the preference: the costs become twenty-four and forty. Neither association is universally cheaper.

The input arrays use A[i,j]=i+j+1, B[j,k]=1+j−k and C[k,l]=k+l+1 with zero-based indices. These asymmetric values make incorrect index order easier to detect than an all-ones example would.

## Connections to AI execution

Attention, chains of linear maps and larger tensor networks all require an execution order in addition to a mathematical expression. The number of possible orders grows with the network. This three-matrix demonstration does not implement a general contraction-path optimizer.

Displayed costs count scalar multiplications, not GPU time, bandwidth, parallelism or kernel-launch overhead. With general floating-point inputs, changing association may also change rounding. The small integer calculations here remain exactly representable, so the two outputs have zero difference.

## Exercises with answers

1. Do equal intermediate sizes guarantee equal total costs? **No: the remaining dimensions also enter the formulas.**
2. If every dimension is one, what are the costs? **Two multiplications for either route.**
3. Which indices survive? **The free indices i and l.**

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
