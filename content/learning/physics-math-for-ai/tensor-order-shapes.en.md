## Order, size and rank are different

Array order counts axes; shape lists their sizes. Shape (1,2,3) has order three and six entries. A singleton axis still counts unless removed. Some libraries call axis count “rank,” but this book separates it from matrix, CP and TT ranks.

$$
X\in\mathbb R^{B\times T\times C},\qquad
\operatorname{size}(X)=BTC,\quad\operatorname{order}(X)=3.
$$

Axis meaning comes from data and model conventions. Equal dimensions do not make a batch axis interchangeable with time or channel.

## Tensor versus coordinate array

An abstract tensor can be an element of a tensor product of spaces. Choosing bases gives an array:

$$
X=\sum_{i,j,k}X_{ijk}\,e_i\otimes f_j\otimes g_k.
$$

Fix those bases before treating coordinates as the object. A multidimensional dataset is not automatically a physical tensor field with a prescribed transformation law. The labs begin with fixed real bases.

## Follow an axis permutation

The example uses X[b,t,c]=100b+10t+c. Shape (1,2,3) gives flattened entries (0,1,2,10,11,12). Swapping the final axes gives shape (1,3,2), with

$$
\widetilde X[b,c,t]=X[b,t,c].
$$

The new flat values are (0,10,1,11,2,12). Entry X[0,1,2]=12 moves to X̃[0,2,1]. Track the coordinate correspondence, not just matching element counts.

## Reshape and storage order

Under last-axis-fastest row-major ordering,

$$
q=(bT+t)C+c.
$$

A reshape reinterprets that flat order; a permutation changes index roles. They generally differ. Views, strides, contiguity and copying depend on the actual library and input layout. This lab computes mathematical entry mappings rather than creating real memory views.

## Read the representation and apply it

The graph plots flat indices against values, not a geometry-preserving three-dimensional tensor object. The table lists every output coordinate for independent checking. Doubling batch size leaves order three but doubles entries.

Attention uses batch, head, token and feature axes; vision adds channel conventions. Shape-compatible code can still apply softmax or averaging over the wrong semantic axis. Name each axis before and after the operation.

## Exercises and answers

1. Shape (2,3,4): **order three, 24 entries**.
2. Keeping the singleton axis in (1,2,3) leaves **order three**.
3. A transpose is **not generally equivalent to a reshape**; verify index mapping.

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
