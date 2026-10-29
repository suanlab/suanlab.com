## Two ways to factor an array

A matrix can be written as a sum of outer products. CP extends that construction to several axes: each term contains one vector per axis and a scalar weight. Tucker chooses coordinates for each axis and stores their interactions in a smaller core. We work with real, three-axis arrays and fixed bases.

$$
X_{ijk}\approx\sum_{s=1}^{r}\lambda_s u_{is}v_{js}w_{ks}.
$$

Each CP term separates the three indices. CP rank is the smallest number of terms required for an exact representation. It is neither the number of axes nor an ordinary matrix rank. Selecting a term budget in this lab does not solve general CP rank determination.

## What a Tucker core records

$$
\widehat X_{ijk}=\sum_{a,b,c=1}^{r}G_{abc}U_{ia}V_{jb}W_{kc}.
$$

The core records how the chosen coordinates interact. Unlike a diagonal CP core, a Tucker core can contain interactions among different component indices. Its three dimensions may differ in general; the lab uses equal dimensions to keep the comparison readable.

## Reconstruct a concrete example

The default 2×2×2 array has value 2 at (0,0,0), value 1 at (1,1,1), and zeros elsewhere. Retaining both coordinate outer products reconstructs every entry exactly. Keeping only the stronger component loses the second corner.

$$
\|X-\widehat X\|_F=\sqrt{\sum_{i,j,k}(X_{ijk}-\widehat X_{ijk})^2}=1.
$$

Read all eight rows to compare the input, CP reconstruction and independently evaluated Tucker reconstruction. The diagram shows the Tucker core and three factor nodes. Internal edges carry summed indices of size r. External legs i, j and k remain as output axes, each of size two.

## Representation is not automatically compression

Including explicit weights, CP stores 7r scalars here. Tucker stores r³+6r. At r=2 these are 14 and 20 scalars, both larger than the original eight. A factorization can be exact and still use more storage. Compression becomes attractive when a sufficiently small rank describes a much larger array accurately.

This lab reconstructs a deliberately orthogonal coordinate example. It does not fit arbitrary data with CP optimization or HOSVD. Approximation quality, optimization cost and uniqueness require separate analysis for general tensors. Changing signs changes values without invalidating the representation.

## Exercises with answers

1. With no retained components, does the output lose its shape? **No: it remains a zero array of the original shape.**
2. How many scalars do the two formats store at r=1? **Seven each.**
3. Does exact reconstruction guarantee smaller storage? **No; compare the r=2 example.**

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
