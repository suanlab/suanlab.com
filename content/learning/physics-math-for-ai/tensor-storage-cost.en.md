## Count stored scalars first

An order-d dense array with n entries per axis stores n to the power d scalars. Each additional axis multiplies the count by n. Factorized formats store smaller factors and cores instead. Fix the format and rank budget before comparing counts.

$$
N_{\mathrm{dense}}=n^d,\qquad N_{\mathrm{CP}}=r(dn+1).
$$

The CP formula explicitly stores d vectors of length n and one weight for each component. Absorbing weights into factors changes the accounting convention. This calculator consistently includes explicit weights.

## Count Tucker and TT cores

Tucker stores d factor matrices plus an order-d core with every core axis of size r. TT requires separate counts for the two boundary cores and the internal cores.

$$
N_{\mathrm{Tucker}}=dnr+r^d,\qquad N_{\mathrm{TT}}=2nr+(d-2)nr^2.
$$

At d=2 there are no internal cores, so the second TT term vanishes. The formula counts allocated dimensions, not necessarily minimal ranks. The controls even allow r greater than n to expose the cost of redundant allocations.

## Work through the numbers

For d=6, n=4, r=2, the counts are 4096 dense, 50 CP, 112 Tucker and 80 TT scalars. With four bytes per scalar, dense storage takes 16384 bytes and TT takes 320 bytes. In contrast, d=2, n=2, r=4 gives four dense scalars but sixteen TT scalars. A factorized representation is not automatically smaller.

$$
M=bN,\qquad R=\frac{N_{\mathrm{dense}}}{N_{\mathrm{TT}}}.
$$

Switching from four-byte to two-byte storage halves bytes without changing scalar counts. The calculator does not simulate reduced-precision arithmetic or its error. The plot uses log10 of the stored count on its vertical axis: twice the plotted height does not mean twice the memory.

## What this comparison does not establish

The same numerical r has different meanings in CP, Tucker and TT. It does not imply equal expressive capacity or approximation error. These are allocation counts for specified formats. Determine a useful rank by evaluating data and downstream task quality.

Optimizer states, gradients, activations, reconstruction buffers, metadata and alignment are excluded. Materializing the full array during execution may erase the practical memory advantage. Implementations often need to contract stored cores directly and benchmark the actual execution path.

With fixed n and r, the displayed TT count grows linearly in d, while the Tucker core still grows exponentially. That favorable formula helps only when the required ranks remain manageable; an arbitrary target may need larger ranks as the problem grows.

## Exercises with answers

1. Does switching byte mode change the number of entries? **No.**
2. Does equal r guarantee equal accuracy across formats? **No.**
3. How does TT allocation grow with d at fixed n and r? **Linearly under this formula.**

Further reading: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).The small arrays, reconstructions and cost examples here are independently constructed.
