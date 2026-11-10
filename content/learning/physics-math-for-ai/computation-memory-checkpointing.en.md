## Reverse derivatives have a storage cost

Backpropagation needs forward values to evaluate local derivatives. Retaining every layer state is straightforward, but activation storage grows with depth and batch size. Checkpointing trades additional forward execution for fewer retained intermediate states. The useful question is which states exist simultaneously, not merely how many checkpoints are named in a diagram.

The experiment executes a dense width-W chain h_l=tanh(A h_(l-1)), where every A entry is alpha/W and h0_i=x(i+1)/W. The scalar output is the final mean. A full tape and a block-recomputed reverse pass independently produce gradients with respect to the input vector and shared parameter alpha.

## Keep boundaries and reconstruct one block

For L layers and block length B, retain states at zero, B, 2B and the final layer. The last block can be shorter. There are ceil(L/B)+1 boundary vectors. During reversal, restore only the interior of one block, use its two retained boundaries, propagate adjoints backward and discard that block before reconstructing another.

$$
R=L-\lceil L/B\rceil,\qquad
V_{\mathrm{checkpoint}}=
\left(\lceil L/B\rceil+\min(B,L)+2\right)W.
$$

R counts recomputed forward layers because the final state of each block is already stored. V counts boundary states, the largest block's internal states and two adjoint vectors. Full-tape storage on the same convention is (L+3)W values. Keeping every reconstructed block simultaneously would invalidate the checkpoint estimate.

## Count specified operations rather than inventing runtime

A dense matvec performs W² scalar multiplications. Both schedules execute L reverse matvecs. The full tape executes L forward matvecs, while checkpointing adds R:

$$
M_{\mathrm{full}}=2LW^2,\qquad
M_{\mathrm{checkpoint}}=(3L-\lceil L/B\rceil)W^2.
$$

These counts exclude activation evaluation, derivative multiplications and parameter-gradient reductions. Storage estimates multiply the retained-value count by four or eight bytes. They exclude parameters, optimizer states, kernel temporaries and the independent full-tape comparator retained by this demonstration. Actual browser arithmetic uses JavaScript numbers; selecting a byte width changes the hypothetical storage model, not its arithmetic dtype.

Bandwidth, parallelism, cache behavior and kernel launch overhead affect measured runtime. Equal operation counts therefore do not imply equal speed. The model makes a particular tradeoff explicit without claiming to benchmark a GPU training implementation.

## Verify the gradients, not only the ledger

For L=16, W=8 and B=4, full-tape storage is 608 bytes at four bytes per value; checkpoint storage is 320. Dense matvec multiplications increase from 2048 to 2816. The actual recomputation count is twelve, and both reverse schedules agree on gradients. The blue and orange curves show storage and multiplication ratios across block sizes.

## Exercise: two endpoints and a practical choice

Set B=1. Every layer is a checkpoint, R=0, and storage matches the full tape. If one block spans the whole chain, all its interior states are restored together and peak state storage again matches the full tape. Explain why an intermediate block size can be useful. Then select eight-byte storage: estimated bytes double, while gradients and operation counts stay fixed. Design a separate timing experiment before claiming that the lower-memory schedule is faster.

Further reading: [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html). The schedule, counting convention and verification chain here are independently implemented.
