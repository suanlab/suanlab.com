## Less work differs from simultaneous execution

Sparsity removes multiplication by zeros. Parallelism executes independent tasks at the same time. These are different optimizations. This experiment computes dense multiplication, CSR multiplication and an actually scheduled balanced-reduction DAG for a banded matrix with diagonal two and selected off-diagonals minus one. Blue and orange are CSR and dense outputs; scheduled outputs are checked separately in the table.

CSR stores row pointers, column indices and values. Row i occupies entries from rowptr[i] to rowptr[i+1]-1. The teaching setup also creates dense matrices and diagnostics, so its entire setup is not claimed to cost O(nnz). Separate core multiplication from representation construction.

## Build an actual work DAG

Create one unit task for each nonzero product, followed by a binary addition tree for each row. An addition becomes ready only after both children finish. An unpaired final item passes to the next layer. Every row contains a diagonal nonzero, so reductions are nonempty.

$$
W=\sum_i\{r_i+(r_i-1)\}=2\,\mathrm{nnz}-N,\qquad
S=\max_i\{1+\lceil\log_2r_i\rceil\}.
$$

Here r_i is the row nonzero count, W is total product-plus-addition tasks, and S is longest dependency depth. Equal task duration, free data access and independent processors are explicit idealizations. This is not a cycle prediction for sparse index loads or a GPU kernel.

## Schedule on finite processors

$$
T_P\ge\max\{\lceil W/P\rceil,S\}.
$$

At most P tasks can execute simultaneously, and no schedule can bypass a dependency chain. The implementation selects up to P ready tasks in index order at the start of each tick and actually computes their values. It does not claim this greedy list schedule is always optimal. Independent checks verify unique execution, earlier completion of every child, processor capacity and correct outputs.

## Worked example and exercise

With N=8 and bandwidth one, endpoint rows have two nonzeros and six interior rows have three. Thus nnz=22, W=44-8=36 and S=3. One processor takes 36 ticks. Four processors have a lower bound of nine ticks, while this actual greedy schedule takes ten. A feasible schedule can exceed a lower bound without being incorrect.

Exercise: set bandwidth zero and N=P=8. Find W, S and T. Solution: each row has only its diagonal product, so W=8 and S=1. Execute all eight tasks in one tick, giving T=1. Increasing P to sixteen cannot reduce time below span one.

## Sparse and parallel AI computation

Sparse attention, graph aggregation and sparse linear layers do not obtain guaranteed wall-time speedups merely from fewer nonzeros. Index storage, irregular access, synchronization and occupancy matter. A balanced reduction also changes addition association, so its floating-point output need not match a serial sum bit for bit. Record accuracy tolerances and data traffic alongside work and span. This model supplies a verifiable dependency baseline; actual device timing is a separate experiment.

Reference: [MIT performance engineering course](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
