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

## A work-conserving schedule has an upper bound too

For a unit-task DAG, a greedy list scheduler executes P ready tasks whenever at least P are available. Divide ticks into full and underfull ones. Every underfull tick executes all ready tasks and shortens the longest remaining path by at least one, so there are at most S such ticks. Counting at least P tasks per full tick and at least one per underfull tick gives

$$
\max(W/P,S)\le T_P\le W/P+(1-1/P)S.
$$

The inequality concerns the ideal task model, not the CPU time spent finding ready tasks. The live implementation scans dependency records to construct its schedule; that diagnostic scheduling overhead is not a unit DAG task. The useful parallelism scale W/S explains why adding processors eventually stops helping even without memory costs.

![Actual unit-task list-schedule ticks and work/span bounds across processor counts, distinguished from wall-clock timing.](/assets/learning/algorithm-work-span-en.svg)

| P for N=8, bandwidth 1 | Integer lower bound | Actual list ticks |
|---|---:|---:|
| 1 | 36 | 36 |
| 4 | 9 | 10 |
| 16 | 3 | 4 |

```python
from fractions import Fraction as F
tasks=[]
for count in (2,3,3,3,3,3,3,2):
    layer=[]
    for _ in range(count):layer.append(len(tasks));tasks.append([])
    while len(layer)>1:
        nxt=[]
        for k in range(0,len(layer),2):
            if k+1==len(layer):nxt.append(layer[k])
            else:nxt.append(len(tasks));tasks.append([layer[k],layer[k+1]])
        layer=nxt
depth=[]
for deps in tasks:depth.append(1+max([depth[d] for d in deps],default=0))
W,S=len(tasks),max(depth)
for P in (1,4,16):
    done=set();ticks=0
    while len(done)<W:
        ready=[i for i,deps in enumerate(tasks) if i not in done and all(d in done for d in deps)][:P]
        assert ready;done.update(ready);ticks+=1
    assert max(F(W,P),S)<=ticks<=F(W,P)+(1-F(1,P))*S
    print(P,W,S,ticks)
data=[1e16,1.,-1e16,1.];serial=0.
for x in data:serial+=x
balanced=(data[0]+data[1])+(data[2]+data[3])
assert serial==1 and balanced==0 and sum(map(F,data))==2
```

The independent DAG example checks both bounds. Arithmetic accuracy is another constraint: for stored numbers (10¹⁶,1,−10¹⁶,1), a serial double sum is 1 and the indicated balanced sum is 0, while the exact sum is 2. A tree can improve an error bound without improving every input. Reporting reduced work and more parallelism does not remove the need to test the reordered numerical computation.

## Worked example and exercise

With N=8 and bandwidth one, endpoint rows have two nonzeros and six interior rows have three. Thus nnz=22, W=44-8=36 and S=3. One processor takes 36 ticks. Four processors have a lower bound of nine ticks, while this actual greedy schedule takes ten. A feasible schedule can exceed a lower bound without being incorrect.

Exercise: set bandwidth zero and N=P=8. Find W, S and T. Solution: each row has only its diagonal product, so W=8 and S=1. Execute all eight tasks in one tick, giving T=1. Increasing P to sixteen cannot reduce time below span one.

## Sparse and parallel AI computation

Sparse attention, graph aggregation and sparse linear layers do not obtain guaranteed wall-time speedups merely from fewer nonzeros. Index storage, irregular access, synchronization and occupancy matter. A balanced reduction also changes addition association, so its floating-point output need not match a serial sum bit for bit. Record accuracy tolerances and data traffic alongside work and span. This model supplies a verifiable dependency baseline; actual device timing is a separate experiment.

Reference: [MIT performance engineering course](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
