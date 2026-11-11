## Equal element counts can have different memory costs

Transpose computes B[j,i]=A[i,j] by reading and writing the same N² elements. A row-major read of A produces column-wise stores to B, potentially losing cache-line reuse. Blocking works within small tiles to reuse lines from both arrays. Equal arithmetic or scalar-access counts therefore need not imply equal data movement.

The experiment creates the actual address trace and transpose output, then simulates a declared single-level cache. It does not read CPU hardware counters or measure DRAM traffic. Blue is selected-order cumulative misses; orange is naive-order misses under identical settings. Both perform exactly 2N² scalar accesses.

For long traces, the table shows the first 32 and last eight accesses. Totals and cache simulation use the entire trace.

## Addresses, lines and LRU

Each double occupies eight bytes. Scalar address iN+j belongs to A, and N²+jN+i belongs to disjoint B. A line holds L doubles, and the fully associative cache holds C lines.

$$
\mathrm{line}(a)=\lfloor a/L\rfloor,\qquad
\mathrm{LRU\ miss}\iff\mathrm{first\ access}\ \text{or}\ RD\ge C.
$$

RD counts distinct other lines since the previous access to that line. A miss evicts the least recently used resident, while a hit makes its entry most recent. Set conflicts, prefetching, multiple cache levels, TLBs and parallel coherence are excluded. In a real hierarchy, an L1 miss does not automatically mean a DRAM read.

## Store policy and final flushing

Write allocation loads a line even for a store miss. Modified lines become dirty and require writeback on eviction. Dirty lines still resident at completion are also flushed:

$$
\mathrm{transfers}=\mathrm{misses}+\mathrm{dirty\ evictions}+\mathrm{final\ flushes},\qquad
\mathrm{bytes}=8L\,\mathrm{transfers}.
$$

Counting misses alone would omit output writes. Conversely, write-through or non-temporal policies need a different accounting model. Fix the policy before comparing tile sizes and capacity.

## A cold-cache traffic lower bound and a nonmonotone tile choice

With cold write allocation, each distinct touched line must be fetched at least once. Each line containing output cells must eventually be written back at least once, either on eviction or final flushing. Hence transfers are at least |lines(A∪B)|+|lines(B)|. When the arrays are line-aligned and L divides N², this is 3N²/L. If allocations share a boundary line, use the actual line sets instead of that aligned formula.

For a fixed address trace, fully associative LRU has the stack property: a capacity-C cache contains the C most recently used distinct lines. Increasing C therefore cannot turn a hit into a miss. This does not say that enlarging a tile improves reuse, because changing the tile changes the trace itself.

| Tile at N=8,L=4,C=8 | Misses | Total line transfers |
|---|---:|---:|
| 1 | 80 | 144 |
| 4 | 32 | 48 |
| 8 | 80 | 144 |

```python
def simulate(tile):
    N,L,C=8,4,8;cache=[];misses=writes=0
    for ii in range(0,N,tile):
        for jj in range(0,N,tile):
            for i in range(ii,min(N,ii+tile)):
                for j in range(jj,min(N,jj+tile)):
                    for address,write in ((i*N+j,False),(N*N+j*N+i,True)):
                        line=address//L
                        found=next((k for k,(a,_) in enumerate(cache) if a==line),None)
                        if found is not None:
                            _,dirty=cache.pop(found);cache.append((line,dirty or write))
                        else:
                            misses+=1
                            if len(cache)==C:writes+=cache.pop(0)[1]
                            cache.append((line,write))
    writes+=sum(dirty for _,dirty in cache)
    return misses,misses+writes
answers=[simulate(t) for t in (1,4,8)]
assert answers==[(80,144),(32,48),(80,144)]
assert answers[1][1]==3*8*8//4
print(answers)
```

The independent read/write simulator attains the lower bound at tile four. Its eight resident lines can cover the tile's input and output working sets, whereas a full-matrix tile reproduces the naive order. This is a cache-model result; real bandwidth, associativity conflicts and kernel fusion can change the ranking. A valid traffic lower bound is not a measured runtime prediction.

## Worked example and exercise

For N=8, L=4 and C=8, naive order has 80 misses and four-by-four blocking has 32. The arrays occupy 32 distinct lines, so the latter reaches the compulsory-line count. Including writes, modeled byte traffic is 4608 versus 1536. These values describe the specific trace and policy, not a universal device speedup.

Exercise: take N=4, L=1 and C=1 with alternating reads and writes. Find total traffic. Solution: all 32 scalar accesses miss. Each of sixteen output stores produces a dirty line that is written once on eviction or final flush. Total transfers are 48, giving 384 bytes. Omitting the final flush would lose one line transfer.

## Data movement in AI kernels

Attention, tensor transposes and fused kernels depend on reuse, layout and intermediate arrays as well as arithmetic. Arithmetic intensity roughly divides operations by moved bytes and interacts with bandwidth limits. Nevertheless modeled traffic savings are not measured wall time. Real benchmarks must record device, dtype, dimensions, warmup, repetitions and correctness. Use this transparent cache model to explain hypotheses before measuring a more complex hardware hierarchy.

Reference: [MIT cache and performance engineering materials](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
