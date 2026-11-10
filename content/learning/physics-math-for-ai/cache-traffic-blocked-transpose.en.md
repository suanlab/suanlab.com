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

## Worked example and exercise

For N=8, L=4 and C=8, naive order has 80 misses and four-by-four blocking has 32. The arrays occupy 32 distinct lines, so the latter reaches the compulsory-line count. Including writes, modeled byte traffic is 4608 versus 1536. These values describe the specific trace and policy, not a universal device speedup.

Exercise: take N=4, L=1 and C=1 with alternating reads and writes. Find total traffic. Solution: all 32 scalar accesses miss. Each of sixteen output stores produces a dirty line that is written once on eviction or final flush. Total transfers are 48, giving 384 bytes. Omitting the final flush would lose one line transfer.

## Data movement in AI kernels

Attention, tensor transposes and fused kernels depend on reuse, layout and intermediate arrays as well as arithmetic. Arithmetic intensity roughly divides operations by moved bytes and interacts with bandwidth limits. Nevertheless modeled traffic savings are not measured wall time. Real benchmarks must record device, dtype, dimensions, warmup, repetitions and correctness. Use this transparent cache model to explain hypotheses before measuring a more complex hardware hierarchy.

Reference: [MIT cache and performance engineering materials](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
