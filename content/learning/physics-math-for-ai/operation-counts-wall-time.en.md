## Cost units on a graph versus milliseconds on a clock

This chapter compares linear and binary search on the same array and target in two separate ways. The graph applies user-declared unit costs and fixed overhead to actual probe counts. The button below executes actual code in the current browser and measures elapsed time. They are not presented as the same data.

With linear probe cost α, binary probe cost γ and fixed binary overhead β,

$$
C_L(N)=\alpha q_L(N),\qquad C_B(N)=\beta+\gamma q_B(N).
$$

The q values are actual search probes. Cost units have not been calibrated to nanoseconds or milliseconds. Changing α, γ or β changes the modeled curves, not the speed of the search implementation. The repetitions setting separately changes executions per measured batch.

## Compare the same task

Both methods receive the same sorted array and present target selected by relative rank. Array construction is outside the timer. Verify equal returned indices and state whether preprocessing is included. Searching a pre-sorted array is different from the end-to-end task of constructing an index from unsorted data.

Asymptotic bounds describe an input family for sufficiently large N. Finite crossover points, caches, branches, interpreter behavior, JIT compilation and allocation costs are separate effects. A query always requesting the first element gives linear search one probe, so algorithm names alone cannot determine every finite ranking.

## Actual benchmark procedure

The button performs twenty warmup executions per method, then seven timed batches per method, each containing the selected repetitions. Trial order alternates, and the browser regains control between trials. The timed interval includes repeated search calls and checksum accumulation, including comparison counters and returned-object allocation. Array setup remains outside it.

$$
\widehat t=\operatorname{median}(t_1,\ldots,t_7)\quad\text{milliseconds per batch}.
$$

The checksum consumes the results. Repeat the same configuration to observe variability. An interval below clock resolution can appear as zero; it is not replaced with an invented positive value. Other tabs, system load, browser version and JIT state affect measurements. No result is adjusted to force one method to win.

## Worked example and exercise

For the last key in N=1024, linear search uses 1024 probes and binary uses eleven. With α=γ=1 and β=0, modeled costs are 1024 and eleven. With γ=500 and β=1000, binary cost becomes 6500, reversing the modeled ranking. That reversal is a result of declared weights, not a measured slowdown.

Exercise: use N=4, last key, α=1, γ=5 and β=50. Solution: q_L=4 and q_B=3, so costs are four and 65. Change the weights and rerun the actual benchmark. Do not interpret weight changes as a causal implementation change to the same executed search function.

## Performance claims in AI research

Record device, runtime, version, dtype, input sizes, warmup, repetitions, timed boundaries and correctness. Asynchronous devices such as GPUs also require appropriate synchronization. This button measures synchronous JavaScript CPU execution and does not replace GPU timing procedures. Present a general complexity argument, declared cost model and finite-environment measurements together while preserving their different meanings.

Reference: [MIT performance engineering course](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
