## 1. Count arithmetic and movement at a named boundary

A compute kernel executes operations and moves data. Choose a boundary such as the interface between an accelerator and external memory. Count $F$ floating-point operations and $D$ bytes that cross that boundary. Operational intensity is $I=F/D$ in FLOP/byte. Arithmetic count can stay constant while tiling or caching reduces traffic. Always state the boundary and counting convention, including whether fused multiply–add counts as two FLOPs. [Williams, Waterman and Patterson, Roofline technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)

## 2. Derive the two throughput ceilings

Let peak compute throughput be $P$ FLOP/s and memory bandwidth $B$ byte/s. Compute time is at least $F/P$ and memory transfer time is at least $D/B$. With perfect overlap the optimistic runtime is their maximum. The corresponding performance ceiling is

$$
P_{\rm roof}(I)=\min(P,BI).
$$

The two ceilings cross at $I_*=P/B$. On a logarithmic plot, the bandwidth ceiling rises with slope one and the compute ceiling is flat. A model using sustained benchmark rates is a practical envelope for comparable conditions. A ceiling based on documented absolute rates gives an optimistic bound under the stated workload and boundary assumptions.

## 3. Work through a memory-reuse example

Use a hypothetical accelerator with $P=8$ TFLOP/s and $B=200$ GB/s, using decimal prefixes. The ridge is 40 FLOP/byte. A kernel with $F=10^9$ FLOPs and $D=100$ MB has intensity 10 FLOP/byte and a 2 TFLOP/s roof. Its compute time is 0.125 ms and transfer time is 0.5 ms, giving a 0.5 ms lower bound in this model.

Reducing traffic to 20 MB raises intensity to 50 FLOP/byte. The transfer term becomes 0.1 ms and the runtime floor becomes 0.125 ms. Both versions execute the same arithmetic. The predicted bottleneck changes because more operations reuse each byte that crosses the chosen boundary.

## 4. Keep latency and bandwidth distinct

Bandwidth describes bytes moved per second once transfers are in flight. Latency describes how long a particular request takes to complete. A chain of dependent reads may expose a large fraction of latency even with high peak bandwidth. Kernel launch, synchronization, cache misses, and partial overlap can also extend observed runtime. If a measured launch cost of 0.02 ms occurs serially before all kernel work, add it to the modeled floor. If it overlaps another stage, describe that schedule before combining times.

## 5. Reproduce the two ceilings in Python

```python
from math import isclose

peak, bandwidth, operations = 8e12, 200e9, 1e9

for moved in [100e6, 25e6, 20e6]:
    intensity = operations / moved
    compute_time = operations / peak
    transfer_time = moved / bandwidth
    lower_bound = max(compute_time, transfer_time)
    roof = min(peak, bandwidth * intensity)
    assert isclose(operations / lower_bound, roof)
    print(f"{intensity:5.1f} FLOP/B, {roof/1e12:4.1f} TFLOP/s, {lower_bound*1e3:.3f} ms")

assert isclose(peak / bandwidth, 40)
```

## 6. Design a reproducible performance comparison

Measure elapsed time and external bytes for each implementation of the same computation. Record input shape, numeric precision, batch size, warm-up, and whether compilation or transfers are included. Hardware counters, a sustained-bandwidth microbenchmark, and a separate launch measurement let you test different explanations for time spent below the roof. A change in latency may require a different experiment from a change in the volume of traffic.
