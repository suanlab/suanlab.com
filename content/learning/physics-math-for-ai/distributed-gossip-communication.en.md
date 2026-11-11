## Preserving a sum differs from agreeing

Six nodes hold initial values [x₀,1,2,3,4,5]. Computing their mean through neighbor communication requires an explicit graph, update rule and message accounting. A preserved global sum does not force disconnected components to agree. The experiment compares ring, path, two-component, complete and empty networks, with optional static removal of the first edge.

The upper graph shows current values and remaining edges. Blue is global max-minus-min spread. Orange is drift from the initial global mean. Zero orange drift does not imply zero blue spread. Componentwise limits in the table describe the stated fixed-network asymptotics, not an assertion that a finite trajectory has already reached them.

## Synchronous Metropolis averaging

For each active edge, define weights and diagonal entries by

$$
W_{ij}=\frac1{1+\max(d_i,d_j)},\qquad
W_{ii}=1-\sum_{j\ne i}W_{ij},\qquad x^{k+1}=Wx^k.
$$

Symmetry makes the row-stochastic matrix column-stochastic as well, preserving the sum. Positive diagonal weights avoid periodicity. On a fixed connected graph, repeated averaging converges to the initial global mean. Disconnected blocks instead preserve their own sums and converge to component means. This conclusion does not cover arbitrary time-varying failures or malicious messages.

## Actual pair exchanges

The second mode cyclically selects one active edge and averages only its two endpoints. Both old values must be read before either update:

$$
x_i^{new}=x_j^{new}=\tfrac12(x_i^{old}+x_j^{old}),\qquad
V_{new}=V_{old}-\tfrac12(x_i-x_j)^2.
$$

V is squared disagreement around the fixed mean. Each pair preserves the sum and decreases disagreement. Convergence on a connected fixed graph requires fair repeated edge selection; the stated cyclic rule visits every active edge. A short finite run need not have reached exact consensus.

## Spectral contraction requires the right invariant average

For a fixed symmetric stochastic W, decompose x into its mean component Jx and disagreement e=(I−J)x, J=11ᵀ/N. On a connected graph with positive diagonal, the only eigenvalue one is the constant mode and all other eigenvalues have absolute value below one. If ρ is their largest absolute value,

$$
\|x^k-Jx^0\|_2\le\rho^k\|x^0-Jx^0\|_2.
$$

This is an error bound per communication round. A ring and complete network spend different message counts per round. The six-node ring has eigenvalues 1/3+(2/3)cos(2πj/6), so ρ=2/3. Two disconnected triangles retain a component-disagreement eigenvalue one even though each component reaches its own average.

![Exact rational averaging disagreement versus directed message count for three fixed six-node topologies.](/assets/learning/algorithm-gossip-en.svg)

| Topology | Global disagreement factor ρ | Directed messages per round |
|---|---:|---:|
| Ring | 2/3 | 12 |
| Complete | 0 | 30 |
| Two triangles | 1 | 12 |

```python
from fractions import Fraction as F
initial=list(map(F,range(6)));mean=sum(initial)/6
ring=[[F(int(i==j or (i-j)%6 in (1,5)),3) for j in range(6)] for i in range(6)]
x=initial[:];energy=lambda v:sum((a-mean)**2 for a in v)
start=energy(x)
for k in range(1,11):
    x=[sum(a*b for a,b in zip(row,x)) for row in ring]
    assert sum(x)==sum(initial) and energy(x)<=F(4,9)**k*start
print('ring after 120 messages:',float(energy(x)))
W=[[F(9,10),F(1,10)],[F(2,5),F(3,5)]]
z=[F(0),F(1)];weights=[F(4,5),F(1,5)]
for _ in range(20):
    z=[sum(a*b for a,b in zip(row,z)) for row in W]
    assert sum(a*b for a,b in zip(weights,z))==F(1,5)
print('directed values:',list(map(float,z)))
```

Row stochasticity alone preserves constant vectors, not the arithmetic mean. The directed two-state matrix [[0.9,0.1],[0.4,0.6]] has stationary left weights (0.8,0.2); from (0,1), consensus tends to 0.2, not 0.5. The auxiliary rational example checks that weighted invariant. The live network remains undirected. Time-varying graphs, asynchronous delivery and Byzantine behavior require different hypotheses and cannot be certified by this fixed symmetric spectrum.

## Message accounting and worked example

A synchronous round sends one scalar in each direction on every active edge. A pair step sends one scalar each way only on its chosen edge. Each payload is an eight-byte double; headers, synchronization and latency are excluded. Equal step counts therefore need not have equal communication cost.

In the complete network, degree five gives every weight 1/6. Initial values [0,1,2,3,4,5] all become 2.5 after one round. Fifteen edges require thirty directed messages, or 240 payload bytes. Two disconnected triangles instead approach [1,1,1,4,4,4]. Their mean remains 2.5 while global spread remains three.

Exercise: count messages for thirty synchronous ring rounds versus thirty pair steps. Solution: six ring edges give 30·12=360 directed messages in the first case, versus 30·2=60 in the second. Comparing their final spreads as if the communication budgets were identical would be misleading.

## Distributed AI reporting

For distributed training and decentralized learning, record local work, topology, payload size, round definition and failure assumptions. This finite model is scalar averaging on a fixed undirected network, not a complete Byzantine or arbitrary-asynchronous consensus protocol. Check conservation, connectivity, convergence conditions, finite observed error and communication budgets separately before making broader scalability claims.

Reference: [MIT parallel and distributed performance engineering](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
