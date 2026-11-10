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

## Message accounting and worked example

A synchronous round sends one scalar in each direction on every active edge. A pair step sends one scalar each way only on its chosen edge. Each payload is an eight-byte double; headers, synchronization and latency are excluded. Equal step counts therefore need not have equal communication cost.

In the complete network, degree five gives every weight 1/6. Initial values [0,1,2,3,4,5] all become 2.5 after one round. Fifteen edges require thirty directed messages, or 240 payload bytes. Two disconnected triangles instead approach [1,1,1,4,4,4]. Their mean remains 2.5 while global spread remains three.

Exercise: count messages for thirty synchronous ring rounds versus thirty pair steps. Solution: six ring edges give 30·12=360 directed messages in the first case, versus 30·2=60 in the second. Comparing their final spreads as if the communication budgets were identical would be misleading.

## Distributed AI reporting

For distributed training and decentralized learning, record local work, topology, payload size, round definition and failure assumptions. This finite model is scalar averaging on a fixed undirected network, not a complete Byzantine or arbitrary-asynchronous consensus protocol. Check conservation, connectivity, convergence conditions, finite observed error and communication budgets separately before making broader scalability claims.

Reference: [MIT parallel and distributed performance engineering](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
