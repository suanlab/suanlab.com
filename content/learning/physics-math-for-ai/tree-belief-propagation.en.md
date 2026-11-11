## A message summarizes the rest of a subtree

On a tree, message i→j summarizes the i-side subtree as a function of j. Multiply i's unary with messages from its other neighbors, then sum i out. Exclude the recipient's incoming message to avoid sending its information straight back.

$$
m_{i\to j}(x_j)\propto\sum_{x_i}\phi_i(x_i)\psi_{ij}(x_i,x_j)
\prod_{k\in N(i)\setminus j}m_{k\to i}(x_i).
$$

Positive message scaling cancels in final normalization. The implementation normalizes each two-state message for stability.

## Space and rounds in this example

The tree is A−B−C−D, with field h only at A. Initial messages are (.5,.5). A synchronous round computes every new message from the previous round. Information travels one edge per round, requiring three rounds for A's influence to reach D.

## Current beliefs versus exact marginals

Normalize a node's unary times all incoming messages. Blue shows current beliefs; orange enumerates the exact sixteen-state joint. At J=1,h=.7 after one round, C,D still have belief .5 while exact probabilities are about .6753 and .6335.

$$
b_i(x_i)\propto\phi_i(x_i)\prod_{k\in N(i)}m_{k\to i}(x_i).
$$

Three rounds communicate the entire tree, producing agreement within floating-point precision. A directed tree schedule can instead calculate each directional message once. Synchronous round count and scheduled message-operation count are different measures.

## Conditions and objectives

Exactness relies on tree structure and exact sum-product operations. The same repeated update on a cycle has different guarantees. Choosing each node's marginal mode also need not produce a joint MAP assignment; distinguish max-product and appropriate backtracking.

## Induction over subtree sums

Delete edge i−j and call the component containing i Tᵢ→ⱼ. An unnormalized message sums all weights inside that component while leaving only the boundary value xⱼ. If i is a leaf, only xᵢ must be summed, giving the message formula directly. At an internal node, the components from neighbors k≠j are disjoint. Their already-computed messages can therefore multiply, reusing each subtree sum once. Summing xᵢ completes the induction. Combining all incoming directions at a node gives its exact marginal.

On a cycle, those components no longer separate in the same way: information can return along another route. Excluding the recipient's immediate incoming message does not remove this global duplication.

## A closed form shows when information arrives

The current chain has weight exp(hs₀+JΣsᵢsᵢ₊₁), spins sᵢ∈{−1,1}, and a field only at its first node. Successively summing the end spin gives a factor 2cosh J, hence Z=2cosh h(2cosh J)³. The conditional mean is E[sᵢ₊₁|sᵢ]=sᵢ tanh J and the initial mean is tanh h. Repeated expectation gives

$$
E[s_i]=\tanh(h)\,\tanh(J)^i,
\qquad P(X_i=1)=\frac{1+\tanh(h)\tanh(J)^i}{2}.
$$

Starting from uniform messages, r synchronous rounds bring the first field's information to nodes i≤r; more distant spin means remain zero. This simple description uses the special case of one field at the chain's endpoint. General trees require checking distances from all sources and leaves. For finite positive factors, exact arithmetic and this initialization, the tree diameter bounds the rounds needed for complete information propagation.

| Round | Nodes reached by field information | Maximum marginal error, J=1,h=.7 |
|---|---|---|
| 0 | A | .230141 |
| 1 | A,B | .175274 |
| 2 | A,B,C | .133488 |
| 3 | A,B,C,D | 0 |

The code compares independent enumeration, the closed form and message updates, including negative J and h. The final zero is the exact-arithmetic result; executable assertions use a floating-point tolerance.

**Exercise:** What changes for J<0? **Explanation:** The negative tanh J alternates the mean's sign with distance. For finite J its magnitude decays as a power of |tanh J|.

```python
from math import exp, tanh, cosh
from itertools import product

edges = [(0,1),(1,2),(2,3)]
neighbors = [[j for u,v in edges for j in ([v] if u==i else [u] if v==i else [])]
             for i in range(4)]
def normalize(v):
    return [x/sum(v) for x in v]

for J, h in [(1,.7), (-1,.7), (.4,-.7)]:
    joint = [(s, exp(h*s[0] + J*sum(s[i]*s[j] for i,j in edges)))
             for s in product([-1,1], repeat=4)]
    Z = sum(w for s,w in joint)
    exact = [sum(w*s[i] for s,w in joint)/Z for i in range(4)]
    analytic = [tanh(h)*tanh(J)**i for i in range(4)]
    assert max(abs(a-b) for a,b in zip(exact,analytic)) < 1e-12
    assert abs(Z/(2*cosh(h)*(2*cosh(J))**3)-1) < 1e-12
    messages = {(i,j): [.5,.5] for u,v in edges for i,j in [(u,v),(v,u)]}
    for rounds in range(4):
        means = []
        for i in range(4):
            v = [exp(h*x if i==0 else 0) for x in [-1,1]]
            for k in neighbors[i]:
                v = [v[t]*messages[k,i][t] for t in range(2)]
            means.append(2*normalize(v)[1]-1)
        expected = [analytic[i] if i<=rounds else 0 for i in range(4)]
        assert max(abs(a-b) for a,b in zip(means,expected)) < 1e-12
        if J==1 and h==.7:
            print('round, max marginal error:', rounds,
                  round(max(abs(a-b)/2 for a,b in zip(means,exact)),6))
        updated = {}
        for i,j in messages:
            v = []
            for y in [-1,1]:
                total = 0
                for t,x in enumerate([-1,1]):
                    weight = exp(J*x*y + (h*x if i==0 else 0))
                    for k in neighbors[i]:
                        if k!=j:
                            weight *= messages[k,i][t]
                    total += weight
                v.append(total)
            updated[i,j] = normalize(v)
        messages = updated
```


## Exercises with explanations

1. Is j's incoming message used when sending back to j? **No.**
2. Does a small local error after one round certify all information has propagated? **No: inspect paths from distant observations.**
3. What happens at J=0? **Variables are independent and initial local beliefs are already exact.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
