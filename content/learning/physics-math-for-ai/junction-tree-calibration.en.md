## Convert a cycle into a tree of clusters

Add diagonal AC to cycle A−B−C−D−A, forming clusters ABC and ACD with separator AC. The original graph has a cycle, but the cluster connection is a tree.

Every original factor must fit within a cluster. Clusters containing the same variable must form a connected subtree: the running-intersection property. An arbitrary tree of arbitrary clusters does not suffice.

## Assign every factor once

Use φ₁(A,B,C)=φAψABψBCe(C) and φ₂(A,C,D)=ψCDψDA. Shared variables do not justify duplicating unary or evidence factors. Doing so changes the model.

## Exchange two messages

Sum out nonshared variables: μ₁₂(A,C)=ΣBφ₁ and μ₂₁(A,C)=ΣDφ₂. Summing their product gives the model/evidence normalizer Z.

$$
b_{ABC}=\phi_1\mu_{21}/Z,\qquad b_{ACD}=\phi_2\mu_{12}/Z,
\quad b_{AC}=\mu_{12}\mu_{21}/Z.
$$

Marginalizing either cluster belief to AC must give the same separator belief. Blue and orange compare those routes. A tiny numerical residual reflects floating-point arithmetic, not two logically different required probabilities.

## Evidence and table interpretation

Observing C=1 gives zero separator mass to rows with C=0. Raw messages need not sum to one; the final two table columns are normalized marginals. With zero field and coupling, all sixteen assignments have weight one, Z=16 and each separator mass is .25.

## Exactness and computation

Valid cluster factorization, running intersection and complete tree message exchange permit exact inference. Large clusters still produce exponentially large tables. Here cluster tables have eight entries and separator tables four. Large treewidth can make exact junction-tree inference expensive.

## Where running intersection enters the proof

Cut one edge of the cluster tree into two subtrees. Under running intersection, every variable appearing on both sides must belong to that edge's separator. Otherwise the path between its two occurrences would pass through a cluster missing the variable, violating the condition.

Fixing the separator therefore lets the two sides sum their remaining variables independently. Define a message as the product of a subtree's factors summed over its internal variables. Induction from the leaves yields the general update:

$$
m_{i\to j}(x_{S_{ij}})=
\sum_{x_{C_i\setminus S_{ij}}}\Phi_i(x_{C_i})
\prod_{k\in N(i)\setminus\{j\}}m_{k\to i}(x_{S_{ki}}).
$$

Assign each original factor to exactly one Φᵢ. Complete message exchange then makes bᵢ∝Φᵢ∏m an exact cluster marginal. The separator must retain every shared variable needed by the two subproblems.

## A small separator residual does not certify the structure

Consider a separate three-variable model w(a,b,c)=1{a=b}1{a=c}. Only 000 and 111 are possible, so Z=2. Connect clusters as AB−BC−AC and give the middle BC cluster the constant factor one. A appears at both endpoints but not in the middle, violating running intersection. The invalid tree computation treats A as separate left and right variables:

$$
Z_{\mathrm{bad}}=\sum_{a_L,b,c,a_R}1\{a_L=b\}1\{a_R=c\}=4\ne2.
$$

Both leaf messages are constant one. Both B and C separator marginals calibrate to (1/2,1/2), yet the original Z is wrong. The invalid middle BC belief is uniform, whereas the intended B,C always agree. Expanding the middle cluster gives AB−ABC−AC and restores the shared A, as the diagram shows. Each original factor still occurs only once.

![An invalid AB–BC–AC path disconnects the occurrences of A; enlarging the middle cluster to ABC restores running intersection. Calibrated separators alone can accompany a wrong normalizer of four.](/assets/learning/graphical-running-intersection-en.svg)

| Construction | Shared-variable connection | Z |
|---|---|---|
| Original joint | One shared A | 2 |
| AB−BC−AC | A missing in middle | 4 |
| AB−ABC−AC | A along entire path | 2 |

The fraction code verifies this counterexample and its separator calibration. It permits zero factors and is separate from the screen's positive four-cycle Ising model. For a positive separator, the calibrated two-clique joint reconstructs as bABC·bACD/bAC. When bAC=0, do not evaluate 0/0; the joint has zero mass outside the separator's support.

**Exercise:** Can more message iterations restore the missing A? **Explanation:** No. The cluster structure must first retain the shared variable.

```python
from fractions import Fraction as F
from itertools import product

def equal(x, y):
    return F(x == y)

# Intended three-variable law; its normalizer is two.
z = sum(equal(a,b)*equal(a,c)
        for a,b,c in product([0,1], repeat=3))
# Invalid AB--BC--AC path forgets A between the endpoint clusters.
# Its tree sum has separate copies a_left and a_right.
z_bad = sum(equal(al,b)*equal(ar,c)
            for al,b,c,ar in product([0,1], repeat=4))
assert z == 2 and z_bad == 4
# Messages from either leaf are the constant function one.
m_left = [sum(equal(a,b) for a in [0,1]) for b in [0,1]]
m_right = [sum(equal(a,c) for a in [0,1]) for c in [0,1]]
assert m_left == m_right == [1,1]
# Both separator marginals calibrate to (1/2,1/2) despite wrong Z.
separator_b = [sum(m_left[b]*m_right[c]
                   for c in [0,1])/z_bad for b in [0,1]]
separator_c = [sum(m_left[b]*m_right[c]
                   for b in [0,1])/z_bad for c in [0,1]]
assert separator_b == separator_c == [F(1,2),F(1,2)]
print('intended Z, invalid cluster-tree Z:', z, z_bad)
print('both calibrated separators:', separator_b)
```


## Exercises with explanations

1. Should shared variables make a factor appear twice? **No: assign each original factor once.**
2. Must messages be probability distributions? **No: they are separator functions.**
3. How should the separator marginals compare? **They agree within numerical precision.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
