## A path does not always mean the same dependence

Compare chain A→B→C, fork A←B→C and collider A→B←C. Conditioning on middle node B blocks chain and fork paths. An unobserved collider blocks its path; conditioning on it or a descendant can open the path. This lab has only three nodes and no added descendant.

In a general DAG, the conditioning set must block every undirected path. A path is blocked by an observed noncollider, or by a collider for which neither it nor any descendant is observed. Blocking just one of several paths does not establish separation.

## Verify with actual tables

Chains and forks use a fair root and independent flip noise ε. Collider roots A,C are independent and fair. B follows noisy OR: probability of B=1 is 1−ε if either parent is one, and ε if both are zero.

At ε=.1, unconditioned A,C masses are all .25. Observing B=1 gives weights (.025,.225,.225,.225), totaling .7. The posterior is (1/28,9/28,9/28,9/28). If one cause is absent, the other becomes more likely to explain the observation.

## Joint versus product of marginals

Blue is the conditional A,C joint and orange the product of conditional marginals. Different bars demonstrate dependence. The readout is mutual information under one selected event, not conditional mutual information averaged over every B value.

$$
p(a,c\mid B=b)\stackrel{?}=p(a\mid B=b)p(c\mid B=b).
$$

## D-separation is a sufficient condition

D-separation guarantees independence for distributions Markov to the graph. The reverse direction needs further conditions such as faithfulness. At ε=.5, interactions ignore their inputs, giving zero information despite an active path. At ε=0, observing B=0 in the collider fixes both parents to zero, another special conditional independence.

Do not identify graphical connectivity with dependence at every parameter value. Causal identification is a separate question too.

## Observing only a descendant can open the path

Extend the three-node example by adding B→D. A,C remain independent fair bits, B uses the same noisy-OR with ε=.1, and D copies B with accuracy q. Observe D=1 without observing B. At q=.8, P(D=1|A,C) is .26 when both parents are zero and .74 otherwise. The posterior table is therefore (13,37,37,37)/124.

For binary A,C, independence is equivalent to the normalized 2×2 table having determinant Δ=p00p11−p01p10 equal to zero. Expanding the marginals and using total mass one gives Δ=p11−P(A=1)P(C=1), the covariance of their indicator variables.

| q | P(D=1) | P(A=1\|D=1) | Δ |
|---|---|---|---|
| 1/2 | 1/2 | 1/2 | 0 |
| 4/5 | 31/50 | 37/62 | −111/1922 |
| 1 | 7/10 | 9/14 | −9/98 |

At q=1/2, D supplies no information about B. The path is graphically active, but this parameter setting removes actual dependence. At q=1 the result equals directly observing B=1. The code below sums the D=1 weights among sixteen (A,C,B,D) assignments using exact fractions. This extension does not add a D control to the three-node screen.

## Preserve small dependence instead of rounding it to zero

Without observing B in the chain or fork, endpoint spin correlation is ρ=(1−2ε)² and both marginals are fair. Mutual information is consequently the following expression, with leading term ρ²/(2 ln 2) for small ρ:

$$
I(A;C)=\frac{(1+\rho)\ln(1+\rho)+(1-\rho)\ln(1-\rho)}{2\ln 2}.
$$

At ε=.5−2⁻²⁰ the positive information is about 9.55×10⁻²⁴ bits. Subtracting three order-one entropies previously lost it to zero. The corrected calculation uses product masses qᵢ and differences δᵢ, summing qᵢ[(1+δᵢ/qᵢ)ln(1+δᵢ/qᵢ)−δᵢ/qᵢ]. The bracket is evaluated as x²/2−x³/6+…; the removed linear terms sum to zero. Near independence in the collider, an algebraic model determinant similarly preserves the differences. This distinguishes weak positive information from exact zero at ε=.5 or under d-separation. It remains finite-precision computation, not an arbitrary-precision certificate.

**Exercise:** May the graph's edges be discarded because dependence vanishes at q=1/2? **Explanation:** That equality holds at a special parameter setting. It does not establish a structural independence throughout the model family.

```python
from fractions import Fraction as F
from itertools import product

epsilon = F(1, 10)
for accuracy in [F(1, 2), F(4, 5), F(1)]:
    joint = {(a, c): F(0) for a, c in product([0, 1], repeat=2)}
    for a, c, b in product([0, 1], repeat=3):
        prob_b1 = 1 - epsilon if a or c else epsilon
        prob_b = prob_b1 if b else 1 - prob_b1
        prob_d1 = accuracy if b else 1 - accuracy
        joint[a, c] += F(1, 4) * prob_b * prob_d1
    evidence = sum(joint.values())
    posterior = {key: w / evidence for key, w in joint.items()}
    pa = sum(v for (a, c), v in posterior.items() if a)
    pc = sum(v for (a, c), v in posterior.items() if c)
    determinant = posterior[0, 0]*posterior[1, 1] - posterior[0, 1]*posterior[1, 0]
    assert posterior[1, 1] - pa*pc == determinant
    assert (determinant == 0) == (accuracy == F(1, 2))
    print('q, evidence, P(A=1|D=1), covariance:', accuracy, evidence, pa, determinant)
```


## Exercises with explanations

1. Does an unobserved collider open its path? **No: it blocks it.**
2. What happens when B is observed in a chain? **The endpoints are conditionally independent.**
3. Is MI after selecting B=1 averaged over B? **No: it conditions on that one event.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
