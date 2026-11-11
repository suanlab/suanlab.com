## Push marginalization inside the product

To obtain A's marginal in chain A−B−C−D, sum the other variables. Enumerating all sixteen assignments works, but elimination multiplies only factors containing the selected variable, sums it out and reuses the resulting smaller table.

$$
\tau(C)=\sum_D\psi_{CD}(C,D)e(D),
$$

Evidence e(D) is a zero-one factor. The intermediate τ is an unnormalized function, not necessarily a probability distribution.

## Same answer, different intermediate size

Let A be fair and edge agreement r=.8. Given D=1, three-edge correlation is (2r−1)³=.216, so P(A=1|D=1)=.608. Evidence probability is .5.

Order D,C,B creates only two-variable products, peaking at four entries. Starting at B multiplies ψABψBC over A,B,C, creating eight entries. Order B,C,D gives the same answer when calculated exactly.

## Scope determines cost

Removing a variable does not ensure every intermediate function shrinks. Remaining neighbors can become connected through fill-in. A binary k-variable product has 2ᵏ entries, so peak scope strongly affects time and memory. The displayed cost is the actual largest multiplied table, not a full accounting of every arithmetic operation.

## Summation differs from maximization

This task uses sum-product for marginals. Joint MAP needs maximization and often backtracking. Mixed marginal-MAP tasks generally do not allow sums and maxima to be reordered freely.

## The elimination invariant

Maintain that the product of current factors equals the original unnormalized joint summed over all variables already removed. For the next variable v, collect its factors, multiply and sum v. All other factors are independent of v, so the distributive law preserves the invariant. Induction yields the exact joint weight of the remaining query and evidence. Normalize only after all non-query variables have been removed and only when the final mass is positive. Exact real arithmetic gives order-independent answers; finite precision can still introduce order-dependent rounding, overflow or underflow.

## Exchanging max and sum changes marginal MAP

In this positive joint table, rows index A and columns index B. The marginal mode is A=1, but the joint MAP assignment is (0,0).

| A | p(A,B=0) | p(A,B=1) | p(A) |
|---|---|---|---|
| 0 | .40 | .01 | .41 |
| 1 | .30 | .29 | .59 |

$$
\max_a\sum_b p(a,b)=.59
\quad<\quad
\sum_b\max_a p(a,b)=.69.
$$

The right side chooses A=0 when B=0 and A=1 when B=1. The left side must choose one A without knowing B. Allowing the decision to depend on B silently enlarges the optimization problem. For every nonnegative table, p(a,b)≤max_a' p(a',b); summing and then maximizing proves the left side is at most the right side. A shared maximizing A for every B can give equality, but it need not exist.

Sums can exchange with other finite sums. A mixed sum/max reordering can change the objective. The exact-fraction code separately checks the marginal mode, joint MAP and both operation orders.

**Exercise:** Does the joint MAP value A=0 imply that A=0 has the largest marginal mass? **Explanation:** No. The two states with A=1 together have mass .59.

```python
from fractions import Fraction as F

# Rows: A=0,1; columns: B=0,1. All four masses are positive.
p = [[F(40, 100), F(1, 100)], [F(30, 100), F(29, 100)]]
assert sum(map(sum, p)) == 1
marginal = [sum(row) for row in p]
joint_map = max(((p[a][b], a, b) for a in range(2) for b in range(2)))
sum_of_max = sum(max(p[a][b] for a in range(2)) for b in range(2))
max_of_sum = max(marginal)
assert joint_map[1:] == (0, 0)
assert marginal.index(max_of_sum) == 1
assert sum_of_max == F(69, 100) > max_of_sum == F(59, 100)
print('A marginals:', marginal)
print('joint MAP (A,B):', joint_map[1:])
print('sum_B max_A, max_A sum_B:', sum_of_max, max_of_sum)
```


## Experiment and exercises

Removing evidence returns A to .5. Include the observation factor once and normalize the final two A weights by their sum. Compare small-table calculations with independent enumeration.

1. Does elimination order change the model? **No: it changes computational cost.**
2. Must every intermediate sum equal one? **No: intermediate factors can be unnormalized.**
3. Is finding an optimal order always easy? **No: treewidth and computational complexity matter.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/ve/).
