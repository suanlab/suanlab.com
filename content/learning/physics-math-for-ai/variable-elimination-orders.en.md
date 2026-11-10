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

## Experiment and exercises

Removing evidence returns A to .5. Include the observation factor once and normalize the final two A weights by their sum. Compare small-table calculations with independent enumeration.

1. Does elimination order change the model? **No: it changes computational cost.**
2. Must every intermediate sum equal one? **No: intermediate factors can be unnormalized.**
3. Is finding an optimal order always easy? **No: treewidth and computational complexity matter.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/ve/).
