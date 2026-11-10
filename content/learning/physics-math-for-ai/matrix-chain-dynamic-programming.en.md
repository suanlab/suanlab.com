## Identical matrix order can have different costs

Matrix multiplication is associative in exact mathematics but not commutative. Merely changing parentheses in A₁A₂A₃ can change intermediate dimensions and computational cost. This chapter optimizes scalar multiplication counts for the classical dense product; it does not multiply matrix entries. Floating-point association effects and sparse structure are separate issues.

For dimensions d₀,...,dₙ, matrix A_i has shape d_{i-1} by d_i. Choose uniform, alternating long/thin, or increasing dimension families. Blue shows optimal prefix costs. Orange shows always multiplying from the left.

## Prove optimal substructure

Suppose the final multiplication for interval i through j splits at k. The final product dimensions, and therefore its scalar multiplication count, are fixed. If either subchain were not solved optimally, replacing that subchain would improve the whole solution, contradicting optimality.

$$
M_{i,j}=\min_{i\le k<j}\{M_{i,k}+M_{k+1,j}+d_i d_{k+1}d_{j+1}\},\qquad M_{i,i}=0.
$$

This formula uses zero-based intervals. The last multiplication combines shapes d_i by d_{k+1} and d_{k+1} by d_{j+1}. Adding those costs specifies the computational model; it does not reorder the matrices themselves.

## Divide and conquer versus dynamic programming

A recursive search over every split repeatedly evaluates identical intervals. Memoization stores visited results; tabulation fills intervals in increasing length. The experiment actually executes both bottom-up DP and a nonmemoized recursion. Their reported split evaluations and recursive calls are different counters, not interchangeable elementary operations.

$$
\#\mathrm{DP\ splits}=\frac{n(n^2-1)}6,\qquad
\mathrm{space}=O(n^2),\qquad\mathrm{time}=O(n^3).
$$

The split table also reconstructs a witness parenthesization. As a contrast, ordinary mergesort divides into disjoint halves and performs a linear merge, giving T(n)=2T(n/2)+O(n). Its divide-and-conquer structure differs from reusing overlapping subchain problems here.

## Worked example and exercise

For dimensions 1,2,3,4, multiplying (A₁A₂)A₃ costs 1·2·3+1·3·4=18. Multiplying A₁(A₂A₃) costs 2·3·4+1·2·4=32. The exact mathematical product is identical, but the work differs. Doubling every dimension multiplies each scalar cost by eight.

Exercise: compute both costs for dimensions 5,30,5,30. Solution: the left association costs 5·30·5+5·5·30=1500, while the right costs 30·5·30+5·30·30=9000. Compare those numbers with the n=3 split table. At n=6, the DP evaluates 35 split candidates and the specified recursion makes 243 calls. Dividing those different counters does not directly yield a measured speedup.

## AI computation graphs

Tensor-contraction planning, query optimization and sequence decoding depend on subproblem state and intermediate costs. Verify optimal substructure, sufficient boundary state and the relevance of the chosen cost model. Minimizing scalar multiplication count need not minimize memory traffic or GPU wall time. A correct combinatorial optimum answers the declared cost question, not every systems-performance question.

Reference: [MIT algorithms and dynamic programming materials](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
