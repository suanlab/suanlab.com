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

## A memory constraint changes the sufficient DP state

Scalar work alone need not select a feasible contraction order. Consider dimensions (1,10,100,5). Exclude storage of the original inputs, count every newly allocated intermediate and final output, and assume kernel scratch is zero. Both children remain live while their parent output is allocated, and child intermediates are freed afterward.

For left-first evaluation with child peaks pL,pR, retained intermediate sizes sL,sR, and output size s, the peak is max(pL,sL+pR,sL+sR+s). Right-first exchanges L and R. Leaves use zero additional storage in this model. Combining costs and taking the cheaper evaluation order gives two nondominated plans: left association has work 1500 and peak 105 scalars; right association has work 5050 and peak 55.

| Temporary/output capacity | Feasible minimum-work plan | Work |
|---|---|---:|
| 50 | none | unavailable |
| 60 | A₁(A₂A₃) | 5050 |
| 110 | (A₁A₂)A₃ | 1500 |

```python
d=(1,10,100,5)
def plans(i,j):
    if i==j:return [(0,0,0,f'A{i+1}')]
    out=[]
    for k in range(i,j):
        for wl,pl,sl,tl in plans(i,k):
            for wr,pr,sr,tr in plans(k+1,j):
                size=d[i]*d[j+1]
                work=wl+wr+d[i]*d[k+1]*d[j+1]
                left=max(pl,sl+pr,sl+sr+size)
                right=max(pr,sr+pl,sl+sr+size)
                out.append((work,min(left,right),size,f'({tl}{tr})'))
    return out
candidates=plans(0,2)
assert {(w,p) for w,p,_,_ in candidates}=={(1500,105),(5050,55)}
for capacity in (50,60,110):
    feasible=[p for p in candidates if p[1]<=capacity]
    best=min(feasible) if feasible else None
    print(capacity,best)
```

An augmented planner can retain a Pareto frontier of (work,peak) pairs for each interval. Discarding a state is justified only when another state has no greater cost in either coordinate and preserves the same boundary shape. Keeping only the lowest-work subplan can remove the sole plan satisfying a later memory limit. Frontier size and enumeration cost require their own analysis; the original scalar O(n³) bound does not automatically survive this extension.

The code is an auxiliary exhaustive planner with custom dimensions. It does not change the live dimension families or estimate GPU allocator behavior, input residency or multiplication workspace.

## Worked example and exercise

For dimensions 1,2,3,4, multiplying (A₁A₂)A₃ costs 1·2·3+1·3·4=18. Multiplying A₁(A₂A₃) costs 2·3·4+1·2·4=32. The exact mathematical product is identical, but the work differs. Doubling every dimension multiplies each scalar cost by eight.

Exercise: compute both costs for dimensions 5,30,5,30. Solution: the left association costs 5·30·5+5·5·30=1500, while the right costs 30·5·30+5·30·30=9000. Compare those numbers with the n=3 split table. At n=6, the DP evaluates 35 split candidates and the specified recursion makes 243 calls. Dividing those different counters does not directly yield a measured speedup.

## AI computation graphs

Tensor-contraction planning, query optimization and sequence decoding depend on subproblem state and intermediate costs. Verify optimal substructure, sufficient boundary state and the relevance of the chosen cost model. Minimizing scalar multiplication count need not minimize memory traffic or GPU wall time. A correct combinatorial optimum answers the declared cost question, not every systems-performance question.

Reference: [MIT algorithms and dynamic programming materials](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
