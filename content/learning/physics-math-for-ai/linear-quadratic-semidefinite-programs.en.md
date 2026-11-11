## Three optimization languages

Linear programs use affine objectives and constraints. Convex quadratic programs add a positive semidefinite quadratic objective. Semidefinite programs constrain an affine symmetric matrix to be positive semidefinite. These descriptions specify structure that algorithms and certificates can exploit; they do not imply that every quadratic objective is convex.

The lab solves three small families exactly. It is useful to compare their geometry before considering a general solver. Blue marks the feasible boundary, orange marks an optimal objective line or the projection segment. The query point is independent of the optimizer and can be infeasible.

## Linear objectives on a triangle

Let T contain nonnegative pairs with x₁+x₂≤1. Every point is a convex combination of its three vertices, so the minimum of c·x is min(0,c₁,c₂). A vertex attains the minimum, but the optimum need not be unique. With c=(−1,−1), every point on the edge x₁+x₂=1 is optimal.

Introduce λ≥0 for the sum constraint and μ≥0 for nonnegativity. The dual equations are

$$
c+\lambda(1,1)-\mu=0,\qquad d=-\lambda.
$$

Choosing λ=max(0,−c₁,−c₂) and μᵢ=cᵢ+λ gives a lower bound equal to the best vertex value. This is a certificate, rather than evidence based only on a dense plot.

## Quadratic projection

The second family minimizes half the squared Euclidean distance from c to T. Strict convexity gives a unique projection. KKT conditions give xᵢ=max(cᵢ−λ,0), where λ is zero if the positive parts already sum to at most one; otherwise their sum is exactly one.

For c=(2,0.5), the projection is (1,0), λ=1 and μ=(0,0.5). The objective is 0.625. Writing r=λ(1,1)−μ, the dual value c·r−‖r‖²/2−λ is also 0.625. A nearby feasible query can have a larger objective while still satisfying every constraint.

## A genuine matrix inequality

The SDP family uses two variables a,b:

$$
\min c_1a+c_2b,\qquad
F(a,b)=\begin{pmatrix}1&a&b\\a&1&0\\b&0&1\end{pmatrix}\succeq0.
$$

The Schur complement of the lower identity block gives a²+b²≤1. Equivalently, the eigenvalues are 1 and 1±√(a²+b²). Thus a circular picture represents an actual matrix constraint, rather than relabeling a scalar quadratic program.

For nonzero c the optimum is −c/‖c‖ with value −‖c‖. Let u=c/‖c‖ and v=(1,u₁,u₂). The matrix Z=‖c‖vvᵀ/2 is positive semidefinite. Its off-diagonal trace constraints match c, and the dual value −tr Z equals −‖c‖. At the optimum Fv=0, proving complementary slackness. Negative individual entries do not prevent a matrix from being positive semidefinite.

## Derive a conic dual and its complementarity certificate

For F(x)=F₀+ΣxᵢFᵢ⪰0, introduce Z⪰0 and use L=cᵀx−⟨Z,F(x)⟩. The unconstrained infimum is finite only if ⟨Z,Fᵢ⟩=cᵢ for every i; its value is then −⟨Z,F₀⟩. The primal-dual gap of a feasible pair is

$$
c^Tx+\langle Z,F_0\rangle=\langle Z,F(x)\rangle\ge0.
$$

The final inequality follows because F¹ᐟ²ZF¹ᐟ² is positive semidefinite and has nonnegative trace. It is not an entrywise positivity argument. If the trace is zero, this PSD matrix is zero; factoring its squared Frobenius norm gives Z¹ᐟ²F¹ᐟ²=0 and hence FZ=0. This justifies the matrix complementary-slackness test used in the example.

The exact code constructs the rational certificate for c=(6/5,8/5). Its trace is two. It then compares feasible and infeasible query points. A negative algebraic gap at an infeasible matrix does not contradict weak duality.

| Query (a,b) | Objective minus dual value | Certificate status |
|---|---:|---|
| (0,0) | 2 | Feasible upper bound |
| (-3/5,-4/5) | 0 | Optimal |
| (-1,-1) | -4/5 | Infeasible; no certificate |

The same discipline applies to LP and QP multipliers: enforce the dual domain and stationarity equations before treating a printed number as a lower bound. General solvers require residual and scaling analysis beyond this exact family.

## Reproduce the calculation in Python

```python
from fractions import Fraction as Q
c=[Q(6,5),Q(8,5)]; v=[Q(1),Q(3,5),Q(4,5)]
Z=[[x*y for y in v] for x in v]
assert sum(Z[i][i] for i in range(3))==2
assert [2*Z[0][1],2*Z[0][2]]==c
for a,b in [(Q(0),Q(0)),(-Q(3,5),-Q(4,5)),(-Q(1),-Q(1))]:
    F=[[Q(1),a,b],[a,Q(1),Q(0)],[b,Q(0),Q(1)]]
    inner=sum(F[i][j]*Z[i][j] for i in range(3) for j in range(3))
    assert inner == c[0]*a+c[1]*b+2
    print((a,b),'gap:',inner,'feasible:',a*a+b*b<=1)
    if (a,b)==(-Q(3,5),-Q(4,5)):
        assert all(sum(F[i][k]*Z[k][j] for k in range(3))==0 for i in range(3) for j in range(3))
```

## Exercise and solver limitations

Use c=(1.2,1.6): the SDP optimum is (−0.6,−0.8), value −2. Construct Z=vvᵀ and verify its trace is two. Then put the query at (1,1): its minimum eigenvalue is 1−√2, so its objective cannot be used as a feasible upper bound.

Set both costs to zero and compare uniqueness in the LP, QP and SDP modes. The QP remains uniquely minimized at the origin; the linear families have many optimizers. These closed forms solve the displayed families only. General SDP algorithms require substantially more machinery and numerical residual checks.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
