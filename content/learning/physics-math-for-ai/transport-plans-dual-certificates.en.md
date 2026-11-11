## A transport plan needs an independently checkable lower bound

For fixed source and target marginals, nonnegative couplings form a linear-program feasible set. The dual assigns a potential uᵢ to each source and vⱼ to each target:

$$
\max_{u,v}\sum_i a_i u_i+\sum_j b_j v_j,
\qquad u_i+v_j\le C_{ij}.
$$

Potentials can be negative. Adding a constant to all u and subtracting it from all v leaves the objective and inequalities unchanged when total masses agree. The lab fixes one gauge by setting u₀=0.

## Derive weak duality from marginal conservation

For a feasible plan and feasible potentials,

$$
\sum_{ij}\gamma_{ij}C_{ij}
-\left(\sum_i a_i u_i+\sum_j b_j v_j\right)
=\sum_{ij}\gamma_{ij}(C_{ij}-u_i-v_j)\ge0.
$$

Every term is mass times nonnegative slack. At an optimal pair, positive transported mass can use only tight edges. Zero mass does not require a positive slack: unused tight edges can exist.

The certified gap bounds current primal suboptimality without requiring the true optimum. This small family also computes the exact optimum, allowing the certificate to be compared with actual error.

## Work a complete primal–dual pair

Source positions are (0,1), targets (0,L), with squared costs [[0,L²],[1,(L−1)²]]. When a≤b, one optimal potential choice is u=(0,1), v=(0,L²−2L). When a>b, use u=(0,1−2L), v=(0,L²). The monotone coupling makes all positive-flow edges tight.

For a=0.4,b=0.6,L=2, the optimal coupling is [[0.4,0],[0.2,0.4]]. Potentials u=(0,1),v=(0,0) give dual value 0.6, matching the plan cost. The unused edge from zero to L has slack four.

## Perturb a potential and inspect feasibility first

The control adds δ to v₁. Negative δ preserves the inequalities but lowers the dual objective by |δ|(1−b). At δ=−1 the dual is 0.2 and the gap is 0.4, despite the candidate plan already being optimal.

Positive δ violates a tight inequality. At δ=1 the algebraic primal-minus-dual difference is −0.4. This is not a weak-duality counterexample because the potential vector is infeasible. The readout withholds a certified bound and keeps the signed algebraic value visible.

## Repair dual feasibility and quantify mass on slack edges

For arbitrary finite potentials define their largest positive violation δ=max(0,maxᵢⱼ(uᵢ+vⱼ−Cᵢⱼ)). Subtract δ from every source potential. This makes every inequality feasible and lowers the dual value by δ for unit total mass. With a genuinely feasible candidate plan P, the repaired certificate is

$$
0\le\langle C,P\rangle-OT
\le\langle C,P\rangle-D(u,v)+\delta.
$$

For a different total mass M the correction is Mδ. A numerical violation estimate must itself bound all inequalities; a small subset of inspected edges is insufficient. Repairing potentials does not repair a candidate with incorrect marginals.

For any feasible primal–dual pair with gap G, edges whose slack is at least τ>0 carry total mass at most G/τ. This follows by discarding the other nonnegative terms in G=ΣPᵢⱼslackᵢⱼ. It turns complementary slackness into a quantitative statement, rather than treating every tiny positive flow as evidence of an exactly tight edge.

Use P=[[1/5,1/5],[2/5,1/5]], C=[[0,4],[1,1]], and potentials u=(0,1),v=(0,d). The true candidate error is 4/5:

| d | Raw dual | Repaired dual | Gap |
|---|---:|---:|---:|
| 0 | 3/5 | 3/5 | 4/5 |
| 1/10 | 16/25 | 27/50 | 43/50 |
| 1 | 1 | 0 | 7/5 |

```python
from fractions import Fraction as F
P = ((F(1, 5), F(1, 5)), (F(2, 5), F(1, 5)))
C, a, b = ((0, 4), (1, 1)), (F(2, 5), F(3, 5)), (F(3, 5), F(2, 5))
primal = sum(P[i][j]*C[i][j] for i in range(2) for j in range(2))
for d in (F(0), F(1, 10), F(1)):
    u, v = (F(0), F(1)), (F(0), d)
    violation = max(F(0), max(u[i]+v[j]-C[i][j] for i in range(2) for j in range(2)))
    repaired = tuple(x-violation for x in u)
    raw_dual = sum(x*y for x, y in zip(a, u))+sum(x*y for x, y in zip(b, v))
    dual = raw_dual-violation
    slack = [[C[i][j]-repaired[i]-v[j] for j in range(2)] for i in range(2)]
    gap = sum(P[i][j]*slack[i][j] for i in range(2) for j in range(2))
    assert min(x for row in slack for x in row) >= 0
    assert gap == primal-dual and gap >= primal-F(3, 5)
    assert P[0][1] <= gap/4
    print(d, raw_dual, dual, gap)
```

The program performs a separate repair. The original lab correctly continues to withhold a certificate for its unmodified infeasible potentials.

## Exercise and numerical interpretation

Move the feasible-plan blend away from its optimal endpoint while keeping δ=0. Verify that the weighted slack sum equals the cost increase. Then lower v₁ and separate primal error from dual looseness.

The implementation evaluates complementary-slackness terms directly to avoid losing a small gap by subtracting nearly equal objectives. It still displays the expanded subtraction for comparison. In larger solvers, check marginal residuals and all dual inequalities before using a reported number as a certificate; a low cost at an infeasible plan is not a valid upper bound.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
