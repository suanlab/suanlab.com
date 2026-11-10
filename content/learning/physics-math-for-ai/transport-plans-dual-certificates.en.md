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

## Exercise and numerical interpretation

Move the feasible-plan blend away from its optimal endpoint while keeping δ=0. Verify that the weighted slack sum equals the cost increase. Then lower v₁ and separate primal error from dual looseness.

The implementation evaluates complementary-slackness terms directly to avoid losing a small gap by subtracting nearly equal objectives. It still displays the expanded subtraction for comparison. In larger solvers, check marginal residuals and all dual inequalities before using a reported number as a certificate; a low cost at an infeasible plan is not a valid upper bound.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
