## Regularization changes the optimization problem

For positive marginals, the experiment minimizes

$$
J_\epsilon(\gamma)=\langle C,\gamma\rangle+
\epsilon\sum_{ij}\gamma_{ij}(\log\gamma_{ij}-1),
\qquad\gamma\in\Pi(a,b).
$$

This entropy convention matters: the regularized value can be negative and is not a Wasserstein metric. Its transport-cost term can exceed the unregularized optimum because entropy rewards spreading mass. Sending ε toward zero and solving each regularized problem accurately are separate operations.

## Alternate row and column scaling in logarithmic coordinates

The entropy-conjugate dual is

$$
D(f,g)=a^Tf+b^Tg-\epsilon\sum_{ij}
\exp((f_i+g_j-C_{ij})/\epsilon).
$$

Finite potentials are unconstrained here; they do not obey the unregularized inequality fᵢ+gⱼ≤Cᵢⱼ. Their exponential term supplies the correct conjugate penalty.

Updating f enforces row marginals and updating g enforces columns. The code uses log-sum-exp to evaluate these scalings without directly dividing tiny kernels. The table records complete row–column cycles. After a column update, column residual is small but row residual can remain nonzero.

## A finite iterate need not be a feasible upper bound

The raw plan is exp((f+g−C)/ε). Before convergence, its cost or regularized objective is not a primal upper bound unless both marginals match. The dual objective remains a lower bound for any finite potentials.

For this two-by-two family a positive reference solution can be computed independently. Its entries satisfy the cross-ratio condition

$$
\frac{\gamma_{00}\gamma_{11}}{\gamma_{01}\gamma_{10}}
=\exp(2L/\epsilon).
$$

A stable positive quadratic root determines the small off-diagonal mass directly, instead of subtracting it from a nearly saturated diagonal. In the balanced L=1 case the off-diagonal mass is 1/[2(1+exp(1/ε))].

## Fenchel residual gives a stable comparison

Using the feasible reference γ*, its primal–dual gap equals ε times the sum of scalar generalized KL terms γ*log(γ*/γ_raw)−γ*+γ_raw. These terms are evaluated with stable local remainders when masses nearly agree. This avoids subtracting nearly equal objective values and preserves very small positive residuals.

At a=b=0.5,L=1,ε=0.5, one complete cycle solves the symmetric scaling. Reference transport cost is approximately 0.119203, while the regularized objective is approximately −0.910038. Neither value should be mislabeled as unregularized W₂.

## Distinguish a regularized optimizer certificate from transport bias

Finite costs, strictly positive marginals and ε>0 give a unique entropic optimizer. The feasible set is compact and the entropy term extends continuously at zero. Strict convexity gives uniqueness. An optimizer cannot have a zero entry: blending toward the positive product coupling would have an entropy directional derivative of −∞ at that entry, outweighing finite cost derivatives.

At the interior optimum P*, stationarity makes ∇Jε(P*) a sum of row and column potentials. For any feasible probability plan P, their pairing with P−P* vanishes. Expanding the objective gives

$$
J_\epsilon(P)-J_\epsilon(P^*)
=\epsilon\sum_{ij}P_{ij}\log(P_{ij}/P^*_{ij}),\qquad
\|P-P^*\|_1\le\sqrt{2G/\epsilon},
$$

where G is any valid regularized primal–dual gap for P. The second bound uses Pinsker's inequality for the flattened probability matrices and Jε(P)−Jε(P*)≤G. It is not a bound obtained by applying probability Pinsker directly to an infeasible raw matrix with mass different from one.

For equal half marginals, L=1,ε=1/2, write s=1/[2(1+exp2)] for the optimal off-diagonal mass. A feasible symmetric plan with off-diagonal r has L1 error 4|r−s|:

| r | L1 error |
|---|---:|
| 0 | 4s |
| s | 0 |
| 1/4 | 1−4s |

```python
from decimal import Decimal as D, localcontext
with localcontext() as ctx:
    ctx.prec = 80
    eps = D('0.5')
    s = 1/(2*(1+D(2).exp()))
    reference = (D('0.5')-s, s, s, D('0.5')-s)
    def objective(P):
        return P[1]+P[2]+eps*sum(x*(x.ln()-1) if x else D(0) for x in P)
    for r in (D(0), s, D('0.25')):
        P = (D('0.5')-r, r, r, D('0.5')-r)
        gap = objective(P)-objective(reference)
        kl = sum(x*(x/y).ln() if x else D(0) for x, y in zip(P, reference))
        error = sum(abs(x-y) for x, y in zip(P, reference))
        assert abs(gap-eps*kl) < D('1e-75')
        assert error*error <= 2*kl+D('1e-75')
        print("r,L1,gap:", r, error, gap)
```

A different small quantity is the reference's excess unregularized cost. In the two-atom family it is exactly 2Ls. The lab evaluates this product directly: at L=3,ε=0.05 it retains about 2.63×10⁻²⁶ even when subtracting two rounded transport costs gives zero. The absolute costs remain separate diagnostics. Neither a small optimizer gap nor this specialized formula is a universal entropy-bias estimate.

## Exercise and explained answer

Reduce ε to 0.05 in that symmetric case. Off-diagonal reference mass is about 1.030577×10⁻⁹, still positive. Then select the asymmetric zero-iteration kernel: its raw objective can lie below the true regularized optimum because it is infeasible. Identify the row and column errors before interpreting its cost.

A small residual, a regularized gap and a small bias relative to unregularized transport are distinct accuracy questions. The displayed bounded example does not guarantee that a fixed iteration count solves every transport problem.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
