## Regularization changes the optimization problem

For positive marginals, the experiment minimizes

$$
J_\epsilon(\gamma)=\langle C,\gamma\rangle+
\epsilon\sum_{ij}\gamma_{ij}(\log\gamma_{ij}-1)
\quad\text{over }\Pi(a,b).
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

## Exercise and explained answer

Reduce ε to 0.05 in that symmetric case. Off-diagonal reference mass is about 1.030577×10⁻⁹, still positive. Then select the asymmetric zero-iteration kernel: its raw objective can lie below the true regularized optimum because it is infeasible. Identify the row and column errors before interpreting its cost.

A small residual, a regularized gap and a small bias relative to unregularized transport are distinct accuracy questions. The displayed bounded example does not guarantee that a fixed iteration count solves every transport problem.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
