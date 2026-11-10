## A cheap factorization can support a better answer

Mixed-precision iterative refinement solves correction equations with lower-precision factors while evaluating residuals and updating the solution more accurately. The idea is to reuse expensive factorization work without accepting its initial solution error as final. Whether it succeeds depends on conditioning, factor quality and the precision used in several distinct stages.

The lab uses A=[[1,1],[1,1+delta]], delta=2^-k, and b=(2,2+delta), so the exact solution is (1,1). Low precision means p significant binary bits with ties to even and unbounded exponent range. It is a controlled arithmetic model rather than an emulation of every feature of a named hardware dtype.

## Follow the rounded elimination

The first pivot is one. After eliminating the first column, the second pivot is fl(fl(1+delta)-1). The rhs undergoes its own rounding, and its larger magnitude gives it a different representable spacing.

At delta=1/8 and p=4, 1.125 is representable but 2.125 is halfway between two p-bit values and rounds to two. The transformed rhs therefore loses its second component. Low-precision substitution gives xhat=(2,0), not the exact answer. This is a concrete rounding mechanism, rather than a fabricated noise term attached to an otherwise exact solve.

## Residual precision changes the correction

Refinement performs

$$
r=b-A\hat x,\qquad A_{\mathrm{low}}d\approx r,\qquad
\hat x_{\mathrm{new}}=\hat x+d.
$$

The residual can be evaluated with original data in binary64 or through the selected toy arithmetic. Factors and correction solves always remain low precision, while the accumulated update uses binary64. These roles are deliberately separate controls and readout meanings.

For xhat=(2,0), the true residual is (0,1/8). Solving the correction gives d=(-1,1), recovering (1,1) in one update. In the low-residual variant, the rhs has already rounded to (2,2); the computed residual is zero. The method stagnates while the original-system residual norm remains 0.125 and relative solution error remains one.

## Some factors are unusable

If delta is too small for the selected precision, fl(1+delta)=1 and the second pivot becomes zero. The lab marks factorization invalid and leaves solution and error absent. Plotting a zero error curve in this state would falsely claim success.

For this SPD matrix, the determinant is delta. Computing the small eigenvalue as delta/lambda_max avoids cancellation when estimating its condition number. A sufficient convergence theorem for a particular refinement algorithm needs explicit precision and factor-error assumptions; the success of the worked example is not a theorem for every ill-conditioned matrix.

## Exercise: expose a false stopping criterion

Start with k=3, p=4 and zero corrections. Verify (2,0) by rounding the rhs before elimination. Perform one high-residual correction and then repeat with low-residual evaluation. Explain why stopping solely on the low computed residual would accept a wrong original-system solution. Finally choose k=6 and p=3: show how the rounded zero pivot prevents the correction process from starting at all.

Further reading: [LAPACK error measures and accuracy framework](https://netlib.org/lapack/lug/node75.html). The toy factors, correction and failure example are derived independently.
