## Three solutions, three different questions

A small discrepancy is meaningful only after naming its reference. Let A=diag(1,delta), delta=10^-k, and b=(1,delta), so the original solution is x=(1,1). Perturb the right-hand side to b_eta and call its exact solution x_eta. Finally compute a rounded solution xhat. These three objects separate data uncertainty from arithmetic error.

The data-induced change is ||x_eta-x||/||x||. Arithmetic forward error is ||xhat-x_eta||/||x_eta||. The total change from the original solution can contain both, and neither should be silently substituted for the other. The lab explicitly reports the distinct quantities using Euclidean norms.

## Conditioning belongs to the mathematical problem

The singular values of this diagonal matrix are one and delta, giving

$$
\kappa_2(A)=\|A\|_2\|A^{-1}\|_2=1/\delta.
$$

Add eta=10^-q to the weak second coordinate. Its solution becomes (1,1+eta/delta). With delta=eta=10^-6, a tiny absolute data change doubles the second component. Increasing arithmetic precision cannot remove this sensitivity: the exact perturbed problem already has that different answer.

Apply the same perturbation to the first coordinate instead. The solution change is only eta. A condition number describes the worst possible direction in the chosen norm; it does not assert that every data change experiences that amplification. The two presets make this distinction visible with identical perturbation magnitudes.

## Backward error asks what was actually solved

Toy p-bit arithmetic rounds the diagonal entries, the perturbed rhs and each division. Afterward evaluate the residual r=A xhat-b_eta using the original matrix and binary64. Since A xhat=b_eta+r, the computed answer exactly solves a nearby rhs problem in the mathematical model.

$$
\eta_b=\frac{\|r\|_2}{\|b_\eta\|_2},\qquad
\frac{\|\hat x-x_\eta\|_2}{\|x_\eta\|_2}
\le\kappa_2(A)\eta_b.
$$

This is rhs-only backward error with A fixed. A general solver may also perturb matrix entries, and componentwise definitions can tell a different story. Norm, denominator and permitted perturbation must therefore accompany an error claim.

## Read the precision experiment

The blue curve is the exact data-induced solution change and does not depend on toy precision. Orange measures arithmetic error against x_eta. Higher precision can reduce orange while leaving blue unchanged. This is the practical difference between improving an algorithm and obtaining more accurate observations.

## Exercise: equal data sizes, unequal effects

For delta=10^-6 and eta=10^-6, calculate both perturbed exact solutions. Weak-direction perturbation gives (1,2), a relative change 1/sqrt(2). Strong-direction perturbation gives (1.000001,1), a relative change 10^-6/sqrt(2). Their rhs perturbation norms are equal. Explain why both obey the same condition-number upper bound, even though only one is strongly amplified.

Further reading: [LAPACK error measures](https://netlib.org/lapack/lug/node75.html). The diagonal experiment and its error decomposition are independently derived.
