## Three errors require three different checks

An entropic transport computation can differ from a desired continuum problem because of regularization, incomplete optimization and discretized marginals. Increasing the grid size does not automatically control the other two. This lab separates them in a family with an independently known continuum answer.

Let μ be uniform on [0,1] and ν its translation by s. Their W₂ distance is |s|. Use N=2ᵏ midpoint atoms for each law. Matching corresponding midpoints also gives discrete W₂=|s| for every N, so actual pair-distance discretization error is zero.

## A positive bound can coexist with zero actual error

For each individual marginal, direct integration over midpoint cells gives

$$
W_2(\mu,\mu_N)=\frac{1}{\sqrt{12}N},\qquad
|W_2(\mu,\nu)-W_2(\mu_N,\nu_N)|
\le\frac{2}{\sqrt{12}N}.
$$

The bound is positive while the actual pair error in this translation family is zero. It bounds perturbations; it does not claim equality. These quantities are distances, whereas the regularization certificate below is in squared-cost units.

## Restore feasibility before using a primal bound

A finite Sinkhorn matrix may have wrong marginals. The experiment converts it into a feasible plan with mass quantum 2⁻²⁴. Every row and column has integer capacity 2²⁴/N. First retain floored cell masses subject to remaining capacities, then distribute residual integer capacities.

All resulting entries are nonnegative integer multiples of the quantum. Their row and column sums are exactly 1/N in binary arithmetic. This operation is a feasibility correction, not another Sinkhorn convergence theorem. Its reported L1 correction includes both marginal repair and quantization effects.

## Combine an entropy bound with an optimization certificate

Let G be the regularized primal–dual Fenchel gap for the corrected feasible plan and the current potentials. For uniform N-point marginals, joint entropy lies between log N and 2log N. Comparing with the unregularized permutation plan yields

$$
0\le \langle C,\gamma_{feas}\rangle-OT
\le\epsilon\log N+G.
$$

The εlog N term bounds the transport-cost bias of the exact regularized optimizer. G separately controls regularized optimization error. Neither term is the marginal-discretization bound above. This worst-case entropy bound can grow with N without proving that actual bias grows.

Blue plots corrected excess transport cost and orange its certificate. The excess is evaluated as Σγᵢⱼ(xᵢ−xⱼ)², avoiding cancellation against the translation cost s². The feasible regularized objective and dual are also shown with the stable generalized-KL gap.

## Exercise and explained answer

At N=8,s=0.5,ε=0.1 and twenty iterations, each marginal quantization distance is approximately 0.036084, actual pair error zero and corrected excess cost approximately 0.038481. The regularization bound is approximately 0.207944 and the regularized gap about 5.71×10⁻⁸.

Double N: marginal quantization distance halves, but optimization residual and entropy effects must still be inspected. Set zero iterations and observe that exact feasibility after correction does not imply a small regularized certificate. Report all these accuracy notions before claiming that a grid computation approximates a continuum transport problem.

Further reading: [Peyré and Cuturi, Computational Optimal Transport](https://optimaltransport.github.io/book/).
