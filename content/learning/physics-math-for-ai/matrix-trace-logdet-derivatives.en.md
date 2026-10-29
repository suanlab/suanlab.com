## A matrix differential maps a matrix direction to a scalar

Under the Frobenius inner product,

$$
dL=\langle\nabla_A L,dA\rangle_F=\operatorname{tr}((\nabla_A L)^T dA).
$$

State shapes before manipulating transposes. The direction dA is a matrix, while dL is scalar. Vectorization and inner-product conventions affect notation.

## Differentiate a trace quadratic

For L(A)=tr(AᵀA), entrywise differentiation gives

$$
dL=2\operatorname{tr}(A^T dA),\qquad\nabla_A L=2A.
$$

The lab uses [[a,b],[b,c]]. Free matrix entries each have gradient 2Aᵢⱼ, but the shared parameter b changes two entries, so its scalar derivative is 4b. Parameter sharing requires its own chain rule.

## Derive log-determinant on the SPD domain

Factor A+tE=A(I+tA⁻¹E), then use det(I+tM)=1+t tr M+O(t²):

$$
D\log\det A[E]=\operatorname{tr}(A^{-1}E),\qquad\nabla_A\log\det A=A^{-T}.
$$

For symmetric A the transpose distinction disappears numerically. This lab fixes the real SPD domain rather than mixing log|det| or complex logarithm branches into the same model.

## Work the reference

For [[2,1],[1,2]], determinant is three and inverse is (1/3)[[2,−1],[−1,2]]. Direction E₀₀ selects the inverse’s first diagonal entry, two thirds:

$$
\frac{\partial}{\partial a}\log(ac-b^2)=\frac c{ac-b^2}.
$$

A central difference uses both a−h and a+h. Both matrices must stay SPD. Singularity gives undefined logdet and derivative, not zero. A one-sided valid evaluation cannot masquerade as a central difference.

## Plot and numerical limits

The horizontal axis perturbs A₀₀ by t. The graph omits values outside the domain instead of joining them through zero. A zero difference step gives —. Step refinement trades truncation error against eventual rounding and cancellation.

Near singularity, inverse entries and derivatives can grow. Distinguish conditioning from implementation failure. Larger systems often use Cholesky and linear solves rather than explicit inverses, while retaining domain checks.

## AI connections and exercises

Gaussian likelihoods, covariance estimation, entropy and logdet barriers use this differential. For A=LLᵀ, derive the gradient with respect to L through the chain rule; it is not simply the gradient with respect to A.

1. At A=2I, the selected derivative is **one half**.
2. At b=1, the shared-b quadratic derivative is **four**.
3. A central difference crossing the SPD boundary **is not a valid derivative check**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
