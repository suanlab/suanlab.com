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

## Differentiate through a covariance factor

Let A=LLᵀ with a real, square, invertible d×d matrix L. For a differentiable scalar loss ℓ(A), put G=∇_Aℓ. Since dA=dL Lᵀ+L dLᵀ, cyclic trace identities give

$$
d\ell=\operatorname{tr}(G^T dL L^T)+\operatorname{tr}(G^T L dL^T)
=\operatorname{tr}\left(((G+G^T)L)^T dL\right).
$$

Hence ∇_Lℓ=(G+Gᵀ)L. In particular, logdet has G=A⁻¹ on the SPD domain, so

$$
\nabla_L\log\det(LL^T)=2A^{-1}L=2L^{-T}.
$$

The factor gradient describes how changing L changes both occurrences of L in A. This is why copying the gradient with respect to A would give the wrong answer.

## Identify the actual free Cholesky parameters

For lower-triangular L with positive diagonal, logdet(LLᵀ)=2Σᵢlog Lᵢᵢ. The free lower-triangular entries therefore have derivatives 2/Lᵢᵢ on the diagonal and zero strictly below it. The full matrix gradient above also includes directions outside this triangular parameter space.

Take L=[[2,0],[1,3]]. Then A=[[4,2],[2,10]], det A=36 and

$$
2L^{-T}=\begin{pmatrix}1&-1/3\\0&2/3\end{pmatrix}.
$$

| Free parameter | Entries of A that change | Derivative of logdet |
|---|---|---|
| L₁₁ | A₁₁ and both off-diagonal entries | 1 |
| L₂₁ | both off-diagonal entries and A₂₂ | 0 |
| L₂₂ | A₂₂ | 2/3 |

Changing L₂₁ changes the covariance but leaves its determinant unchanged when the diagonals are held fixed. The upper-right value −1/3 in the full gradient multiplies dL₁₂, which is zero for a triangular perturbation.

## Reproduce the factor derivatives in Python

This small example forms A’s entries and its 2×2 determinant directly, then compares central differences in all three free factor coordinates with the derived gradient. The positive perturbed diagonals retain the SPD domain. Direct determinant arithmetic is suitable for this check; larger systems can evaluate logdet from a Cholesky diagonal.

```python
from math import log

def logdet_from_factor(l11, l21, l22):
    assert l11 > 0 and l22 > 0
    a, b, c = l11**2, l11 * l21, l21**2 + l22**2
    return log(a * c - b * b)

point = [2.0, 1.0, 3.0]
assert abs(logdet_from_factor(*point) - 2 * log(6)) < 1e-12
h = 1e-5
for i, expected in enumerate([1.0, 0.0, 2 / 3]):
    plus, minus = point.copy(), point.copy()
    plus[i] += h
    minus[i] -= h
    difference = (logdet_from_factor(*plus) - logdet_from_factor(*minus)) / (2 * h)
    assert abs(difference - expected) < 1e-8
    print(i, difference, expected)
```

Exercise: substitute L₁₁=exp(s) and L₂₂=exp(t). Their logdet derivatives become two by the scalar chain rule. A covariance likelihood contains other terms too; the zero logdet derivative with respect to L₂₁ does not make that parameter irrelevant to the full loss.

## AI connections and exercises

Gaussian likelihoods, covariance estimation, entropy and logdet barriers use this differential. For A=LLᵀ, apply the factor chain rule above using the actual free parameters.

1. At A=2I, the selected derivative is **one half**.
2. At b=1, the shared-b quadratic derivative is **four**.
3. A central difference crossing the SPD boundary **is not a valid derivative check**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
