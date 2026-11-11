## Store the action rather than a sea of zeros

A one-dimensional Poisson operator with zero boundary values has diagonal two and adjacent entries minus one. Adding reaction c gives diagonal 2+c. For N interior unknowns the matrix has 3N-2 nonzeros. The experiment applies this three-point stencil directly instead of allocating a dense matrix for each iterative product.

The rhs is a vector of ones. A separate pivoted direct solve supplies a reference for measuring errors. In the c=0 case there is also an independent formula: component i, using one-based indexing, is i(N+1-i)/2. This discrete parabola is a useful verification target before trusting a more complicated simulation.

## Weighted Jacobi propagates errors spectrally

Starting at zero, apply

$$
x_{k+1}=x_k+\frac\omega{2+c}(b-Ax_k),\qquad
\lambda_j=2+c-2\cos\frac{j\pi}{N+1}.
$$

The iteration matrix has spectral radius max_j |1-omega lambda_j/(2+c)|. A radius below one gives convergence for every initial error in this symmetric model. A radius above one permits growing modes; it does not imply that every particular vector excites them immediately.

Increasing N with c=0 decreases the smallest eigenvalue. Consequently, smooth error modes become harder for simple stationary iterations to remove. This explains why a fine mesh can require far more iterations even though the sparse matrix action still costs only linear work per iteration.

## CG uses a sequence of conjugate directions

For this SPD matrix, conjugate gradient combines residuals into A-conjugate search directions. The lab actually computes step lengths, new residuals and direction coefficients. It does not draw a prescribed exponential decay curve. The displayed iteration limit is a maximum: once an independently recomputed relative residual is at most 10^-12, CG freezes and its actual update count is reported.

With N=8, c=0 and symmetric rhs ones, CG finishes in four updates in this example. Symmetry reduces the active subspace. Exact-arithmetic termination within N steps should not be interpreted as an unconditional floating-point guarantee for all systems.

## A residual becomes evidence through a bound

Let e=x-xstar and r=b-Ax. Since A is SPD,

$$
\|e\|_2\le\|r\|_2/\lambda_{\min}(A).
$$

The inverse eigenvalue matters: small residual alone is insufficient when the operator is nearly singular. The readout compares actual solution error, true recomputed residual and this bound. Recursive CG residuals can drift in finite precision, so the stopping check uses a fresh stencil action.

## From Krylov optimality to a condition-dependent CG bound

For exact arithmetic with SPD A, CG minimizes the A-norm error over x₀+span(r₀,Ar₀,…,Aᵏ⁻¹r₀). The derivative of the quadratic along each search direction is zero at that minimizer. Conjugate directions diagonalize the restricted quadratic, so successive exact line minimizations give this property. Any competing error has the form p(A)e₀ with degree at most k and p(0)=1. Diagonalizing A gives

$$
\|e_k\|_A\le\min_{p(0)=1,\deg p\le k}
\max_{\lambda\in[\lambda_{min},\lambda_{max}]}|p(\lambda)|\,\|e_0\|_A
\le2\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^k\|e_0\|_A.
$$

The last choice is a scaled Chebyshev polynomial on the spectral interval. Its magnitude is at most one on [−1,1]; its value at the mapped zero grows as half the inverse kth power of the displayed ratio. This supplies the factor two. For κ=1, A is a scalar multiple of identity and one exact step solves the system.

A diagonal example with eigenvalues (1,4,9), rhs ones and zero initial state has κ=9 and ratio 1/2. Execute CG with rational arithmetic and compare squared A-errors:

| k | Guaranteed squared-error factor |
|---|---:|
| 1 | 1 |
| 2 | 1/4 |
| 3 | 1/16 |

```python
from fractions import Fraction as F
A = (F(1), F(4), F(9))
x, r = [F(0)]*3, [F(1)]*3
p = list(r)
initial_error = sum(1/a for a in A)
for k in range(1, 4):
    Ap = [a*v for a, v in zip(A, p)]
    rr = sum(v*v for v in r)
    alpha = rr/sum(v*w for v, w in zip(p, Ap))
    x = [v+alpha*d for v, d in zip(x, p)]
    next_r = [v-alpha*w for v, w in zip(r, Ap)]
    beta = sum(v*v for v in next_r)/rr
    p = [v+beta*d for v, d in zip(next_r, p)]
    r = next_r
    error = sum(a*(v-1/a)**2 for a, v in zip(A, x))
    bound = 4*F(1, 4)**k*initial_error
    assert error <= bound
    print(k, error, bound)
assert error == 0
```

The actual third error is zero here, whereas the bound remains positive. Finite precision can lose conjugacy; the original lab still uses independently recomputed residuals and reports actual updates rather than treating this exact-arithmetic rate as a measured trajectory.

## Exercise: verify the stencil and its cost

For N=4 and c=0, the exact vector is (2,3,3,2). Multiplying by the tridiagonal operator gives four ones. There are ten nonzeros, versus sixteen entries in a dense matrix. Now increase N and inspect the smallest eigenvalue and Jacobi error. Finally set omega=1.2: explain why the reported spectral radius removes a general convergence guarantee even if an early part of one trajectory looks favorable.

Further reading: [SciPy conjugate-gradient documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html). The stencil, spectral calculation and parabola check are derived independently.
