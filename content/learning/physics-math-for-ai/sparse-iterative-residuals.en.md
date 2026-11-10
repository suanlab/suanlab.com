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

## Exercise: verify the stencil and its cost

For N=4 and c=0, the exact vector is (2,3,3,2). Multiplying by the tridiagonal operator gives four ones. There are ten nonzeros, versus sixteen entries in a dense matrix. Now increase N and inspect the smallest eigenvalue and Jacobi error. Finally set omega=1.2: explain why the reported spectral radius removes a general convergence guarantee even if an early part of one trajectory looks favorable.

Further reading: [SciPy conjugate-gradient documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html). The stencil, spectral calculation and parabola check are derived independently.
