## Verification stronger than a plausible picture

Developing a solver only on problems with unknown answers makes error diagnosis difficult. A manufactured solution chooses a smooth u first and derives source and boundary data by substitution. Here u=sin(πx), -u″=π²sin(πx), with zero endpoint values. Failure on this known problem would undermine confidence in a more complicated research simulation.

Actual finite-difference systems are assembled on N, 2N and 4N intervals. Direct mode uses tridiagonal elimination. Jacobi mode starts from zero and performs the specified number of actual iterations. A deliberate left-boundary bias also lets us solve the wrong mathematical problem very accurately.

## Compare errors in the same norm

$$
E_\infty=\max_i|u_i-u(x_i)|,\qquad
E_2=\sqrt{h\left[\tfrac12e_0^2+\sum_{i=1}^{N-1}e_i^2+\tfrac12e_N^2\right]},\qquad
p_{obs}=\log_2(E_h/E_{h/2}).
$$

The maximum includes boundary nodes. The weighted L2 norm approximates a continuous integral using trapezoidal weights; it is different from an unweighted vector Euclidean norm. Ratios between different norms or physical domains would not measure the intended order. Inspect two successive ratios to assess whether an asymptotic range is emerging.

Central differences are expected to show approximately second-order error for this smooth solution, provided algebraic and boundary-model errors are sufficiently small. The plot uses three computed errors rather than an imposed h² line.

## Residual is not solution error

The algebraic residual r=b-Au measures how well the discrete equations were solved. Solution error measures distance from the continuous target. Algebraic error relative to the exact discrete solution involves A^{-1}r, but neither discretization error nor incorrect boundary data is necessarily exposed by a small r.

Changing the left boundary to 0.1 defines a different boundary-value problem. A direct solve can have a residual near roundoff while its error against the original exact solution does not approach zero. Conversely, a fixed small number of Jacobi iterations may leave dominant algebraic error, especially on finer meshes where low-frequency relaxation is slower.

## Worked example and exercise

The sine sample vector is an eigenvector of the FD matrix. With correct boundary data the exact discrete solution therefore has amplitude

$$
u_i=\frac{\pi^2h^2}{4\sin^2(\pi h/2)}\sin(\pi x_i).
$$

Expanding the denominator gives amplitude 1+π²h²/12+O(h⁴). Hence maximum error has a leading h² term and measured order approaches two. This analytic discrete answer independently checks the elimination solver rather than comparing it only with itself.

Exercise: why must maximum error be at least 0.1 when the left-boundary bias is δ=0.1? Solution: the norm includes x=0, where the computed value is already 0.1 away from the true zero boundary. A tiny interior matrix residual cannot hide this disagreement.

## Record reproducible validation

For AI surrogates and inverse problems, record the PDE, manufactured function, source derivation, boundaries, norm, mesh, and linear-solver tolerance or iteration count. Roundoff plateaus and conditioning deserve separate checks at smaller h. A finite table cannot prove convergence for every problem or validate a physical data model. Matching several independent identities on a transparent example is a starting point for those larger claims, not their replacement.

Reference: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
