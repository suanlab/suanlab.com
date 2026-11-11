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

## Turn the residual into one part of a verification budget

The 1D Dirichlet FD matrix is an M-matrix with a nonnegative inverse. The discrete barrier wᵢ=xᵢ(1−xᵢ)/2 satisfies Aw=1 and max w≤1/8. Applying the comparison principle to residual bounds gives an algebraic-error estimate ‖e_alg‖∞≤‖r‖∞/8. With boundary mismatch δ and consistency defect τ for the manufactured target,

$$
\|u_{computed}-u_{target}\|_\infty
\le |\delta|+\tfrac18(\|r\|_\infty+\|\tau\|_\infty).
$$

This separates wrong data, linear-solve error and discretization. For a second-order method, making the algebraic residual O(h²) is a sufficient matching budget in this norm; a fixed iteration count does not enforce it.

For zero-start Jacobi and the sine right-hand side, the error against the exact discrete solution is its amplitude times cos(πh)ᵏ. Requiring this to be at most h² takes O(h⁻²log(1/h)) iterations. Refinement therefore increases both the number of unknowns and the required relaxation work.

| N | Required algebraic-error target | Iteration count |
|---|---|---|
| 8 | h²=1/64 | computed below |
| 16 | h²=1/256 | computed below |
| 32 | h²=1/1024 | computed below |

```python
from fractions import Fraction as F
from math import pi,sin,cos,log,ceil
for N in (4,8,16):
    h=F(1,N);xs=[F(i,N) for i in range(1,N)]
    inverse=[[h*min(x,y)*(1-max(x,y)) for y in xs] for x in xs]
    for x,row in zip(xs,inverse):assert sum(row)==x*(1-x)/2<=F(1,8)
for N in (8,16,32):
    h=1/N;scale=pi*pi*h*h/(4*sin(pi*h/2)**2);rho=cos(pi*h)
    iterations=ceil(log(h*h/scale)/log(rho))
    assert scale*rho**iterations<=h*h
    assert scale*rho**(iterations-1)>h*h
    print(N,iterations,'algebraic error:',scale*rho**iterations)
```

Exact inverse row sums verify the stability constant independently of elimination. The iteration estimate is an auxiliary work calculation; it may exceed the live 500-iteration selector. It explains why a fine-grid error curve can flatten without invalidating the consistency order. A floating residual measurement still needs its own arithmetic-error assessment before it can be used as a rigorous certificate.

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
