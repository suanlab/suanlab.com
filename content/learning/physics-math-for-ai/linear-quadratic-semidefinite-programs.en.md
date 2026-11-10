## Three optimization languages

Linear programs use affine objectives and constraints. Convex quadratic programs add a positive semidefinite quadratic objective. Semidefinite programs constrain an affine symmetric matrix to be positive semidefinite. These descriptions specify structure that algorithms and certificates can exploit; they do not imply that every quadratic objective is convex.

The lab solves three small families exactly. It is useful to compare their geometry before considering a general solver. Blue marks the feasible boundary, orange marks an optimal objective line or the projection segment. The query point is independent of the optimizer and can be infeasible.

## Linear objectives on a triangle

Let T contain nonnegative pairs with x₁+x₂≤1. Every point is a convex combination of its three vertices, so the minimum of c·x is min(0,c₁,c₂). A vertex attains the minimum, but the optimum need not be unique. With c=(−1,−1), every point on the edge x₁+x₂=1 is optimal.

Introduce λ≥0 for the sum constraint and μ≥0 for nonnegativity. The dual equations are

$$
c+\lambda(1,1)-\mu=0,\qquad d=-\lambda.
$$

Choosing λ=max(0,−c₁,−c₂) and μᵢ=cᵢ+λ gives a lower bound equal to the best vertex value. This is a certificate, rather than evidence based only on a dense plot.

## Quadratic projection

The second family minimizes half the squared Euclidean distance from c to T. Strict convexity gives a unique projection. KKT conditions give xᵢ=max(cᵢ−λ,0), where λ is zero if the positive parts already sum to at most one; otherwise their sum is exactly one.

For c=(2,0.5), the projection is (1,0), λ=1 and μ=(0,0.5). The objective is 0.625. Writing r=λ(1,1)−μ, the dual value c·r−‖r‖²/2−λ is also 0.625. A nearby feasible query can have a larger objective while still satisfying every constraint.

## A genuine matrix inequality

The SDP family uses two variables a,b:

$$
\min c_1a+c_2b,\qquad
F(a,b)=\begin{pmatrix}1&a&b\\a&1&0\\b&0&1\end{pmatrix}\succeq0.
$$

The Schur complement of the lower identity block gives a²+b²≤1. Equivalently, the eigenvalues are 1 and 1±√(a²+b²). Thus a circular picture represents an actual matrix constraint, rather than relabeling a scalar quadratic program.

For nonzero c the optimum is −c/‖c‖ with value −‖c‖. Let u=c/‖c‖ and v=(1,u₁,u₂). The matrix Z=‖c‖vvᵀ/2 is positive semidefinite. Its off-diagonal trace constraints match c, and the dual value −tr Z equals −‖c‖. At the optimum Fv=0, proving complementary slackness. Negative individual entries do not prevent a matrix from being positive semidefinite.

## Exercise and solver limitations

Use c=(1.2,1.6): the SDP optimum is (−0.6,−0.8), value −2. Construct Z=vvᵀ and verify its trace is two. Then put the query at (1,1): its minimum eigenvalue is 1−√2, so its objective cannot be used as a feasible upper bound.

Set both costs to zero and compare uniqueness in the LP, QP and SDP modes. The QP remains uniquely minimized at the origin; the linear families have many optimizers. These closed forms solve the displayed families only. General SDP algorithms require substantially more machinery and numerical residual checks.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
