## A condition in every direction

A real symmetric matrix is positive semidefinite, PSD, when every quadratic value is nonnegative. Positive definite, PD, requires strict positivity for every nonzero vector:

$$
A\succeq0\iff\forall v,\ v^TAv\ge0,\qquad
A\succ0\iff\forall v\ne0,\ v^TAv>0.
$$

Exclude zero for PD because its quadratic value is always zero. Symmetry matters: a general real matrix has the same quadratic form as its symmetric part (A+Aᵀ)/2. In complex spaces use conjugate transpose and Hermitian matrices.

## Proof in an eigenbasis

Orthogonal diagonalization gives A=QΛQᵀ. With z=Qᵀv,

$$
v^TAv=\sum_i\lambda_i z_i^2.
$$

Nonnegative eigenvalues make every term nonnegative. A negative eigenvalue supplies a counterexample by choosing its eigenvector. Thus PSD is equivalent to nonnegative eigenvalues; PD requires strictly positive eigenvalues.

The reference [[2,1],[1,2]] has eigenvalues three and one. Its unit-direction quadratic values range from one to three. The matrix [[1,1],[1,1]] has eigenvalues two and zero and vanishes along (1,−1), so it is PSD but not PD.

## Exact two-dimensional predicates

For A=[[a,b],[b,c]],

$$
A\succeq0\iff a\ge0,\quad c\ge0,\quad ac-b^2\ge0.
$$

When a>0, complete the square: q(x,y)=a(x+(b/a)y)²+(c−b²/a)y². When a=0, the determinant condition forces b=0, leaving cy². Necessity follows by testing basis and minimizing directions, or using eigenvalues.

PD requires a>0 and ac−b²>0. Positive diagonal entries alone are insufficient: [[1,2],[2,1]] has determinant minus three and minimum eigenvalue minus one. Positive determinant alone is also insufficient: −I is negative definite.

## Interpret the directional graph

Set v=(cosθ,sinθ) and plot q=vᵀAv. The horizontal coordinate is angle, the vertical coordinate is quadratic value. This is neither an ellipse nor a probability density. Crossing below the zero line exhibits a negative direction.

The lab certifies PSD with exact integer predicates, not sampled angles. A finite collection of positive plotted values cannot prove a universal statement for an arbitrary unknown matrix; the algebra provides the certificate.

## Gram matrices and covariance

For any real X, G=XᵀX is PSD because

$$
v^TGv=\|Xv\|_2^2\ge0.
$$

It is PD exactly when X has independent columns. Centered sample covariance is a positively scaled Gram matrix, so it is PSD but can be singular with redundant features or limited samples. Numerical roundoff may produce tiny negative computed eigenvalues without changing the exact mathematical theorem.

## AI connections and exercises

PSD Hessians connect to convexity, but a Hessian at one point does not establish global convexity of a nonlinear function. Domain and differentiability assumptions matter. Kernel Gram matrices will reuse PSD reasoning in later chapters.

1. The zero matrix is **PSD, not PD**.
2. A null direction of [[1,1],[1,1]] is **(1,−1)**.
3. XᵀX is PD exactly when **X has independent columns**.

Further reading: [MIT Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/). The interactive model is restricted to small real symmetric integer matrices.
