## Conjugation gives a positive norm

Complex vector spaces use componentwise addition and complex scaling. Under the convention conjugating the first argument,

$$
\langle v,w\rangle=v^*w=\sum_j\overline{v_j}w_j,\qquad
\|v\|_2^2=\sum_j|v_j|^2.
$$

The first argument is conjugate-linear and the second linear. Some texts reverse the convention, so check before transferring formulas. Conjugate symmetry relates the swapped products.

## Plain transpose loses positivity

The nonzero vector v=(1,i) has vᵀv=1+i²=0. That bilinear expression is not a squared norm. Instead,

$$
v^*v=1+(-i)i=2.
$$

Plain transpose can therefore report zero for a nonzero complex vector. Bilinear forms can be useful, but they are different objects from the positive Hermitian inner product.

## Phase-dependent orthogonality

The lab fixes v=(1,i) and uses w=s(1,e^{iθ}). At θ=90 degrees,s=1, w=v and the product is two. At −90 degrees,

$$
v^*w=1+(-i)(-i)=0.
$$

The vectors are Hermitian-orthogonal, while their bilinear product is two. Cauchy–Schwarz bounds |v*w| by the norm product. Its normalized magnitude lies in [0,1] for nonzero vectors; at s=0 it is undefined, not arbitrarily assigned a similarity.

## Unitary and Hermitian structure

$$
U^*U=I\implies\|Uv\|_2=\|v\|_2,\qquad A^*=A\text{ defines Hermitian }A.
$$

Unitary maps preserve complex inner products. Hermitian matrices have real eigenvalues and orthonormal eigenbases. Extending real symmetric results requires conjugate transpose. Complex PSD means v*Av≥0 for all complex v.

## Interpret the component picture

The figure draws two complex components of w on one plane. C² has real dimension four; this is not a geometry-preserving embedding of the whole vector space. Combine component plots with full-vector numeric products.

Complex attention, Fourier coefficients and quantum overlaps depend on conjugation conventions. One norm-preserving layer does not establish that an entire nonlinear model is unitary.

## Exercises and answers

1. The norm of (1,i) is **√2**.
2. If w=iv, normalized absolute overlap is **one**: absolute similarity ignores a global phase.
3. Self-products are **nonnegative real**, while products of different vectors may be complex.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
