## Spaces and spans

A real vector space is closed under vector addition and real scaling and satisfies the vector-space axioms. R² is a familiar example, but suitably defined spaces of functions or matrices also qualify. Vectors with strictly positive components do not: they exclude zero and are not closed under negative scaling.

The span consists of all linear combinations:

$$
\operatorname{span}(v_1,\ldots,v_k)=\left\{\sum_{i=1}^k\alpha_i v_i:\alpha_i\in\mathbb R\right\}.
$$

It is a subspace. The empty span is {0}; that space has one element but dimension zero. Cardinality and dimension answer different questions.

## Independence and bases

Vectors are independent if the zero combination has only zero coefficients:

$$
\alpha_1v_1+\cdots+\alpha_kv_k=0\implies\alpha_1=\cdots=\alpha_k=0.
$$

An independent spanning set is a basis. Every basis of a finite-dimensional space has the same number of vectors, its dimension. Basis coordinates are unique: subtracting two representations gives a zero combination, and independence makes every coefficient difference zero.

For A=[[1,2],[2,4]], the columns are (1,2) and (2,4). The second is twice the first. One nonzero column is a basis of the column space, whose dimension is one.

## Image and nullspace

For A:Rⁿ→Rᵐ, the image is its column space, and rank is the image dimension. The nullspace contains inputs mapped to zero:

$$
\ker A=\{x:Ax=0\},\qquad\operatorname{im}A=\{Ax:x\in\mathbb R^n\}.
$$

In the reference, A(−2,1)ᵀ=0, giving kernel span((−2,1)). The vector (−4,2) generates the same space. Bases are generally not unique; the displayed generator is one choice.

## Why rank plus nullity equals input dimension

Extend a kernel basis k₁,…,kᵣ to an input basis k₁,…,kᵣ,u₁,…,uₛ. The vectors Auⱼ span the image because kernel components vanish. If ΣβⱼAuⱼ=0, then Σβⱼuⱼ belongs to the kernel. Independence of the extended basis forces every βⱼ to zero, so the Auⱼ form an image basis. Hence

$$
\dim\ker A+\operatorname{rank}A=n.
$$

The right-hand side is input dimension n, not output dimension m. In this two-by-two example it is two; for rectangular matrices it remains the number of columns.

## Interpret the picture and boundaries

The picture shows columns in output space. Kernel vectors live in input space. Both happen to be R² here, but their roles differ. Equal axis units preserve geometry; overlapping columns do not add independent directions.

The identity has rank two and zero-dimensional kernel {0}. The zero map has rank zero and kernel R² of dimension two. Neither case is represented by one nonzero kernel generator, so that readout is — while the kernel dimension remains explicit.

Small integer entries permit exact determinant and zero predicates. With floating-point data, numerical rank can depend on tolerance and singular-value scale. A tiny nonzero singular value is not automatically an exact zero.

## AI connections and exercises

A linear layer cannot distinguish inputs differing by a kernel vector. A rank restriction limits its output subspace, but does not alone characterize an entire nonlinear network’s expressiveness.

1. Nullity two for an input R⁵ implies **rank three**.
2. Adding a zero vector to a spanning list **does not change the span**.
3. Distinct vectors (1,2),(2,4) are **dependent**, because they are proportional.

Further reading: [MIT Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/). The examples and dimension argument are independently constructed.
