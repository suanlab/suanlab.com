## Represent a symmetry with matrices

A group representation sends group elements to invertible linear maps while preserving composition. Here J is a quarter-turn and S is reflection in the horizontal axis, acting on real two-dimensional vectors. This is a stronger requirement than a generic numerical embedding. Checking J to the fourth equals identity and SJS equals the inverse of J exhibits the square-group relations.

$$
\rho(gh)=\rho(g)\rho(h),\qquad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad S=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$

## An equivariant linear-layer condition

Using the same representation for inputs and outputs, a linear layer must satisfy MJ=JM for rotation equivariance. With different representations, the condition is Mρin(g)=ρout(g)M. This lab treats only the same two-dimensional representation. The default M has rows (2,1) and (0,1), and its rotation commutator has Frobenius norm two. A nonzero residual refutes equivariance for all inputs.

$$
[M,J]=MJ-JM,\qquad \|[M,J]\|_F=0\iff MJ=JM.
$$

## Project onto the constraint

Average conjugation by the four rotations to obtain a matrix commuting with rotation. The program evaluates the four matrix products directly. For entries a,b,c,d, the projected diagonal is (a+d)/2 and the upper-right entry is (b−c)/2, with its negative below. The default projection has rows (1.5,0.5) and (−0.5,1.5). Compare each original and projected entry in the table.

$$
P_{C_4}(M)=\frac14\sum_{k=0}^3J^{-k}MJ^k.
$$

## Additional symmetries and research scope

A matrix commuting with rotations need not commute with reflection. Adding the D4 reflection constraint leaves the scalar matrix (a+d)/2 times identity. The two projections illustrate how the chosen group changes permitted model parameters. Constraining one matrix does not automatically make nonlinear activations or an entire network equivariant. Continuous rotation groups, other representations and finite-field characteristics require separate analysis.

$$
P_{D_4}(M)=\frac{a+d}{2}I.
$$

## Exercises with explanations

1. Is the identity equivariant for rotations and reflections? **Yes: it commutes with both.**
2. Can reflection residual remain nonzero after rotation projection? **Yes, as in the default example.**
3. Is the zero map equivariant? **Yes; that alone guarantees no predictive usefulness.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
