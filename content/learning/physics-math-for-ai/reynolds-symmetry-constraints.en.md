## Symmetry becomes a constraint on allowable linear maps

For identical input and output vector actions, a linear map is equivariant when Wg=gW for every group element. These equations define a linear subspace of matrices, called the commutant of the representation. Its dimension depends on the action and scalar field, not merely on the number of group elements.

A finite-group Reynolds operator constructs an equivariant matrix by averaging conjugates:

$$
P_G(W)=\frac1{|G|}\sum_{g\in G}g^{-1}Wg.
$$

Reindexing the finite sum shows h⁻¹P_G(W)h=P_G(W) for each h. If W already commutes with the group, every term is W, so P_G(W)=W. Applying the operator twice therefore changes nothing. This is an algebraic projection, not iterative training or an approximate penalty method.

## Why the projection is orthogonal here

The group matrices are orthogonal, so conjugation preserves the Frobenius inner product. Averaging these actions projects orthogonally onto their common fixed subspace. Consequently W−P_G(W) is orthogonal to every allowed equivariant matrix, giving a closest equivariant matrix in Frobenius norm.

This argument uses a finite group over the real numbers and an orthogonal representation. Compact groups admit related normalized Haar averages; an arbitrary noncompact group need not have a normalized invariant probability measure. Do not apply the same finite-average formula without checking what group and measure are available.

## Rotations and reflections impose different restrictions

For the planar C4 quarter-turn representation, solving WJ=JW gives W=aI+bJ, a two-dimensional real space. Adding horizontal reflection S also requires WS=SW, which forces b=0. The D4 commutant is therefore the one-dimensional scalar space. There is no contradiction with the real irreducibility of the C4 plane: the scalar-field assumptions behind simplified Schur-lemma statements matter.

## Work the shear example

Take W=[[1,2],[0,1]]. The actual group average gives

$$
P_{C_4}(W)=\begin{pmatrix}1&1\\-1&1\end{pmatrix},\qquad
P_{D_4}(W)=I.
$$

Squared Frobenius distances are two and four. The blue curve is the original unit-circle image, and orange the projected image. The numerical table exposes the matrices, maximum commutator norms over every group element, and idempotence error. Changing from C4 to D4 adds a genuine modeling restriction rather than simply a stricter numerical tolerance.

## Exercise and explained answer

For W=diag(1,−1), predict the D4 projection to be zero because its trace is zero. Verify the collapsed orange image and all commutators. Then explain why an equivariant linear layer followed by arbitrary pointwise ReLU is not automatically SO(2)-equivariant: rotate (1,−1) by π/2 and compare ReLU before and after rotation. The nonlinear outputs differ.

## Research connection

Steerable layers use representation-dependent constraints on kernels and feature couplings. Reynolds averaging provides a transparent finite example. Compatible nonlinearities, pooling, boundary rules and discretization still need their own checks. Projecting one matrix does not certify the full network or establish that the chosen symmetry is appropriate for its labels.

Further reading: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
