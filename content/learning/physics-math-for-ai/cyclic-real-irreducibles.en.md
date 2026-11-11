## A representation makes group structure linear

A representation ρ maps group elements to invertible linear operators and preserves products. For C4, the cyclic group of order four, let its generator shift a four-sample signal by one: (Px)[n]=x[(n−1) mod 4]. Since P⁴=I, this is a representation on the real vector space R⁴.

An invariant subspace is mapped into itself by every group element. An irreducible representation is nonzero and has no proper nonzero invariant subspace over the specified scalar field. That final qualification matters: irreducibility over R can differ from irreducibility after complexification.

## Find an orthonormal decomposition

Use four orthonormal basis vectors:

$$
u_0=\tfrac12(1,1,1,1),\quad
u_2=\tfrac12(1,-1,1,-1),\quad
u_c=\tfrac1{\sqrt2}(1,0,-1,0),\quad
u_s=\tfrac1{\sqrt2}(0,1,0,-1).
$$

The first basis vector spans the trivial representation. The second spans the alternating representation, whose coefficient changes sign under one shift. The last two span a plane. Their coefficients are a_c=(x₀−x₂)/√2 and a_s=(x₁−x₃)/√2. A shift rotates that pair by π/2:

$$
(a_c,a_s)\longmapsto(-a_s,a_c),\qquad
\mathbb R^4=V_0\oplus V_2\oplus V_{cs}.
$$

The blue and orange vectors in the plot are these plane coefficients before and after shifting. They are not the time samples themselves. The table reconstructs shifted samples from all three representation blocks and compares them with the direct cyclic shift.

## Why the plane is irreducible over the reals

A proper nonzero real subspace of a two-dimensional plane would be a line. An invariant line would provide a real eigenvector of the quarter-turn matrix J. But its characteristic polynomial λ²+1 has no real root. Therefore the plane is irreducible over R.

Over C, J has eigenvalues +i and −i, with one-dimensional eigenlines. The complexified plane splits into two characters. This is consistent with complex representations of the finite Abelian group C4 being sums of one-dimensional characters; it does not contradict the real two-dimensional block.

## Energy and a worked example

Orthonormality gives Σxₙ²=a₀²+a₂²+a_c²+a_s². For x=(1,0,0,0), the scalar coefficients are each 0.5 and the plane coefficients are (1/√2,0). Energies are 1/4,1/4,1/2, totaling one. After one shift the plane becomes (0,1/√2), the alternating coefficient becomes −0.5, and reconstruction gives (0,1,0,0).

## Construct invariant complements and projectors

A finite real representation can be made orthogonal by averaging any positive-definite inner product over the group. The average stays positive definite, and reindexing proves invariance. If a subspace is invariant, its orthogonal complement in this averaged metric is invariant too. Induction on dimension gives complete reducibility. Division by the group order is essential; the same proof fails in a field whose characteristic divides that order.

In the regular C4 example the original Euclidean metric is already invariant. The projectors below isolate the constant, alternating and remaining plane blocks:

$$
E_0=\frac{I+P+P^2+P^3}{4},\quad
E_2=\frac{I-P+P^2-P^3}{4},\quad
E_{cs}=I-E_0-E_2.
$$

Their squares equal themselves, pairwise products vanish, and their sum is identity. Their traces are their ranks, 1,1,2. This constructs the decomposition without choosing sine/cosine basis vectors first. On the plane, commuting operators have form aI+bJ, a copy of the complex numbers as a real algebra. Real irreducibility therefore does not force every commuting operator to be a real scalar. Over an algebraically closed field, the familiar scalar version of Schur's lemma has different hypotheses.

| Projector | Rank | Action of one shift on its image |
|---|---:|---|
| E₀ | 1 | +1 |
| E₂ | 1 | -1 |
| Ecs | 2 | Quarter turn |

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
n=4
I=[[F(i==j) for j in range(n)] for i in range(n)]
P=[[F(i==(j+1)%n) for j in range(n)] for i in range(n)]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
powers=[I,P]
for _ in range(2): powers.append(mm(powers[-1],P))
E0=[[sum(M[i][j] for M in powers)/4 for j in range(n)] for i in range(n)]
E2=[[sum((-1)**k*M[i][j] for k,M in enumerate(powers))/4 for j in range(n)] for i in range(n)]
Ecs=[[I[i][j]-E0[i][j]-E2[i][j] for j in range(n)] for i in range(n)]
for a,A in enumerate([E0,E2,Ecs]):
    assert mm(A,A)==A
    for b,B in enumerate([E0,E2,Ecs]):
        if a!=b: assert all(x==0 for row in mm(A,B) for x in row)
    print('rank:', sum(A[i][i] for i in range(n)))
```

## Exercise and research connection

For the constant signal (1,1,1,1), predict only a₀=2, with energy four. For (1,−1,1,−1), predict only a₂=2 and a sign flip after one shift. Verify both directly.

Irreducible blocks organize allowable feature types and equivariant linear couplings. Always specify the group, action and scalar field. A representation irreducible for one group can become reducible when restricted to a subgroup; a tensor's double-angle block later provides a concrete example.

Further reading: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
