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

## Why this average is the nearest equivariant matrix

Solving NJ=JN entry by entry gives the C4 commutant N=[[α,β],[−β,α]]. Requiring NS=SN as well forces β=0, leaving the D4 commutant αI. These are linear subspaces of the real matrix space. For the orthogonal representations used here, conjugation preserves the Frobenius inner product. The adjoint of conjugation by g is conjugation by g⁻¹, and inversion permutes the group. Consequently the group average is self-adjoint. Reindexing the average shows that its output commutes with the group; averaging such an output leaves it unchanged. Thus it is an idempotent, self-adjoint operator, hence an orthogonal projection onto the commutant.

For any equivariant N, the residual M−P(M) is perpendicular to P(M)−N. Expanding the squared distance proves the unique nearest-matrix property:

$$
\|M-N\|_F^2=\|M-P(M)\|_F^2+\|P(M)-N\|_F^2.
$$

| Required group | Projection of the default M | Squared distance from M |
|---|---|---|
| C4 | (3/2,1/2), (−1/2,3/2) | 1 |
| D4 | (3/2,0), (0,3/2) | 3/2 |

The additional symmetry changes both the permitted family and the smallest adjustment. For nonorthogonal representations, conjugation averaging can still be an algebraic projection but need not minimize the ordinary Frobenius distance. For finite groups the average also requires division by |G|; see the [module characteristic counterexample](../modules-linear-actions/).

## A commuting weight does not rescue an incompatible activation

Take the identity weight, which commutes with every action, followed by coordinatewise ReLU. At v=(1,−1), applying J then ReLU gives (1,1); applying ReLU then J gives (0,1). Therefore this nonlinear layer is not C4 equivariant. The issue is visible without any training error or numerical approximation.

$$
\operatorname{ReLU}(Jv)=(1,1)\ne(0,1)=J\operatorname{ReLU}(v).
$$

For an affine layer Mv+b, equivariance further requires that the bias be fixed by the output action. This natural two-dimensional C4 action has only the zero fixed vector, so b must be zero. Other representations can contain nonzero fixed vectors; zero bias is not a universal requirement. Compositions preserve equivariance only when each constituent map respects its specified input and output actions.

## Build a nonlinear equivariant map with an invariant gate

An invariant scalar can gate an equivariant vector. With P equal to the C4 matrix projection, define the following bounded radial factor:

$$
F(v)=\frac{P(M)v}{1+\|v\|^2},\qquad
F(gv)=\frac{P(M)gv}{1+\|gv\|^2}=gF(v)\quad(g\in C_4).
$$

The proof uses both commutation of P(M) and orthogonality of g. For the default matrix and v=(1,2), the output is (5/12,5/12). This construction is C4 equivariant but need not respect reflection; using the D4 projection makes the same construction D4 equivariant. The scalar gate may be replaced by another function of the invariant squared norm. This is a concrete layer-design principle, not evidence that a particular trained model will generalize.

## Check projections and nonlinear maps with exact fractions

The standard-library code independently averages all four or eight conjugates, verifies the distance identity, reproduces the ReLU failure and checks the radial construction on three inputs under every group element. The displayed matrix lab still computes the original projections; the nonlinear example below is a separate executable exercise.

```python
from fractions import Fraction as Q

I, J, S = (1, 0, 0, 1), (0, -1, 1, 0), (1, 0, 0, -1)
M = (2, 1, 0, 1)

def mul(a, b):
    return tuple(sum(a[2*i+k]*b[2*k+j] for k in range(2))
                 for i in range(2) for j in range(2))

def transpose(a):
    return (a[0], a[2], a[1], a[3])

def apply(a, v):
    return tuple(sum(a[2*i+j]*v[j] for j in range(2)) for i in range(2))

def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))

def norm2(a):
    return sum(x*x for x in a)

def project(a, group):
    terms = [mul(mul(transpose(g), a), g) for g in group]
    return tuple(Q(sum(t[i] for t in terms), len(group)) for i in range(4))

rotations = [I]
for _ in range(3):
    rotations.append(mul(rotations[-1], J))
dihedral = rotations + [mul(g, S) for g in rotations]
C, D = project(M, rotations), project(M, dihedral)
assert C == (Q(3, 2), Q(1, 2), Q(-1, 2), Q(3, 2))
assert D == (Q(3, 2), 0, 0, Q(3, 2))
assert norm2(sub(M, C)) == 1 and norm2(sub(M, D)) == Q(3, 2)
for group, P in [(rotations, C), (dihedral, D)]:
    assert project(P, group) == P
    assert all(mul(P, g) == mul(g, P) for g in group)
    assert norm2(M) == norm2(P) + norm2(sub(M, P))
for alpha, beta in [(0, 0), (2, 1), (Q(1, 3), Q(-2, 5))]:
    N = (alpha, beta, -beta, alpha)
    assert norm2(sub(M, N)) == norm2(sub(M, C)) + norm2(sub(C, N))
relu = lambda v: tuple(max(0, x) for x in v)
v = (1, -1)
assert relu(apply(J, v)) == (1, 1)
assert apply(J, relu(v)) == (0, 1)

def radial(a, v):
    return tuple(Q(x) / (1+norm2(v)) for x in apply(a, v))

for group, P in [(rotations, C), (dihedral, D)]:
    for v in [(1, 2), (1, -1), (0, 0)]:
        for g in group:
            assert radial(P, apply(g, v)) == apply(g, radial(P, v))
v = (1, 2)
assert radial(C, v) == (Q(5, 12), Q(5, 12))
assert radial(C, apply(S, v)) != apply(S, radial(C, v))
print("projections:", C, D, "squared distances:", 1, Q(3, 2))
print("radial output:", radial(C, v), "ReLU counterexample:", (1, 1), (0, 1))
```

## Further questions and research use

1. Why is P(M) uniquely nearest? **The second nonnegative term in the distance identity vanishes exactly when N=P(M).**
2. Does coordinatewise ReLU always break equivariance? **No; it commutes with coordinate permutations, for example. The action must be checked rather than inferred from the activation name.**
3. Does radial gating prove equivariance for nonorthogonal matrices? **The norm argument used here fails unless the chosen scalar feature is invariant for that action.**

For group representations and finite-group averaging, see [Etingof et al., §§3.1 and 3.6](https://math.mit.edu/~etingof/replect.pdf). The distance and nonlinear examples above are derived directly for this page's real square action.

## Exercises with explanations

1. Is the identity equivariant for rotations and reflections? **Yes: it commutes with both.**
2. Can reflection residual remain nonzero after rotation projection? **Yes, as in the default example.**
3. Is the zero map equivariant? **Yes; that alone guarantees no predictive usefulness.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
