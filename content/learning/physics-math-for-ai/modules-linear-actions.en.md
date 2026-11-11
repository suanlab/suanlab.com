## From vector spaces to modules

A vector space has field scalars acting on an additive group. Replacing the field by a ring with identity gives a module. Additive and scalar distributive laws remain, but nonzero scalars need not be invertible. The lab uses the free module R² over R=Z/nZ, with n² ordered pairs. This example does not imply that every module is free or has a basis.

$$
r(u+v)=ru+rv,\quad(r+s)v=rv+sv,\quad(rs)v=r(sv),\quad1v=v.
$$

## Compute a ring-linear map

The map T(x,y)=(ax,by) is R-linear. The table enumerates all n² inputs and outputs. For n=6,a=2,b=3, the equation 2x=0 has x=0,3, while 3y=0 has y=0,2,4. Six ordered pairs therefore map to zero. Nonzero coefficients cannot automatically be cancelled or divided out.

$$
|\ker T|=\gcd(n,a)\gcd(n,b),\qquad |\operatorname{im}T|=\frac{n^2}{|\ker T|}.
$$

## Separate dimension and cardinality

Over F5, both diagonal coefficients two and three are units. The map is invertible and its kernel contains only zero, giving kernel dimension zero. The zero map instead has all of F5² as its kernel: twenty-five elements but dimension two. For composite moduli, kernel and image cardinalities are still meaningful, but they are not dimensions over the proposed scalar field. The field-dimension readout remains undefined.

$$
\dim_{\mathbb F_p}\ker T=\#\{\text{zero diagonal coefficients}\}\quad\text{in this diagonal example}.
$$

## Which linear-algebra intuitions survive?

Kernels and images of module homomorphisms are submodules, and quotient constructions remain available. Over general rings, however, submodules need not have vector-space-like complements or bases. A table for the free module R² does not classify all modules. When studying algebraic codes or discrete structured learning, specifying the scalar domain prevents the same matrix notation from concealing different arithmetic.

For a concrete nonfree module, consider Z/2Z over the integer ring Z. Its nonzero element u satisfies 2u=0. In a free integer module, doubling a nonzero vector cannot give zero. Consequently this module cannot have a free basis.

## A representation is a module over a group algebra

A group algebra F[G] consists of formal sums of group elements with coefficients in a field F; multiply using the group law and distributivity. A representation on V extends linearly to an action of F[G]. An invariant subspace is exactly a submodule for this action. This connects the scalar-ring viewpoint to symmetry-constrained layers. It is a separate example from the diagonal residue-module controls above.

$$
\left(\sum_{g\in G}a_g g\right)v=\sum_{g\in G}a_g\rho(g)v.
$$

## When averaging produces an invariant complement

Assume G is finite, V is finite dimensional over F, and the characteristic of F does not divide |G|. Characteristic zero is included. Let W be invariant and choose any linear projection Q onto W by extending a basis of W to a basis of V. Q need not commute with the group. Average its conjugates to repair this defect:

$$
P=\frac1{|G|}\sum_{g\in G}\rho(g)^{-1}Q\rho(g).
$$

Each summand maps into W and acts as the identity on W, because W is invariant under both g and its inverse. Hence P maps into W, restricts to the identity there, and satisfies P²=P. Conjugating P by any group element merely permutes the summands, so P commutes with every action. Consequently ker P is invariant and V is the direct sum of W and ker P: write v=Pv+(v−Pv), and apply P to check both the kernel condition and the zero intersection. This is the splitting argument in Maschke's theorem. It needs no inner product and does not claim that this projection is orthogonal in arbitrary coordinates.

## A characteristic-three obstruction you can enumerate

Now work over F3 with generator J of C3 and invariant line W spanned by (1,0). Since J=I+N with N²=0, cubing gives I+3N=I in this field. The same integer matrix over the reals would not represent C3. Every projection onto W that is the identity on W must have the following form:

$$
J=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad
P_b=\begin{pmatrix}1&b\\0&0\end{pmatrix},\quad
P_bJ-JP_b=\begin{pmatrix}0&1\\0&0\end{pmatrix}\ne0.
$$

| b in F3 | Projection rows | Commutator rows |
|---|---|---|
| 0 | (1,0), (0,0) | (0,1), (0,0) |
| 1 | (1,1), (0,0) | (0,1), (0,0) |
| 2 | (1,2), (0,0) | (0,1), (0,0) |

None commutes with J. If an invariant complement existed, its projection onto W would commute, contradicting the table. Thus the invariant line has no invariant complement. Here I+J+J² is the zero matrix; dividing it by three is impossible, and the raw sum does not act as the identity on W. Failure of the characteristic assumption permits such examples; it does not mean that every representation in characteristic three fails to split.

## Execute the finite obstruction independently

The standard-library code below enumerates all 81 matrices, selects all projections onto W and checks their commutators. Its exhaustive calculation is specific to this two-dimensional F3 example; the general splitting result was proved above.

```python
from itertools import product

p = 3
I, J = (1, 0, 0, 1), (1, 1, 0, 1)

def mul(a, b):
    return tuple(sum(a[2*i+k] * b[2*k+j] for k in range(2)) % p
                 for i in range(2) for j in range(2))

def sub(a, b):
    return tuple((x-y) % p for x, y in zip(a, b))

powers = [I, J, mul(J, J)]
assert mul(powers[-1], J) == I
assert len(set(powers)) == 3
assert tuple(sum(g[i] for g in powers) % p for i in range(4)) == (0,)*4
projections = [a for a in product(range(p), repeat=4)
               if a[0] == 1 and a[2:] == (0, 0) and mul(a, a) == a]
assert projections == [(1, b, 0, 0) for b in range(p)]
for a in projections:
    commutator = sub(mul(a, J), mul(J, a))
    assert commutator == (0, 1, 0, 0)
    print(a, commutator)
fixed = [(x, y) for x, y in product(range(p), repeat=2)
         if ((x+y) % p, y) == (x, y)]
assert fixed == [(0, 0), (1, 0), (2, 0)]
print("all matrices:", p**4, "projections:", len(projections), "fixed:", fixed)
```

## Further questions and research use

1. Why does an invariant complement yield a commuting projection? **Both summands are preserved, so applying the group before or after the projection has the same effect.**
2. Is the zero average an invariant complement? **It is a commuting operator, but it is not the identity on W and therefore cannot split off W.**
3. Can a real-valued symmetry layer use this averaging argument? **For a finite group, yes, once its matrices actually form a representation. Numerical training accuracy and the task's symmetry assumption still need separate evaluation.**

For the averaging proof and its characteristic hypothesis, see [Etingof et al., §3.1, Theorem 3.1](https://math.mit.edu/~etingof/replect.pdf). For invariant nonlinear maps, continue to [algebraic representations](../algebraic-representations-scope/).

## Exercises with explanations

1. What is two times three in Z6? **Zero: a nonzero input is annihilated.**
2. What is the default image size? **Thirty-six divided by six, or six.**
3. What is the kernel dimension for the zero map on F5²? **Two; its cardinality is twenty-five.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
