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

## Exercises with explanations

1. What is two times three in Z6? **Zero: a nonzero input is annihilated.**
2. What is the default image size? **Thirty-six divided by six, or six.**
3. What is the kernel dimension for the zero map on F5²? **Two; its cardinality is twenty-five.**

Further reading: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
