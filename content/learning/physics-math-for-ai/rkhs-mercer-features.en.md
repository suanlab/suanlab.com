## Kernels connect feature inner products and function spaces

A PSD kernel requires a nonnegative quadratic form for every finite input set and coefficient vector. Positive entries alone do not suffice. An RKHS is a Hilbert function space with continuous point evaluation, represented by kernel section kₓ.

$$
f(x)=\langle f,k_x\rangle_H,\qquad \|k_x\|_H^2=k(x,x).
$$

Consequently |f(x)|≤‖f‖H√k(x,x). Unlike general L2, stable evaluation is built into this space.

## Calculate everything with two features

For kρ(x,z)=1+ρxz on [−1,1], positive rho gives features (1,√ρx). Writing f=a+bx uses coefficients (a,b/√ρ):

$$
\|a+bx\|_H^2=a^2+b^2/\rho.
$$

At rho=1, f=1+x and query .5, squared norm is two, evaluation norm √1.25 and reproduction gives 1.5. Changing the kernel changes the norm of the same function. At rho=0 only constants remain. A sloped target lies outside the space: displayed ∞ is extended notation, not a legitimate finite RKHS norm; reproduction is undefined.

## Mercer theory depends on a measure

Continuous symmetric PSD kernels on compact domains, with suitable finite Borel measures, admit integral-operator spectral descriptions. Fix uniform probability measure dx/2. Orthonormal functions e₀=1 and e₁=√3x have eigenvalues one and rho/3:

$$
k(x,z)=1\,e_0(x)e_0(z)+\frac\rho3e_1(x)e_1(z).
$$

The three-point Gram matrix at −1,0,1 instead has eigenvalues 3,2ρ,0. Finite Gram and integral spectra differ because sample count and weights differ.

## Read the curves correctly

Blue is a kernel section and orange the target. Reproduction concerns their **RKHS inner product**, not equality of their heights. Kernel scaling encodes which functions have small norm. This two-feature space is finite dimensional; an RBF RKHS need not be.

## Construct the space before using reproduction

For a real symmetric PSD kernel, start with finite sums f=∑a_i k(·,x_i) and define their inner product by ∑a_i b_j k(x_i,z_j). The Gram quadratic form is a seminorm. Cauchy–Schwarz shows that a zero-norm vector has zero value at every point, so quotienting out null vectors is consistent with evaluation. Complete this space. The bound |f(x)|≤‖f‖√k(x,x) extends each evaluation continuously to the completion and gives its kernel section as Riesz representative. This is the RKHS construction; entrywise positivity would not justify it.

For the two-feature kernel at ρ>0, 1 and x are independent as functions on the full interval. Thus the coefficients of a+bx are unique and the norm formula follows from the Euclidean feature norm. When ρ tends to zero, the norms of a fixed sloped function diverge even though its ordinary values stay bounded. At ρ=0, membership changes rather than assigning that function a very large finite norm.

## State the measure's support in a Mercer reconstruction

A useful sufficient setting is a compact metric domain, continuous symmetric PSD k and a finite Borel measure with full support. The compact positive integral operator admits the usual Mercer kernel expansion with continuous representatives. The measure fixes the operator and its eigenvalues. Eigenfunctions form a basis of the closure of its range, with a kernel complement as needed; they need not span all L² through positive eigenvalues.

For k=1+xz, compare four measures. The code verifies the finite Gram eigenvectors with rational arithmetic and the integral moments ∫x=0, ∫x²=1/3 for dx/2.

| Measure | Positive integral eigenvalues | Scope |
|---|---|---|
| dx/2 on [−1,1] | 1, 1/3 | Full interval support |
| dx on [−1,1] | 2, 2/3 | Doubled total mass |
| Uniform on −1,0,1 | 1, 2/3 | Weighted Gram on these points |
| δ₀ | 1 | Sees constants only |

With δ₀, the x feature is zero in L²(δ₀). Its positive-eigenvalue reconstruction gives 1 and cannot recover k(1,1)=2 off the measure's support. An L² identity is not automatically a pointwise identity on a larger domain.

For very small positive ρ, the live norm calculation scales the slope by √ρ before squaring. At ρ=10⁻³⁰⁰,b=10⁻²⁰⁰,a=0, its squared norm is 10⁻¹⁰⁰. A finite norm outside binary64 range is shown as — with a numeric-range note; ∞ remains reserved here for a target outside the RKHS. Exact zero coefficients still have zero norm.

**Exercise:** Can an unweighted Gram eigenvalue be called a Mercer eigenvalue? **Explanation:** Only after specifying the empirical measure's normalization. The three-point integral operator uses G/3.

```python
from fractions import Fraction as F

rho=F(1)
def moment(power): return F(1-(-1)**(power+1),2*(power+1))
assert moment(0)==1 and moment(1)==0 and moment(2)==F(1,3)
nodes=[F(-1),F(0),F(1)]
G=[[1+rho*x*z for z in nodes] for x in nodes]
for vector,eigenvalue in [([1,1,1],F(3)),([-1,0,1],2*rho),([1,-2,1],F(0))]:
    assert [sum(a*b for a,b in zip(row,vector)) for row in G]==[eigenvalue*b for b in vector]
print('dx/2 integral eigenvalues:',F(1),rho/3)
print('dx integral eigenvalues:',F(2),2*rho/3)
print('uniform 3-point integral eigenvalues:',F(1),2*rho/3,F(0))
# Under delta_0 the x feature is zero almost everywhere for that measure.
print('delta_0 positive integral eigenvalue:',F(1))
assert 1+rho*F(1)*F(1)!=1  # Its eigen-expansion cannot recover k(1,1) off support.
```

## Exercises with explanations

1. Squared norm of x at rho=.5? **Two**, since slope cost is 1/rho.
2. Does a zero Gram eigenvalue invalidate the kernel? **No.** PSD allows singular matrices.
3. Replace dx/2 by dx: what happens to integral eigenvalues? **They double**, while the kernel formula remains unchanged.

Further reading: [MIT 9.520, RKHS and Mercer](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/resources/class14/).
